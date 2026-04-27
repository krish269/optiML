"""Generic low-cost Modal backend for ML training and inference.

Supported tasks:
- classification
- regression
- anomaly

Deploy with:
    modal deploy backend/modal_ml_service.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import modal
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

APP_NAME = "optimlflow-ml-generic"
MODEL_VOLUME_NAME = "optimlflow-ml-models"
MODEL_DIR = "/model_store_ml"

MAX_TRAIN_ROWS = 30_000
MAX_PREDICT_ROWS = 5_000
MAX_COLUMNS = 256
MAX_CELLS_TRAIN = 1_500_000
MAX_CELLS_PREDICT = 500_000
MIN_TRAIN_ROWS = 20
TRAIN_COOLDOWN_SECONDS = 600

MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

TaskType = Literal["classification", "regression", "anomaly"]

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.116.1",
        "pydantic==2.11.4",
        "numpy==2.3.3",
        "scikit-learn==1.7.2",
        "joblib==1.5.2",
    )
)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

web_app = FastAPI(title="OptiMLFlow Modal Generic ML API", version="1.0.0")

# Per-container model cache to reduce repeated disk reads during inference bursts.
_MODEL_CACHE: dict[str, dict[str, Any]] = {}


class TrainRequest(BaseModel):
    X: list[list[float]]
    y: list[float | int | str | bool | None] | None = None
    task_type: TaskType
    model_type: str | None = None
    model_id: str | None = None
    contamination: float = Field(default=0.05, ge=0.001, le=0.2)
    confirm_train: bool = False
    force_retrain: bool = False
    return_serialized_model: bool = False


class PredictRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=64)
    X: list[list[float]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_model_id(model_id: str) -> str:
    candidate = model_id.strip()
    if not MODEL_ID_PATTERN.match(candidate):
        raise HTTPException(
            status_code=400,
            detail="Invalid model_id. Use letters, numbers, underscore, or hyphen (1-64 chars).",
        )
    return candidate


def _new_model_id() -> str:
    return str(uuid.uuid4())


def _model_path(model_id: str) -> Path:
    return Path(MODEL_DIR) / f"{model_id}.joblib"


def _metadata_path(model_id: str) -> Path:
    return Path(MODEL_DIR) / f"{model_id}.metadata.json"


def _index_path() -> Path:
    return Path(MODEL_DIR) / "train_index.json"


def _lock_path(model_id: str) -> Path:
    return Path(MODEL_DIR) / f"{model_id}.train.lock"


def _volume_reload() -> None:
    reload_fn = getattr(model_volume, "reload", None)
    if callable(reload_fn):
        reload_fn()


def _volume_commit() -> None:
    commit_fn = getattr(model_volume, "commit", None)
    if callable(commit_fn):
        commit_fn()


def _validate_matrix(
    raw: list[list[float]],
    *,
    max_rows: int,
    max_cells: int,
    context: str,
) -> np.ndarray:
    try:
        matrix = np.asarray(raw, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"{context}: X must contain numeric values only.") from exc

    if matrix.ndim != 2:
        raise HTTPException(status_code=400, detail=f"{context}: X must be a 2D matrix.")

    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        raise HTTPException(status_code=400, detail=f"{context}: X cannot be empty.")

    if rows > max_rows:
        raise HTTPException(status_code=413, detail=f"{context}: row limit exceeded ({max_rows}).")

    if cols > MAX_COLUMNS:
        raise HTTPException(status_code=413, detail=f"{context}: column limit exceeded ({MAX_COLUMNS}).")

    if rows * cols > max_cells:
        raise HTTPException(status_code=413, detail=f"{context}: payload too large ({max_cells} cells).")

    if not np.isfinite(matrix).all():
        raise HTTPException(status_code=400, detail=f"{context}: X cannot contain NaN or infinity.")

    return matrix


def _validate_target(
    y: list[float | int | str | bool | None] | None,
    *,
    rows: int,
    task_type: TaskType,
) -> np.ndarray | None:
    if task_type == "anomaly":
        return None

    if y is None:
        raise HTTPException(status_code=400, detail=f"train: y is required for {task_type}.")

    if len(y) != rows:
        raise HTTPException(status_code=400, detail="train: y length must match X rows.")

    if task_type == "classification":
        y_arr = np.asarray(y, dtype=object)
        y_norm = np.asarray(["" if v is None else str(v) for v in y_arr], dtype=object)
        unique_classes = np.unique(y_norm)
        if unique_classes.shape[0] < 2:
            raise HTTPException(status_code=400, detail="train: classification requires at least 2 classes.")
        return y_norm

    # regression
    try:
        y_arr = np.asarray(y, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="train: regression y must be numeric.") from exc

    if not np.isfinite(y_arr).all():
        raise HTTPException(status_code=400, detail="train: regression y cannot contain NaN or infinity.")

    return y_arr


def _normalize_model_type(task_type: TaskType, model_type: str | None) -> str:
    value = (model_type or "").strip().lower()

    if task_type == "classification":
        aliases = {
            "": "logistic_regression",
            "logistic_regression": "logistic_regression",
            "logistic": "logistic_regression",
            "logreg": "logistic_regression",
            "decision_tree": "decision_tree",
            "dt": "decision_tree",
            "random_forest": "random_forest",
            "rf": "random_forest",
        }
    elif task_type == "regression":
        aliases = {
            "": "linear_regression",
            "linear_regression": "linear_regression",
            "linear": "linear_regression",
            "decision_tree_regressor": "decision_tree_regressor",
            "decision_tree": "decision_tree_regressor",
            "dtr": "decision_tree_regressor",
        }
    else:  # anomaly
        aliases = {
            "": "isolation_forest",
            "isolation_forest": "isolation_forest",
            "iforest": "isolation_forest",
        }

    resolved = aliases.get(value)
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported model_type '{model_type}' for task_type '{task_type}'."
            ),
        )
    return resolved


def _build_model(task_type: TaskType, model_type: str, contamination: float) -> Any:
    if task_type == "classification":
        if model_type == "logistic_regression":
            return LogisticRegression(max_iter=200, solver="lbfgs")
        if model_type == "decision_tree":
            return DecisionTreeClassifier(max_depth=8, min_samples_leaf=2, random_state=42)
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=60,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            )

    if task_type == "regression":
        if model_type == "linear_regression":
            return LinearRegression(n_jobs=1)
        if model_type == "decision_tree_regressor":
            return DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, random_state=42)

    if task_type == "anomaly" and model_type == "isolation_forest":
        return IsolationForest(
            contamination=float(contamination),
            n_estimators=60,
            max_samples="auto",
            random_state=42,
            n_jobs=1,
        )

    raise HTTPException(status_code=400, detail="Unsupported task/model combination.")


def _read_metadata(model_id: str) -> dict[str, Any] | None:
    path = _metadata_path(model_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_metadata(model_id: str, payload: dict[str, Any]) -> None:
    path = _metadata_path(model_id)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _read_index() -> dict[str, str]:
    path = _index_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
    return {}


def _write_index(payload: dict[str, str]) -> None:
    path = _index_path()
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _hash_train_payload(
    *,
    task_type: TaskType,
    model_type: str,
    X: np.ndarray,
    y: np.ndarray | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(task_type.encode("utf-8"))
    digest.update(model_type.encode("utf-8"))
    digest.update(np.asarray(X.shape, dtype=np.int64).tobytes())
    digest.update(X.tobytes())

    if y is None:
        digest.update(b"y:none")
    else:
        y_list = ["" if v is None else str(v) for v in y.tolist()]
        digest.update(json.dumps(y_list, separators=(",", ":")).encode("utf-8"))

    return digest.hexdigest()


def _parse_iso_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _guard_cooldown(metadata: dict[str, Any] | None, force_retrain: bool) -> None:
    if force_retrain or metadata is None:
        return

    trained_at = _parse_iso_ts(metadata.get("trained_at"))
    if trained_at is None:
        return

    elapsed = (datetime.now(timezone.utc) - trained_at).total_seconds()
    if elapsed < TRAIN_COOLDOWN_SECONDS:
        wait_seconds = int(TRAIN_COOLDOWN_SECONDS - elapsed)
        raise HTTPException(
            status_code=429,
            detail=(
                "Training cooldown active to prevent accidental repeated runs. "
                f"Try again in about {wait_seconds} seconds or set force_retrain=true."
            ),
        )


def _acquire_training_lock(lock_file: Path) -> None:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_file.open("x", encoding="utf-8") as fh:
            fh.write(str(time.time()))
    except FileExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail="Training already in progress for this model_id.",
        ) from exc


def _release_training_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _json_safe_predictions(values: np.ndarray) -> list[Any]:
    result: list[Any] = []
    for value in values.tolist():
        if isinstance(value, (np.integer, int)):
            result.append(int(value))
        elif isinstance(value, (np.floating, float)):
            result.append(float(value))
        elif value is None:
            result.append(None)
        else:
            result.append(str(value))
    return result


def _load_model_cached(model_id: str) -> tuple[Any, dict[str, Any]]:
    model_path = _model_path(model_id)
    metadata = _read_metadata(model_id)

    if metadata is None or not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found. Trigger /train first.")

    mtime = model_path.stat().st_mtime
    cached = _MODEL_CACHE.get(model_id)
    if cached and cached.get("mtime") == mtime:
        return cached["model"], cached["metadata"]

    model = joblib.load(model_path)
    _MODEL_CACHE[model_id] = {"model": model, "metadata": metadata, "mtime": mtime}
    return model, metadata


@web_app.get("/health")
def health(model_id: str | None = None) -> dict[str, Any]:
    _volume_reload()

    if model_id is None:
        model_files = list(Path(MODEL_DIR).glob("*.metadata.json"))
        return {
            "status": "ok",
            "models_tracked": len(model_files),
        }

    safe_id = _safe_model_id(model_id)
    exists = _model_path(safe_id).exists() and _metadata_path(safe_id).exists()
    metadata = _read_metadata(safe_id) if exists else None
    return {
        "status": "ok",
        "model_id": safe_id,
        "model_available": exists,
        "metadata": metadata,
    }


@web_app.post("/train")
def train(request: TrainRequest) -> dict[str, Any]:
    if not request.confirm_train:
        raise HTTPException(
            status_code=400,
            detail="confirm_train must be true. This endpoint is manual-trigger only.",
        )

    X = _validate_matrix(
        request.X,
        max_rows=MAX_TRAIN_ROWS,
        max_cells=MAX_CELLS_TRAIN,
        context="train",
    )
    if X.shape[0] < MIN_TRAIN_ROWS:
        raise HTTPException(status_code=400, detail=f"train: at least {MIN_TRAIN_ROWS} rows are required.")

    y = _validate_target(request.y, rows=int(X.shape[0]), task_type=request.task_type)
    resolved_model_type = _normalize_model_type(request.task_type, request.model_type)

    _volume_reload()

    signature = _hash_train_payload(
        task_type=request.task_type,
        model_type=resolved_model_type,
        X=X,
        y=y,
    )

    index = _read_index()
    if not request.force_retrain:
        existing_model_id = index.get(signature)
        if existing_model_id:
            existing_model_id = _safe_model_id(existing_model_id)
            existing_meta = _read_metadata(existing_model_id)
            if existing_meta and _model_path(existing_model_id).exists():
                return {
                    "status": "already_trained",
                    "model_id": existing_model_id,
                    "metadata": existing_meta,
                    "model_artifact_base64": None,
                }

    model_id = _safe_model_id(request.model_id) if request.model_id else _new_model_id()

    existing_model_meta = _read_metadata(model_id)
    if existing_model_meta and not request.force_retrain:
        raise HTTPException(
            status_code=409,
            detail="model_id already exists. Use a new model_id or set force_retrain=true.",
        )

    _guard_cooldown(existing_model_meta, request.force_retrain)

    lock_file = _lock_path(model_id)
    _acquire_training_lock(lock_file)

    try:
        model = _build_model(request.task_type, resolved_model_type, request.contamination)

        started = time.perf_counter()
        if request.task_type == "anomaly":
            model.fit(X)
        else:
            assert y is not None
            model.fit(X, y)
        train_seconds = round(time.perf_counter() - started, 3)

        buffer = io.BytesIO()
        joblib.dump(model, buffer, compress=3)
        serialized = buffer.getvalue()

        model_path = _model_path(model_id)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(serialized)

        metadata: dict[str, Any] = {
            "model_id": model_id,
            "task_type": request.task_type,
            "model_type": resolved_model_type,
            "n_features": int(X.shape[1]),
            "n_rows": int(X.shape[0]),
            "artifact_bytes": len(serialized),
            "payload_signature": signature,
            "trained_at": _utc_now_iso(),
            "train_seconds": train_seconds,
            "cpu": 1,
        }
        if request.task_type == "anomaly":
            metadata["contamination"] = float(request.contamination)
        if request.task_type == "classification" and hasattr(model, "classes_"):
            metadata["class_labels"] = [str(v) for v in model.classes_.tolist()]

        _write_metadata(model_id, metadata)
        index[signature] = model_id
        _write_index(index)
        _volume_commit()

        _MODEL_CACHE.pop(model_id, None)

        response = {
            "status": "trained",
            "model_id": model_id,
            "metadata": metadata,
        }
        if request.return_serialized_model:
            response["model_artifact_base64"] = base64.b64encode(serialized).decode("ascii")
        else:
            response["model_artifact_base64"] = None

        return response
    finally:
        _release_training_lock(lock_file)


@web_app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    model_id = _safe_model_id(request.model_id)
    X = _validate_matrix(
        request.X,
        max_rows=MAX_PREDICT_ROWS,
        max_cells=MAX_CELLS_PREDICT,
        context="predict",
    )

    _volume_reload()
    model, metadata = _load_model_cached(model_id)

    expected_features = int(metadata.get("n_features", 0))
    if X.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=(
                "predict: feature mismatch. "
                f"Expected {expected_features} columns, got {X.shape[1]}."
            ),
        )

    task_type = str(metadata.get("task_type", "")).strip().lower()
    if task_type not in {"classification", "regression", "anomaly"}:
        raise HTTPException(status_code=500, detail="Stored model metadata is invalid.")

    raw_predictions = model.predict(X)
    response: dict[str, Any] = {
        "status": "success",
        "model_id": model_id,
        "task_type": task_type,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "predictions": _json_safe_predictions(np.asarray(raw_predictions)),
    }

    if task_type == "classification" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        response["probabilities"] = np.asarray(probabilities, dtype=np.float64).tolist()
        response["class_labels"] = metadata.get("class_labels")

    if task_type == "anomaly" and hasattr(model, "score_samples"):
        response["anomaly_scores"] = (-model.score_samples(X)).astype(float).tolist()

    return response


@app.function(
    image=image,
    cpu=1,
    memory=768,
    timeout=120,
    volumes={MODEL_DIR: model_volume},
    allow_concurrent_inputs=10,
    scaledown_window=120,
)
@modal.asgi_app()
def api() -> FastAPI:
    os.makedirs(MODEL_DIR, exist_ok=True)
    return web_app


@app.local_entrypoint()
def run_local() -> None:
    print("Deploy with: modal deploy backend/modal_ml_service.py")
