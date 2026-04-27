"""Modal backend for low-cost anomaly training and inference.

Deploy with:
    modal deploy backend/modal_anomaly_service.py

Then use the generated base URL with backend/modal_anomaly_client.py.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import modal
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest

APP_NAME = "optimlflow-anomaly"
MODEL_VOLUME_NAME = "optimlflow-anomaly-models"
MODEL_DIR = "/model_store"

MAX_TRAIN_ROWS = 30_000
MAX_PREDICT_ROWS = 5_000
MAX_COLUMNS = 256
MAX_CELLS_TRAIN = 1_500_000
MAX_CELLS_PREDICT = 500_000
MIN_TRAIN_ROWS = 30
TRAIN_COOLDOWN_SECONDS = 900

MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

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

web_app = FastAPI(title="OptiMLFlow Modal Anomaly API", version="1.0.0")

# Per-container in-memory cache to avoid reloading model on every prediction.
_MODEL_CACHE: dict[str, dict[str, Any]] = {}


class TrainRequest(BaseModel):
    X: list[list[float]]
    contamination: float = Field(default=0.05, ge=0.001, le=0.2)
    n_estimators: int = Field(default=64, ge=50, le=100)
    model_id: str = Field(default="default", min_length=1, max_length=64)
    confirm_train: bool = False
    force_retrain: bool = False
    return_serialized_model: bool = True


class PredictRequest(BaseModel):
    X: list[list[float]]
    model_id: str = Field(default="default", min_length=1, max_length=64)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_model_id(model_id: str) -> str:
    model_id = model_id.strip()
    if not MODEL_ID_PATTERN.match(model_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid model_id. Use only letters, numbers, underscore, or hyphen (1-64 chars).",
        )
    return model_id


def _model_path(model_id: str) -> Path:
    return Path(MODEL_DIR) / f"{model_id}.joblib"


def _metadata_path(model_id: str) -> Path:
    return Path(MODEL_DIR) / f"{model_id}.metadata.json"


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
        raise HTTPException(
            status_code=413,
            detail=f"{context}: row limit exceeded. Max rows is {max_rows}.",
        )

    if cols > MAX_COLUMNS:
        raise HTTPException(
            status_code=413,
            detail=f"{context}: column limit exceeded. Max columns is {MAX_COLUMNS}.",
        )

    if rows * cols > max_cells:
        raise HTTPException(
            status_code=413,
            detail=f"{context}: payload too large. Max cells is {max_cells}.",
        )

    if not np.isfinite(matrix).all():
        raise HTTPException(status_code=400, detail=f"{context}: X cannot contain NaN or infinity.")

    return matrix


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


def _hash_matrix(matrix: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.tobytes())
    return digest.hexdigest()


def _parse_iso_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _guard_retrain(metadata: dict[str, Any] | None, dataset_hash: str, force_retrain: bool) -> None:
    if force_retrain or not metadata:
        return

    if metadata.get("dataset_hash") == dataset_hash:
        raise HTTPException(
            status_code=409,
            detail="This exact dataset was already trained. Set force_retrain=true to override.",
        )

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
            detail="Training already in progress for this model_id. Try again later.",
        ) from exc


def _release_training_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _load_model_cached(model_id: str) -> tuple[Any, dict[str, Any]]:
    model_path = _model_path(model_id)
    meta = _read_metadata(model_id)

    if meta is None or not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found. Trigger /train first.")

    mtime = model_path.stat().st_mtime
    cached = _MODEL_CACHE.get(model_id)
    if cached and cached.get("mtime") == mtime:
        return cached["model"], cached["metadata"]

    model = joblib.load(model_path)
    _MODEL_CACHE[model_id] = {"model": model, "metadata": meta, "mtime": mtime}
    return model, meta


@web_app.get("/health")
def health(model_id: str = "default") -> dict[str, Any]:
    model_id = _safe_model_id(model_id)
    _volume_reload()

    exists = _model_path(model_id).exists() and _metadata_path(model_id).exists()
    metadata = _read_metadata(model_id) if exists else None
    return {
        "status": "ok",
        "model_id": model_id,
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

    model_id = _safe_model_id(request.model_id)
    matrix = _validate_matrix(
        request.X,
        max_rows=MAX_TRAIN_ROWS,
        max_cells=MAX_CELLS_TRAIN,
        context="train",
    )
    if matrix.shape[0] < MIN_TRAIN_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"train: at least {MIN_TRAIN_ROWS} rows are required.",
        )

    _volume_reload()
    lock_file = _lock_path(model_id)
    _acquire_training_lock(lock_file)

    try:
        dataset_hash = _hash_matrix(matrix)
        existing = _read_metadata(model_id)
        _guard_retrain(existing, dataset_hash, request.force_retrain)

        started = time.perf_counter()
        model = IsolationForest(
            contamination=request.contamination,
            n_estimators=request.n_estimators,
            random_state=42,
            n_jobs=1,
        )
        model.fit(matrix)
        train_seconds = round(time.perf_counter() - started, 3)

        buffer = io.BytesIO()
        joblib.dump(model, buffer, compress=3)
        serialized = buffer.getvalue()

        model_path = _model_path(model_id)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(serialized)

        metadata = {
            "model_id": model_id,
            "algorithm": "IsolationForest",
            "n_features": int(matrix.shape[1]),
            "n_rows": int(matrix.shape[0]),
            "contamination": float(request.contamination),
            "n_estimators": int(request.n_estimators),
            "dataset_hash": dataset_hash,
            "artifact_bytes": len(serialized),
            "trained_at": _utc_now_iso(),
            "train_seconds": train_seconds,
            "cpu": 1,
        }
        _write_metadata(model_id, metadata)
        _volume_commit()

        _MODEL_CACHE.pop(model_id, None)

        response = {
            "status": "trained",
            "model_id": model_id,
            "model_path": str(model_path),
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
    matrix = _validate_matrix(
        request.X,
        max_rows=MAX_PREDICT_ROWS,
        max_cells=MAX_CELLS_PREDICT,
        context="predict",
    )

    _volume_reload()
    model, metadata = _load_model_cached(model_id)

    expected_features = int(metadata.get("n_features", 0))
    if matrix.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=(
                "predict: feature mismatch. "
                f"Expected {expected_features} columns, got {matrix.shape[1]}."
            ),
        )

    labels = model.predict(matrix)
    scores = -model.score_samples(matrix)

    return {
        "status": "success",
        "model_id": model_id,
        "n_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "anomaly_labels": [int(v) for v in labels.tolist()],
        "anomaly_scores": [float(v) for v in scores.tolist()],
    }


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
    print("Deploy with: modal deploy backend/modal_anomaly_service.py")
    print("After deploy, use the generated URL with backend/modal_anomaly_client.py")
