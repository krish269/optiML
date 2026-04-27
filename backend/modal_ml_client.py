"""Generic client for Modal ML backend.

Exposes both a class-based API and module-level convenience functions:
- train_model(X, y, task_type, model_type)
- predict(model_id, X)
"""

from __future__ import annotations

import os
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import requests

TaskType = Literal["classification", "regression", "anomaly"]
MatrixLike = np.ndarray | pd.DataFrame | Sequence[Sequence[float]]
TargetLike = Sequence[float | int | str | bool | None] | np.ndarray | pd.Series | None

MAX_TRAIN_ROWS = 30_000
MAX_PREDICT_ROWS = 5_000
MAX_COLUMNS = 256
MAX_TRAIN_CELLS = 1_500_000
MAX_PREDICT_CELLS = 500_000


class ModalMLClientError(RuntimeError):
    """Raised when Modal ML API calls fail."""


class ModalMLClient:
    """HTTP client for generic Modal ML /train and /predict endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float = 45.0,
        api_key: str | None = None,
    ) -> None:
        resolved_url = (base_url or os.getenv("MODAL_ML_BASE_URL", "")).strip()
        if not resolved_url:
            raise ValueError("Modal ML base_url is required. Set MODAL_ML_BASE_URL or pass base_url.")

        self.base_url = resolved_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        resolved_key = (api_key or os.getenv("MODAL_ML_API_KEY", "")).strip()
        if resolved_key:
            self._session.headers.update({"Authorization": f"Bearer {resolved_key}"})

    def train_model(
        self,
        X: MatrixLike,
        y: TargetLike,
        task_type: TaskType,
        model_type: str | None = None,
        *,
        model_id: str | None = None,
        contamination: float = 0.05,
        confirm_train: bool = True,
        force_retrain: bool = False,
        return_serialized_model: bool = False,
    ) -> dict[str, Any]:
        matrix = self._to_numeric_matrix(
            X,
            max_rows=MAX_TRAIN_ROWS,
            max_cells=MAX_TRAIN_CELLS,
            context="train",
        )

        payload: dict[str, Any] = {
            "X": matrix.tolist(),
            "y": self._to_target_list(y),
            "task_type": task_type,
            "model_type": model_type,
            "model_id": model_id,
            "contamination": float(contamination),
            "confirm_train": bool(confirm_train),
            "force_retrain": bool(force_retrain),
            "return_serialized_model": bool(return_serialized_model),
        }
        return self._post_json("/train", payload, action="train")

    def predict(self, model_id: str, X: MatrixLike) -> dict[str, Any]:
        matrix = self._to_numeric_matrix(
            X,
            max_rows=MAX_PREDICT_ROWS,
            max_cells=MAX_PREDICT_CELLS,
            context="predict",
        )
        payload = {
            "model_id": model_id,
            "X": matrix.tolist(),
        }
        return self._post_json("/predict", payload, action="predict")

    def predict_in_batches(
        self,
        model_id: str,
        X: MatrixLike,
        *,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        if batch_size < 50:
            raise ValueError("batch_size must be >= 50 to avoid expensive tiny calls.")

        matrix = self._to_numeric_matrix(
            X,
            max_rows=10_000_000,
            max_cells=250_000_000,
            context="predict_in_batches",
        )

        n_rows, n_features = matrix.shape
        if n_rows <= batch_size:
            return self.predict(model_id, matrix)

        merged_predictions: list[Any] = []
        merged_probabilities: list[Any] | None = None
        merged_scores: list[float] | None = None
        task_type: str | None = None

        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            result = self.predict(model_id, matrix[start:stop])

            if task_type is None:
                task_type = str(result.get("task_type", ""))

            merged_predictions.extend(result.get("predictions", []))

            probs = result.get("probabilities")
            if probs is not None:
                if merged_probabilities is None:
                    merged_probabilities = []
                merged_probabilities.extend(probs)

            scores = result.get("anomaly_scores")
            if scores is not None:
                if merged_scores is None:
                    merged_scores = []
                merged_scores.extend([float(v) for v in scores])

        response: dict[str, Any] = {
            "status": "success",
            "model_id": model_id,
            "task_type": task_type,
            "n_rows": int(n_rows),
            "n_features": int(n_features),
            "predictions": merged_predictions,
        }
        if merged_probabilities is not None:
            response["probabilities"] = merged_probabilities
        if merged_scores is not None:
            response["anomaly_scores"] = merged_scores
        return response

    def _post_json(self, path: str, payload: dict[str, Any], *, action: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.post(url, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise ModalMLClientError(f"Modal ML {action} call failed: {exc}") from exc

        if response.status_code >= 400:
            detail = self._extract_error_detail(response)
            raise ModalMLClientError(
                f"Modal ML {action} failed ({response.status_code}): {detail}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ModalMLClientError(f"Modal ML {action} returned non-JSON response.") from exc

    @staticmethod
    def _extract_error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or "Unknown error"

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail:
                return str(detail)
            return str(payload)

        return str(payload)

    @staticmethod
    def _to_target_list(y: TargetLike) -> list[Any] | None:
        if y is None:
            return None

        if isinstance(y, pd.Series):
            values = y.tolist()
        elif isinstance(y, np.ndarray):
            values = y.tolist()
        else:
            values = list(y)

        normalized: list[Any] = []
        for value in values:
            if isinstance(value, (np.floating, float)):
                normalized.append(float(value))
            elif isinstance(value, (np.integer, int)):
                normalized.append(int(value))
            elif isinstance(value, (np.bool_, bool)):
                normalized.append(bool(value))
            elif value is None:
                normalized.append(None)
            else:
                normalized.append(str(value))
        return normalized

    @staticmethod
    def _to_numeric_matrix(
        data: MatrixLike,
        *,
        max_rows: int,
        max_cells: int,
        context: str,
    ) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            matrix = data.to_numpy(dtype=np.float32, copy=False)
        else:
            try:
                matrix = np.asarray(data, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                raise ValueError("Input must be a numeric matrix.") from exc

        if matrix.ndim != 2:
            raise ValueError("Input must be 2-dimensional (rows x features).")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Input matrix cannot be empty.")

        rows, cols = matrix.shape
        if rows > max_rows:
            raise ValueError(f"{context}: row limit exceeded. Max rows is {max_rows}.")
        if cols > MAX_COLUMNS:
            raise ValueError(f"{context}: column limit exceeded. Max columns is {MAX_COLUMNS}.")
        if rows * cols > max_cells:
            raise ValueError(f"{context}: payload too large. Max cells is {max_cells}.")
        if not np.isfinite(matrix).all():
            raise ValueError("Input matrix cannot contain NaN or infinity.")

        return matrix


def train_model(
    X: MatrixLike,
    y: TargetLike,
    task_type: TaskType,
    model_type: str | None = None,
    *,
    base_url: str | None = None,
    timeout_seconds: float = 45.0,
    api_key: str | None = None,
    model_id: str | None = None,
    contamination: float = 0.05,
    confirm_train: bool = True,
    force_retrain: bool = False,
    return_serialized_model: bool = False,
) -> dict[str, Any]:
    client = ModalMLClient(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
    )
    return client.train_model(
        X=X,
        y=y,
        task_type=task_type,
        model_type=model_type,
        model_id=model_id,
        contamination=contamination,
        confirm_train=confirm_train,
        force_retrain=force_retrain,
        return_serialized_model=return_serialized_model,
    )


def predict(
    model_id: str,
    X: MatrixLike,
    *,
    base_url: str | None = None,
    timeout_seconds: float = 45.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    client = ModalMLClient(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
    )
    return client.predict(model_id=model_id, X=X)
