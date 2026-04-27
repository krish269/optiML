"""Client for OptiMLFlow Modal anomaly endpoints.

This module is Streamlit-friendly and accepts numpy arrays, pandas DataFrames,
or plain nested Python lists.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
import requests

MatrixLike = np.ndarray | pd.DataFrame | Sequence[Sequence[float]]

MAX_TRAIN_ROWS = 30_000
MAX_PREDICT_ROWS = 5_000
MAX_COLUMNS = 256
MAX_TRAIN_CELLS = 1_500_000
MAX_PREDICT_CELLS = 500_000


class ModalAnomalyClientError(RuntimeError):
    """Raised when the Modal anomaly API returns an error."""


class ModalAnomalyClient:
    """Minimal HTTP client for /train and /predict endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float = 45.0,
        api_key: str | None = None,
    ) -> None:
        resolved_url = (base_url or os.getenv("MODAL_ANOMALY_BASE_URL", "")).strip()
        if not resolved_url:
            raise ValueError("Modal base_url is required. Set MODAL_ANOMALY_BASE_URL or pass base_url.")

        self.base_url = resolved_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        resolved_api_key = (api_key or os.getenv("MODAL_ANOMALY_API_KEY", "")).strip()
        if resolved_api_key:
            self._session.headers.update({"Authorization": f"Bearer {resolved_api_key}"})

    def train(
        self,
        X: MatrixLike,
        *,
        contamination: float = 0.05,
        n_estimators: int = 64,
        model_id: str = "default",
        confirm_train: bool = True,
        force_retrain: bool = False,
        return_serialized_model: bool = False,
    ) -> dict[str, Any]:
        """Train the remote IsolationForest model once and persist it on Modal."""
        matrix = self._to_numeric_matrix(
            X,
            max_rows=MAX_TRAIN_ROWS,
            max_cells=MAX_TRAIN_CELLS,
            context="train",
        )

        payload = {
            "X": matrix.tolist(),
            "contamination": float(contamination),
            "n_estimators": int(n_estimators),
            "model_id": model_id,
            "confirm_train": bool(confirm_train),
            "force_retrain": bool(force_retrain),
            "return_serialized_model": bool(return_serialized_model),
        }
        return self._post_json("/train", payload, action="train")

    def predict(self, X: MatrixLike, *, model_id: str = "default") -> dict[str, Any]:
        """Run batched inference in a single API call when possible."""
        matrix = self._to_numeric_matrix(
            X,
            max_rows=MAX_PREDICT_ROWS,
            max_cells=MAX_PREDICT_CELLS,
            context="predict",
        )
        payload = {
            "X": matrix.tolist(),
            "model_id": model_id,
        }
        return self._post_json("/predict", payload, action="predict")

    def predict_in_batches(
        self,
        X: MatrixLike,
        *,
        model_id: str = "default",
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Chunk very large local arrays while keeping each request meaningfully batched."""
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
            return self.predict(matrix, model_id=model_id)

        all_labels: list[int] = []
        all_scores: list[float] = []
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            batch_result = self.predict(matrix[start:stop], model_id=model_id)
            all_labels.extend(batch_result.get("anomaly_labels", []))
            all_scores.extend(batch_result.get("anomaly_scores", []))

        return {
            "status": "success",
            "model_id": model_id,
            "n_rows": int(n_rows),
            "n_features": int(n_features),
            "anomaly_labels": all_labels,
            "anomaly_scores": all_scores,
        }

    def _post_json(self, path: str, payload: dict[str, Any], *, action: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.post(url, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise ModalAnomalyClientError(f"Modal {action} call failed: {exc}") from exc

        if response.status_code >= 400:
            detail = self._extract_error_detail(response)
            raise ModalAnomalyClientError(
                f"Modal {action} failed ({response.status_code}): {detail}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ModalAnomalyClientError(f"Modal {action} returned non-JSON response.") from exc

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
