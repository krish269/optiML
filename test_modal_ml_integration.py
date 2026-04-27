"""Quick smoke script for generic Modal ML backend.

Required env vars:
- MODAL_ML_BASE_URL
- MODAL_ML_API_KEY (optional)
"""

from __future__ import annotations

import numpy as np

from backend.modal_ml_client import predict, train_model


def _run_classification() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(220, 10)).astype(np.float32)
    y = (X[:, 0] + 0.6 * X[:, 1] > 0.0).astype(int)

    train_result = train_model(
        X,
        y,
        task_type="classification",
        model_type="logistic_regression",
        confirm_train=True,
    )
    model_id = str(train_result["model_id"])

    pred_result = predict(model_id, X[:20])
    print("[classification] model_id:", model_id)
    print("[classification] predictions:", len(pred_result.get("predictions", [])))



def _run_regression() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(240, 8)).astype(np.float32)
    y = (2.4 * X[:, 0] - 1.1 * X[:, 3] + 0.3 * X[:, 6]).astype(np.float32)

    train_result = train_model(
        X,
        y,
        task_type="regression",
        model_type="linear_regression",
        confirm_train=True,
    )
    model_id = str(train_result["model_id"])

    pred_result = predict(model_id, X[:20])
    print("[regression] model_id:", model_id)
    print("[regression] predictions:", len(pred_result.get("predictions", [])))



def _run_anomaly() -> None:
    rng = np.random.default_rng(99)
    X = rng.normal(0, 1, size=(260, 6)).astype(np.float32)
    X[:12] += 5.0

    train_result = train_model(
        X,
        y=None,
        task_type="anomaly",
        model_type="isolation_forest",
        contamination=0.05,
        confirm_train=True,
    )
    model_id = str(train_result["model_id"])

    pred_result = predict(model_id, X[:20])
    print("[anomaly] model_id:", model_id)
    print("[anomaly] predictions:", len(pred_result.get("predictions", [])))
    print("[anomaly] scores:", len(pred_result.get("anomaly_scores", [])))



def main() -> None:
    _run_classification()
    _run_regression()
    _run_anomaly()


if __name__ == "__main__":
    main()
