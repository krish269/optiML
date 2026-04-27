"""Quick smoke script for Modal anomaly train/predict integration.

Required env vars:
- MODAL_ANOMALY_BASE_URL
- MODAL_ANOMALY_API_KEY (optional)
"""

from __future__ import annotations

import numpy as np

from backend.modal_anomaly_client import ModalAnomalyClient


def main() -> None:
    rng = np.random.default_rng(42)

    # Assume preprocessing is already completed. Keep matrices numeric and clean.
    X_train = rng.normal(0, 1, size=(200, 8)).astype(np.float32)
    X_train[:10] += 5.0

    client = ModalAnomalyClient()

    train_result = client.train(
        X_train,
        contamination=0.05,
        n_estimators=64,
        model_id="default",
        confirm_train=True,
        force_retrain=False,
        return_serialized_model=False,
    )
    print("Train status:", train_result.get("status"))

    X_predict = rng.normal(0, 1, size=(64, 8)).astype(np.float32)
    predict_result = client.predict(X_predict, model_id="default")

    labels = predict_result.get("anomaly_labels", [])
    scores = predict_result.get("anomaly_scores", [])

    print("Predictions:", len(labels))
    print("Top-5 scores:", [round(float(v), 4) for v in scores[:5]])


if __name__ == "__main__":
    main()
