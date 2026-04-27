from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import uuid
import os
import math
import shutil
import requests
from typing import Optional, Literal
from datetime import datetime
import threading

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans

from core.model_training import train_multiple_models, train_clustering_models, detect_task_type
from core.preprocessing import preprocess_and_split
from core.file_handler import detect_separator

app = FastAPI(title="OptiMLFlow API")

# Setup CORS for frontend.
# Use a comma-separated OPTIMLFLOW_CORS_ORIGINS env var for custom deployments.
def _get_cors_origins() -> list[str]:
    default_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]
    raw = os.getenv("OPTIMLFLOW_CORS_ORIGINS", "").strip()
    if not raw:
        return default_origins
    parsed = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return parsed or default_origins


cors_origins = _get_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp storage for datasets (in an actual app, use proper DB or cloud storage)
DATA_DIR = "Datasets/temp"
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory async job registry (MVP). Replace with persistent store for production.
TRAINING_JOBS: dict[str, dict] = {}
TRAINING_JOBS_LOCK = threading.Lock()
MAX_JOB_HISTORY = 200

# Optional Modal anomaly backend integration settings.
MODAL_ANOMALY_BASE_URL = os.getenv("MODAL_ANOMALY_BASE_URL", "").strip().rstrip("/")
MODAL_ANOMALY_API_KEY = os.getenv("MODAL_ANOMALY_API_KEY", "").strip()
MODAL_ANOMALY_TIMEOUT_SECONDS = float(os.getenv("MODAL_ANOMALY_TIMEOUT_SECONDS", "45"))

MODAL_MAX_TRAIN_ROWS = 30_000
MODAL_MAX_PREDICT_ROWS = 5_000
MODAL_MAX_COLUMNS = 256
MODAL_MAX_TRAIN_CELLS = 1_500_000
MODAL_MAX_PREDICT_CELLS = 500_000

# Optional generic Modal ML backend integration settings.
MODAL_ML_BASE_URL = os.getenv("MODAL_ML_BASE_URL", "").strip().rstrip("/")
MODAL_ML_API_KEY = os.getenv("MODAL_ML_API_KEY", "").strip()
MODAL_ML_TIMEOUT_SECONDS = float(os.getenv("MODAL_ML_TIMEOUT_SECONDS", "45"))

MODAL_ML_MAX_TRAIN_ROWS = 30_000
MODAL_ML_MAX_PREDICT_ROWS = 5_000
MODAL_ML_MAX_COLUMNS = 256
MODAL_ML_MAX_TRAIN_CELLS = 1_500_000
MODAL_ML_MAX_PREDICT_CELLS = 500_000

class SessionConfig(BaseModel):
    session_id: str
    target_col: Optional[str] = None

class TrainConfig(BaseModel):
    session_id: str
    target_col: Optional[str] = None
    task_type: str
    selected_models: list[str]
    selected_metric: str
    hyperparameter_tuning: bool = False
    test_size: float = 0.2
    cv_fold_option: int = 5
    # Extra parameters
    tree_max_depth: Optional[int] = None
    n_estimators: int = 100
    svm_c: float = 1.0
    knn_neighbors: int = 5
    # Clustering parameters
    kmeans_n_clusters: int = 3
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5


class UpsellDomainTrainConfig(BaseModel):
    session_id: str
    target_col: str
    selected_models: list[str] = [
        "RandomForestClassifier",
        "LogisticRegression",
        "GradientBoostingClassifier",
    ]
    selected_metric: str = "roc_auc"
    hyperparameter_tuning: bool = False
    test_size: float = 0.2
    cv_fold_option: int = 5
    tree_max_depth: Optional[int] = None
    n_estimators: int = 200
    svm_c: float = 1.0
    knn_neighbors: int = 5


class SegmentationDomainTrainConfig(BaseModel):
    session_id: str
    target_col: Optional[str] = None
    selected_models: list[str] = ["KMeans", "DBSCAN"]
    selected_metric: str = "silhouette"
    hyperparameter_tuning: bool = False
    kmeans_n_clusters: int = 4
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5

class ElbowCurveRequest(BaseModel):
    session_id: str
    target_col: str = None
    max_k: int = 10

class FeatureDistributionRequest(BaseModel):
    session_id: str
    feature_name: str


class ModalAnomalyTrainRequest(BaseModel):
    X: list[list[float]]
    contamination: float = Field(default=0.05, ge=0.001, le=0.2)
    n_estimators: int = Field(default=64, ge=50, le=100)
    model_id: str = Field(default="default", min_length=1, max_length=64)
    confirm_train: bool = True
    force_retrain: bool = False
    return_serialized_model: bool = False


class ModalAnomalyPredictRequest(BaseModel):
    X: list[list[float]]
    model_id: str = Field(default="default", min_length=1, max_length=64)


class ModalMLTrainRequest(BaseModel):
    X: list[list[float]]
    y: list[float | int | str | bool | None] | None = None
    task_type: Literal["classification", "regression", "anomaly"]
    model_type: Optional[str] = None
    model_id: Optional[str] = None
    contamination: float = Field(default=0.05, ge=0.001, le=0.2)
    confirm_train: bool = True
    force_retrain: bool = False
    return_serialized_model: bool = False


class ModalMLPredictRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=64)
    X: list[list[float]]


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.utcnow().isoformat() + "Z"


def _to_dict(model: BaseModel) -> dict:
    """Support both pydantic v1 and v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _insert_job(job_id: str, job: dict) -> None:
    """Insert and prune stale completed jobs."""
    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job_id] = job
        if len(TRAINING_JOBS) > MAX_JOB_HISTORY:
            removable = sorted(
                TRAINING_JOBS.items(),
                key=lambda pair: pair[1].get("created_at", ""),
            )
            for old_job_id, _ in removable[: len(TRAINING_JOBS) - MAX_JOB_HISTORY]:
                TRAINING_JOBS.pop(old_job_id, None)


def _update_job(job_id: str, **updates) -> None:
    """Patch job status atomically."""
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = _utc_now_iso()


def _get_job(job_id: str) -> dict | None:
    """Fetch a job snapshot."""
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if not job:
            return None
        return dict(job)

def sanitize_floats(obj):
    """Recursively convert NaN/Infinity to None or string for JSON serialization"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_floats(obj.tolist())
    return obj


def _require_modal_base_url() -> str:
    if not MODAL_ANOMALY_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Modal anomaly backend URL is not configured. Set MODAL_ANOMALY_BASE_URL.",
        )
    return MODAL_ANOMALY_BASE_URL


def _modal_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if MODAL_ANOMALY_API_KEY:
        headers["Authorization"] = f"Bearer {MODAL_ANOMALY_API_KEY}"
    return headers


def _validate_modal_matrix(matrix_like, *, max_rows: int, max_cells: int, context: str) -> None:
    try:
        matrix = np.asarray(matrix_like, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"{context}: X must contain numeric values only.",
        ) from exc

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

    if cols > MODAL_MAX_COLUMNS:
        raise HTTPException(
            status_code=413,
            detail=f"{context}: column limit exceeded. Max columns is {MODAL_MAX_COLUMNS}.",
        )

    if rows * cols > max_cells:
        raise HTTPException(
            status_code=413,
            detail=f"{context}: payload too large. Max cells is {max_cells}.",
        )

    if not np.isfinite(matrix).all():
        raise HTTPException(status_code=400, detail=f"{context}: X cannot contain NaN or infinity.")


def _call_modal_json(method: str, path: str, payload: dict | None = None, params: dict | None = None):
    base_url = _require_modal_base_url()
    url = f"{base_url}{path}"

    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            params=params,
            headers=_modal_headers(),
            timeout=MODAL_ANOMALY_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Modal backend request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = {"detail": response.text.strip() or "Unknown error"}

    if response.status_code >= 400:
        detail = data.get("detail", data) if isinstance(data, dict) else data
        raise HTTPException(status_code=response.status_code, detail=detail)

    return data


def _require_modal_ml_base_url() -> str:
    if not MODAL_ML_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Modal ML backend URL is not configured. Set MODAL_ML_BASE_URL.",
        )
    return MODAL_ML_BASE_URL


def _modal_ml_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if MODAL_ML_API_KEY:
        headers["Authorization"] = f"Bearer {MODAL_ML_API_KEY}"
    return headers


def _call_modal_ml_json(
    method: str,
    path: str,
    payload: dict | None = None,
    params: dict | None = None,
):
    base_url = _require_modal_ml_base_url()
    url = f"{base_url}{path}"

    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            params=params,
            headers=_modal_ml_headers(),
            timeout=MODAL_ML_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Modal ML backend request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = {"detail": response.text.strip() or "Unknown error"}

    if response.status_code >= 400:
        detail = data.get("detail", data) if isinstance(data, dict) else data
        raise HTTPException(status_code=response.status_code, detail=detail)

    return data


def _validate_modal_ml_target(
    target_like,
    *,
    rows: int,
    task_type: str,
) -> None:
    if task_type == "anomaly":
        return

    if target_like is None:
        raise HTTPException(status_code=400, detail=f"train: y is required for {task_type}.")

    if not isinstance(target_like, list):
        raise HTTPException(status_code=400, detail="train: y must be a list.")

    if len(target_like) != rows:
        raise HTTPException(status_code=400, detail="train: y length must match X rows.")

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.data', '.txt')):
        raise HTTPException(status_code=400, detail="Only CSV/Data/TXT files are supported.")
    
    session_id = str(uuid.uuid4())
    file_path = os.path.join(DATA_DIR, f"{session_id}.csv")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Detect separator
        with open(file_path, "rb") as f:
            sep = detect_separator(f)
            
        # Try to parse the dataframe to get columns and info
        df = pd.read_csv(file_path, sep=sep, on_bad_lines='skip')
        if df.empty:
            raise Exception("File is empty.")
            
        columns = df.columns.tolist()
        metadata = {
            "num_rows": len(df),
            "num_columns": len(columns),
            "columns": columns,
            "session_id": session_id
        }
        return {"status": "success", "metadata": metadata}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Error reading dataset: {str(e)}")

@app.post("/api/detect_task")
async def detect_task(config: SessionConfig):
    file_path = os.path.join(DATA_DIR, f"{config.session_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Session not found.")
        
    with open(file_path, "rb") as f:
        sep = detect_separator(f)
        
    df = pd.read_csv(file_path, sep=sep)
    if config.target_col not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found.")
        
    y = df[config.target_col]
    auto_detected_task = detect_task_type(y)
    
    # Send some target distribution summary
    value_counts = y.value_counts().head(20).to_dict() if auto_detected_task == 'classification' else {}
    
    return {
        "status": "success",
        "task_type": auto_detected_task,
        "sample_counts": value_counts
    }

@app.post("/api/data_overview")
async def get_data_overview(config: SessionConfig):
    file_path = os.path.join(DATA_DIR, f"{config.session_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Session not found.")
        
    with open(file_path, "rb") as f:
        sep = detect_separator(f)
        
    df = pd.read_csv(file_path, sep=sep)
    
    # 1. Data Types
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    
    # 2. Missing Values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].to_dict()
    
    # 3. Correlation Matrix (Numeric only)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    correlation_matrix = {}
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr()
        for col in corr_df.columns:
            correlation_matrix[col] = corr_df[col].to_dict()
            
    return sanitize_floats({
        "status": "success",
        "dtype_counts": dtype_counts,
        "missing_values": missing_values,
        "correlation_matrix": correlation_matrix,
        "numeric_cols": numeric_cols
    })

@app.post("/api/feature_distribution")
async def get_feature_distribution(req: FeatureDistributionRequest):
    file_path = os.path.join(DATA_DIR, f"{req.session_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Session not found.")
        
    with open(file_path, "rb") as f:
        sep = detect_separator(f)
        
    df = pd.read_csv(file_path, sep=sep)
    
    if req.feature_name not in df.columns:
        raise HTTPException(status_code=400, detail="Feature not found.")
        
    data = df[req.feature_name].dropna().tolist()
    
    return sanitize_floats({
        "status": "success",
        "feature_name": req.feature_name,
        "data": data,
        "is_numeric": pd.api.types.is_numeric_dtype(df[req.feature_name])
    })

@app.post("/api/elbow_curve")
async def get_elbow_curve(req: ElbowCurveRequest):
    file_path = os.path.join(DATA_DIR, f"{req.session_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Session not found.")
        
    with open(file_path, "rb") as f:
        sep = detect_separator(f)
        
    df = pd.read_csv(file_path, sep=sep)
    
    # Drop target col if provided
    X_all = df.drop(columns=[req.target_col]) if req.target_col and req.target_col in df.columns else df
    
    # Need basic preprocessing
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    numeric_features = X_all.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_all.select_dtypes(include=['object', 'category']).columns.tolist()
    
    transformers = []
    if numeric_features: transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features: transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
    
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers)
        X_all_proc = preprocessor.fit_transform(X_all)
    else:
        X_all_proc = X_all.values
        
    inertias = []
    silhouette_scores = []
    k_range = list(range(2, min(req.max_k + 1, len(X_all_proc))))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_all_proc)
        inertias.append(kmeans.inertia_)
        try:
            silhouette_scores.append(silhouette_score(X_all_proc, kmeans.labels_))
        except:
            silhouette_scores.append(0)
            
    return sanitize_floats({
        "status": "success",
        "k_range": k_range,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores
    })

@app.post("/api/train")
async def train_models(config: TrainConfig):
    file_path = os.path.join(DATA_DIR, f"{config.session_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Session not found.")
        
    with open(file_path, "rb") as f:
        sep = detect_separator(f)
        
    df = pd.read_csv(file_path, sep=sep)
    
    try:
        if config.task_type == 'clustering':
            # Preprocess all data
            target_col = config.target_col
            X_all = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df
            
            from sklearn.preprocessing import StandardScaler
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            
            numeric_features = X_all.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_all.select_dtypes(include=['object', 'category']).columns.tolist()
            
            transformers = []
            if numeric_features:
                transformers.append(('num', StandardScaler(), numeric_features))
            if categorical_features:
                transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
            
            preprocessor = ColumnTransformer(transformers=transformers)
            X_all_proc = preprocessor.fit_transform(X_all)
            
            performance, best_model_name, trained_models, best_params_dict, all_metrics_dict, cluster_labels = \
                train_clustering_models(
                    X_data=X_all_proc,
                    selected_models_clustering=config.selected_models,
                    selected_metric_clustering=config.selected_metric,
                    hyperparameter_tuning=config.hyperparameter_tuning,
                    n_clusters=config.kmeans_n_clusters,
                    progress_callback=lambda c,t,m: None,
                    kmeans_init='k-means++',
                    dbscan_eps=config.dbscan_eps,
                    dbscan_min_samples=config.dbscan_min_samples,
                    dbscan_metric='euclidean'
                )
                
            # Serialize
            serializable_cluster_labels = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in cluster_labels.items()}
            
            # --- START PHASE 2 CLUSTERING ADDITIONS ---
            # Compute PCA for 2D/3D visualization
            pca_2d = PCA(n_components=2)
            X_pca_2d = pca_2d.fit_transform(X_all_proc)
            
            # Standardize PCA dict for serialization
            pca_data = {
                "pc1": X_pca_2d[:, 0].tolist(),
                "pc2": X_pca_2d[:, 1].tolist(),
                "explained_variance_ratio": pca_2d.explained_variance_ratio_.tolist()
            }
            
            # Add 3D if possible
            if X_all_proc.shape[1] >= 3:
                pca_3d = PCA(n_components=3)
                X_pca_3d = pca_3d.fit_transform(X_all_proc)
                pca_data["pc3"] = X_pca_3d[:, 2].tolist()
                pca_data["explained_variance_ratio_3d"] = pca_3d.explained_variance_ratio_.tolist()
                
            # Compute Silhouette Samples for "Silhouette Analysis" plot
            silhouette_data = {}
            for m_name, labels in cluster_labels.items():
                if len(set(labels)) > 1:
                    avg_score = silhouette_score(X_all_proc, labels)
                    sample_scores = silhouette_samples(X_all_proc, labels)
                    silhouette_data[m_name] = {
                        "average": float(avg_score),
                        "samples": sample_scores.tolist()
                    }
            # --- END PHASE 2 CLUSTERING ADDITIONS ---

            return sanitize_floats({
                "status": "success",
                "task_type": "clustering",
                "performance": performance,
                "best_model_name": best_model_name,
                "best_params_dict": best_params_dict,
                "all_metrics_dict": all_metrics_dict,
                "cluster_labels": serializable_cluster_labels,
                "pca_data": pca_data,
                "silhouette_data": silhouette_data
            })
            
        else:
            # Classification or Regression
            X_train_proc, X_test_proc, y_train, y_test, preprocessor, column_info = \
                preprocess_and_split(df, config.target_col, test_size=config.test_size)
                
            performance, best_model_name, use_cv, cv_folds, current_task, trained_models, best_params_dict, all_metrics_dict = \
                train_multiple_models(
                    X_train_proc, X_test_proc, y_train, y_test,
                    task_type=config.task_type,
                    selected_models_cls=config.selected_models if config.task_type == 'classification' else [],
                    selected_models_reg=config.selected_models if config.task_type == 'regression' else [],
                    selected_metric_cls=config.selected_metric if config.task_type == 'classification' else 'accuracy',
                    selected_metric_reg=config.selected_metric if config.task_type == 'regression' else 'r2',
                    hyperparameter_tuning=config.hyperparameter_tuning,
                    cv_fold_option=config.cv_fold_option,
                    progress_callback=lambda c,t,m: None,
                    tree_max_depth=config.tree_max_depth,
                    n_estimators=config.n_estimators,
                    SVM_C=config.svm_c,
                    KNN_neighbors=config.knn_neighbors
                )
            
            # Additional logic to generate predictions for charting on frontend
            predictions = {}
            if current_task == "regression":
                for m_name, model in trained_models.items():
                    y_pred = model.predict(X_test_proc)
                    predictions[m_name] = {"actual": y_test.tolist(), "predicted": y_pred.tolist()}
                    
            return sanitize_floats({
                "status": "success",
                "task_type": current_task,
                "performance": performance,
                "best_model_name": best_model_name,
                "best_params_dict": best_params_dict,
                "all_metrics_dict": all_metrics_dict,
                "use_cv": use_cv,
                "predictions": predictions
            })
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.post("/api/v2/sessions/upload")
async def upload_dataset_v2(file: UploadFile = File(...)):
    """V2 alias for dataset upload endpoint."""
    return await upload_dataset(file)


@app.get("/api/v2/sessions/{session_id}/overview")
async def get_data_overview_v2(session_id: str):
    """V2 endpoint for dataset health/profile retrieval."""
    config = SessionConfig(session_id=session_id)
    return await get_data_overview(config)


async def _run_training_job(job_id: str, config_payload: dict) -> None:
    """Background worker for async train jobs."""
    _update_job(job_id, status="running", started_at=_utc_now_iso(), error=None)

    try:
        config = TrainConfig(**config_payload)
        result = await train_models(config)

        if result.get("status") == "success":
            summary = {
                "task_type": result.get("task_type"),
                "best_model_name": result.get("best_model_name"),
            }
            _update_job(
                job_id,
                status="completed",
                completed_at=_utc_now_iso(),
                summary=summary,
                result=result,
            )
        else:
            _update_job(
                job_id,
                status="failed",
                completed_at=_utc_now_iso(),
                error=result.get("message", "Training failed."),
                result=result,
            )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            completed_at=_utc_now_iso(),
            error=str(exc),
        )


@app.post("/api/v2/jobs/train")
async def create_train_job(config: TrainConfig, background_tasks: BackgroundTasks):
    """Create an asynchronous training job and return polling links."""
    job_id = str(uuid.uuid4())
    now = _utc_now_iso()
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "session_id": config.session_id,
        "task_type": config.task_type,
        "selected_metric": config.selected_metric,
        "error": None,
        "summary": None,
        "result": None,
    }
    _insert_job(job_id, job)

    background_tasks.add_task(_run_training_job, job_id, _to_dict(config))

    return {
        "status": "accepted",
        "job_id": job_id,
        "poll_url": f"/api/v2/jobs/{job_id}",
        "result_url": f"/api/v2/jobs/{job_id}/result",
    }


@app.post("/api/v2/domains/upsell/jobs/train")
async def create_upsell_train_job(
    config: UpsellDomainTrainConfig,
    background_tasks: BackgroundTasks,
):
    """Create async train job with upsell defaults and classification task type."""
    train_config = TrainConfig(
        session_id=config.session_id,
        target_col=config.target_col,
        task_type="classification",
        selected_models=config.selected_models,
        selected_metric=config.selected_metric,
        hyperparameter_tuning=config.hyperparameter_tuning,
        test_size=config.test_size,
        cv_fold_option=config.cv_fold_option,
        tree_max_depth=config.tree_max_depth,
        n_estimators=config.n_estimators,
        svm_c=config.svm_c,
        knn_neighbors=config.knn_neighbors,
    )
    return await create_train_job(train_config, background_tasks)


@app.post("/api/v2/domains/segmentation/jobs/train")
async def create_segmentation_train_job(
    config: SegmentationDomainTrainConfig,
    background_tasks: BackgroundTasks,
):
    """Create async train job with segmentation defaults and clustering task type."""
    train_config = TrainConfig(
        session_id=config.session_id,
        target_col=config.target_col,
        task_type="clustering",
        selected_models=config.selected_models,
        selected_metric=config.selected_metric,
        hyperparameter_tuning=config.hyperparameter_tuning,
        kmeans_n_clusters=config.kmeans_n_clusters,
        dbscan_eps=config.dbscan_eps,
        dbscan_min_samples=config.dbscan_min_samples,
    )
    return await create_train_job(train_config, background_tasks)


@app.get("/api/v2/jobs/{job_id}")
async def get_train_job(job_id: str):
    """Return job status and compact metadata for polling."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "session_id": job.get("session_id"),
        "task_type": job.get("task_type"),
        "selected_metric": job.get("selected_metric"),
        "summary": job.get("summary"),
        "error": job.get("error"),
    }


@app.get("/api/v2/jobs/{job_id}/result")
async def get_train_job_result(job_id: str):
    """Return final job result once training is complete."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] in {"queued", "running"}:
        return {
            "status": job["status"],
            "message": "Job is still in progress.",
            "job_id": job_id,
        }

    if job["status"] == "failed":
        return {
            "status": "failed",
            "job_id": job_id,
            "error": job.get("error", "Training failed."),
            "result": job.get("result"),
        }

    return {
        "status": "completed",
        "job_id": job_id,
        "result": job.get("result"),
    }


@app.post("/api/modal/anomaly/train")
def train_modal_anomaly(config: ModalAnomalyTrainRequest):
    """Trigger manual Modal training for anomaly detection."""
    _validate_modal_matrix(
        config.X,
        max_rows=MODAL_MAX_TRAIN_ROWS,
        max_cells=MODAL_MAX_TRAIN_CELLS,
        context="train",
    )
    return _call_modal_json("POST", "/train", payload=_to_dict(config))


@app.post("/api/modal/anomaly/predict")
def predict_modal_anomaly(config: ModalAnomalyPredictRequest):
    """Send batched feature matrix to Modal for anomaly inference."""
    _validate_modal_matrix(
        config.X,
        max_rows=MODAL_MAX_PREDICT_ROWS,
        max_cells=MODAL_MAX_PREDICT_CELLS,
        context="predict",
    )
    return _call_modal_json("POST", "/predict", payload=_to_dict(config))


@app.get("/api/modal/anomaly/health")
def health_modal_anomaly(model_id: str = "default"):
    """Check Modal model availability and metadata."""
    return _call_modal_json("GET", "/health", params={"model_id": model_id})


@app.post("/api/modal/ml/train")
def train_modal_ml(config: ModalMLTrainRequest):
    """Trigger generic Modal ML training for classification/regression/anomaly."""
    _validate_modal_matrix(
        config.X,
        max_rows=MODAL_ML_MAX_TRAIN_ROWS,
        max_cells=MODAL_ML_MAX_TRAIN_CELLS,
        context="train",
    )
    _validate_modal_ml_target(config.y, rows=len(config.X), task_type=config.task_type)
    return _call_modal_ml_json("POST", "/train", payload=_to_dict(config))


@app.post("/api/modal/ml/predict")
def predict_modal_ml(config: ModalMLPredictRequest):
    """Send feature matrix to generic Modal ML inference endpoint."""
    _validate_modal_matrix(
        config.X,
        max_rows=MODAL_ML_MAX_PREDICT_ROWS,
        max_cells=MODAL_ML_MAX_PREDICT_CELLS,
        context="predict",
    )
    return _call_modal_ml_json("POST", "/predict", payload=_to_dict(config))


@app.get("/api/modal/ml/health")
def health_modal_ml(model_id: str | None = None):
    """Check generic Modal ML backend status and optional model availability."""
    params = {"model_id": model_id} if model_id else None
    return _call_modal_ml_json("GET", "/health", params=params)
