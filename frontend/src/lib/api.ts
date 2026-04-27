import type {
  CreateTrainJobResponse,
  DetectTaskResponse,
  JobResultResponse,
  JobStatusResponse,
  OverviewResponse,
  TrainConfig,
  UploadResponse,
} from "@/lib/types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function normalizeBase(base: string) {
  return base.endsWith("/") ? base.slice(0, -1) : base;
}

const API = normalizeBase(API_BASE);

async function parseResponse<T>(res: Response): Promise<T> {
  const text = await res.text();
  const payload = text
    ? (JSON.parse(text) as T | { detail?: string; message?: string })
    : ({} as T);

  if (!res.ok) {
    const details = (payload as { detail?: string; message?: string }).detail;
    const message = (payload as { detail?: string; message?: string }).message;
    throw new Error(
      details || message || `Request failed with status ${res.status}`,
    );
  }

  return payload as T;
}

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API}/api/v2/sessions/upload`, {
    method: "POST",
    body: formData,
  });
  return parseResponse<UploadResponse>(res);
}

export async function getDatasetOverview(
  sessionId: string,
): Promise<OverviewResponse> {
  const res = await fetch(`${API}/api/v2/sessions/${sessionId}/overview`, {
    method: "GET",
  });
  return parseResponse<OverviewResponse>(res);
}

export async function detectTask(
  sessionId: string,
  targetCol: string,
): Promise<DetectTaskResponse> {
  const res = await fetch(`${API}/api/detect_task`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, target_col: targetCol }),
  });
  return parseResponse<DetectTaskResponse>(res);
}

export async function createTrainJob(
  config: TrainConfig,
): Promise<CreateTrainJobResponse> {
  const res = await fetch(`${API}/api/v2/jobs/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return parseResponse<CreateTrainJobResponse>(res);
}

export async function createUpsellDomainJob(
  config: Omit<TrainConfig, "task_type">,
): Promise<CreateTrainJobResponse> {
  const res = await fetch(`${API}/api/v2/domains/upsell/jobs/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return parseResponse<CreateTrainJobResponse>(res);
}

export async function createSegmentationDomainJob(
  config: Omit<TrainConfig, "task_type">,
): Promise<CreateTrainJobResponse> {
  const res = await fetch(`${API}/api/v2/domains/segmentation/jobs/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return parseResponse<CreateTrainJobResponse>(res);
}

export async function getTrainJobStatus(
  jobId: string,
): Promise<JobStatusResponse> {
  const res = await fetch(`${API}/api/v2/jobs/${jobId}`, {
    method: "GET",
  });
  return parseResponse<JobStatusResponse>(res);
}

export async function getTrainJobResult(
  jobId: string,
): Promise<JobResultResponse> {
  const res = await fetch(`${API}/api/v2/jobs/${jobId}/result`, {
    method: "GET",
  });
  return parseResponse<JobResultResponse>(res);
}

export const modelCatalog = {
  classification: [
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "LogisticRegression",
    "AdaBoostClassifier",
    "GaussianNB",
    "SVC",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "XGBClassifier",
    "LGBMClassifier",
  ],
  regression: [
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "LinearRegression",
    "AdaBoostRegressor",
    "SVR",
    "KNeighborsRegressor",
    "DecisionTreeRegressor",
    "XGBRegressor",
    "LGBMRegressor",
  ],
  clustering: [
    "KMeans",
    "DBSCAN",
    "AgglomerativeClustering",
    "MeanShift",
    "SpectralClustering",
  ],
} as const;

export const metricCatalog = {
  classification: [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "roc_auc",
  ],
  regression: ["r2", "adjusted_r2", "mae", "mse", "rmse", "mape"],
  clustering: ["silhouette", "davies_bouldin", "calinski_harabasz"],
} as const;
