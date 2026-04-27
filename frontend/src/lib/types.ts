export type TaskType = "classification" | "regression" | "clustering";

export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface UploadMetadata {
  session_id: string;
  num_rows: number;
  num_columns: number;
  columns: string[];
}

export interface UploadResponse {
  status: "success";
  metadata: UploadMetadata;
}

export interface DetectTaskResponse {
  status: "success";
  task_type: TaskType;
  sample_counts: Record<string, number>;
}

export interface OverviewResponse {
  status: "success";
  dtype_counts: Record<string, number>;
  missing_values: Record<string, number>;
  correlation_matrix: Record<string, Record<string, number>>;
  numeric_cols: string[];
}

export interface TrainConfig {
  session_id: string;
  target_col?: string | null;
  task_type: TaskType;
  selected_models: string[];
  selected_metric: string;
  hyperparameter_tuning: boolean;
  test_size: number;
  cv_fold_option: number;
  tree_max_depth?: number | null;
  n_estimators?: number;
  svm_c?: number;
  knn_neighbors?: number;
  kmeans_n_clusters?: number;
  dbscan_eps?: number;
  dbscan_min_samples?: number;
}

export interface CreateTrainJobResponse {
  status: "accepted";
  job_id: string;
  poll_url: string;
  result_url: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  created_at?: string;
  updated_at?: string;
  started_at?: string;
  completed_at?: string;
  session_id?: string;
  task_type?: TaskType;
  selected_metric?: string;
  summary?: {
    task_type?: TaskType;
    best_model_name?: string;
  } | null;
  error?: string | null;
}

export interface TrainResult {
  status: "success" | "error";
  task_type?: TaskType;
  performance?: Record<string, number | [number, number]>;
  best_model_name?: string | null;
  best_params_dict?: Record<string, unknown>;
  all_metrics_dict?: Record<string, Record<string, unknown>>;
  use_cv?: boolean;
  predictions?: Record<string, { actual: number[]; predicted: number[] }>;
  cluster_labels?: Record<string, number[]>;
  pca_data?: {
    pc1: number[];
    pc2: number[];
    pc3?: number[];
    explained_variance_ratio: number[];
    explained_variance_ratio_3d?: number[];
  };
  silhouette_data?: Record<string, { average: number; samples: number[] }>;
  message?: string;
}

export interface JobResultResponse {
  status: JobStatus;
  message?: string;
  job_id: string;
  result?: TrainResult;
  error?: string;
}
