"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { PlotData } from "plotly.js";

import {
  createTrainJob,
  detectTask,
  getDatasetOverview,
  getTrainJobResult,
  getTrainJobStatus,
  metricCatalog,
  modelCatalog,
  uploadDataset,
} from "@/lib/api";
import type {
  JobStatus,
  OverviewResponse,
  TaskType,
  TrainResult,
  UploadMetadata,
} from "@/lib/types";
import { PlotChart } from "@/components/charts/plot";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { FieldWrapper, SelectInput, TextInput } from "@/components/ui/field";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Stepper } from "@/components/ui/stepper";

const STEPS = ["Upload", "Configure", "Run", "Results"];

const TASK_OPTIONS: Array<{ label: string; value: "auto" | TaskType }> = [
  { label: "Auto", value: "auto" },
  { label: "Classification", value: "classification" },
  { label: "Regression", value: "regression" },
  { label: "Clustering", value: "clustering" },
];

const JOB_PROGRESS: Record<JobStatus, number> = {
  queued: 18,
  running: 68,
  completed: 100,
  failed: 100,
};

interface ScoreRow {
  model: string;
  score: number;
  cvScore?: number;
}

function normalizeScores(result: TrainResult | null): ScoreRow[] {
  if (!result?.performance) {
    return [];
  }

  return Object.entries(result.performance).map(([model, metric]) => {
    if (Array.isArray(metric)) {
      return {
        model,
        score: Number(metric[0] ?? 0),
        cvScore: Number(metric[1] ?? 0),
      };
    }
    return {
      model,
      score: Number(metric),
    };
  });
}

function toFixed(value: number | undefined, digits = 4) {
  if (!Number.isFinite(value)) {
    return "N/A";
  }
  return Number(value).toFixed(digits);
}

export default function Home() {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<UploadMetadata | null>(null);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [targetCol, setTargetCol] = useState("");
  const [taskSelection, setTaskSelection] = useState<"auto" | TaskType>("auto");
  const [detectedTask, setDetectedTask] = useState<TaskType | null>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedMetric, setSelectedMetric] = useState("");
  const [testSize, setTestSize] = useState(0.2);
  const [cvFolds, setCvFolds] = useState(5);
  const [hyperparameterTuning, setHyperparameterTuning] = useState(false);
  const [treeMaxDepth, setTreeMaxDepth] = useState<string>("none");
  const [nEstimators, setNEstimators] = useState(100);
  const [kmeansClusters, setKmeansClusters] = useState(4);
  const [dbscanEps, setDbscanEps] = useState(0.5);
  const [dbscanMinSamples, setDbscanMinSamples] = useState(5);

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus>("queued");
  const [isUploading, setIsUploading] = useState(false);
  const [isDetectingTask, setIsDetectingTask] = useState(false);
  const [isStartingTraining, setIsStartingTraining] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("Ready for upload");
  const [result, setResult] = useState<TrainResult | null>(null);

  const resolvedTask: TaskType = useMemo(() => {
    if (taskSelection === "auto") {
      return detectedTask ?? "classification";
    }
    return taskSelection;
  }, [taskSelection, detectedTask]);

  const modelOptions = useMemo(
    () => modelCatalog[resolvedTask],
    [resolvedTask],
  );
  const metricOptions = useMemo(
    () => metricCatalog[resolvedTask],
    [resolvedTask],
  );
  const scoreRows = useMemo(() => normalizeScores(result), [result]);

  useEffect(() => {
    if (!metadata?.columns.length) {
      return;
    }
    if (!targetCol) {
      setTargetCol(metadata.columns[0]);
    }
  }, [metadata, targetCol]);

  useEffect(() => {
    if (!metricOptions.length || !modelOptions.length) {
      return;
    }
    setSelectedMetric(metricOptions[0]);
    setSelectedModels(modelOptions.slice(0, Math.min(3, modelOptions.length)));
  }, [resolvedTask, metricOptions, modelOptions]);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const status = await getTrainJobStatus(jobId);
        if (cancelled) {
          return;
        }
        setJobStatus(status.status);

        if (status.status === "completed") {
          const finalResult = await getTrainJobResult(jobId);
          if (cancelled) {
            return;
          }
          setResult(finalResult.result ?? null);
          setStatusText("Training completed");
          setCurrentStep(3);
          setJobId(null);
        } else if (status.status === "failed") {
          setError(status.error ?? "Training job failed.");
          setStatusText("Training failed");
          setCurrentStep(2);
          setJobId(null);
        }
      } catch (pollError) {
        if (!cancelled) {
          setError((pollError as Error).message);
          setJobId(null);
        }
      }
    };

    void poll();
    const timer = setInterval(() => {
      void poll();
    }, 1200);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [jobId]);

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please choose a dataset file first.");
      return;
    }

    try {
      setError(null);
      setIsUploading(true);
      setStatusText("Uploading dataset...");
      const upload = await uploadDataset(selectedFile);
      setMetadata(upload.metadata);
      setCurrentStep(1);

      const overviewPayload = await getDatasetOverview(
        upload.metadata.session_id,
      );
      setOverview(overviewPayload);
      setStatusText("Dataset ready");
    } catch (uploadError) {
      setError((uploadError as Error).message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDetectTask = async () => {
    if (!metadata?.session_id || !targetCol) {
      setError("Upload a dataset and choose a target column first.");
      return;
    }

    try {
      setError(null);
      setIsDetectingTask(true);
      const task = await detectTask(metadata.session_id, targetCol);
      setDetectedTask(task.task_type);
      setStatusText(`Auto-detected ${task.task_type}`);
    } catch (detectError) {
      setError((detectError as Error).message);
    } finally {
      setIsDetectingTask(false);
    }
  };

  const toggleModel = (modelName: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelName)
        ? prev.filter((name) => name !== modelName)
        : [...prev, modelName],
    );
  };

  const startTraining = async () => {
    if (!metadata?.session_id) {
      setError("Upload a dataset first.");
      return;
    }

    if (resolvedTask !== "clustering" && !targetCol) {
      setError("Select a target column before training.");
      return;
    }

    if (!selectedModels.length) {
      setError("Select at least one model.");
      return;
    }

    try {
      setError(null);
      setIsStartingTraining(true);
      setCurrentStep(2);
      setStatusText("Submitting training job...");

      const payload = {
        session_id: metadata.session_id,
        target_col: resolvedTask === "clustering" ? null : targetCol,
        task_type: resolvedTask,
        selected_models: selectedModels,
        selected_metric: selectedMetric,
        hyperparameter_tuning: hyperparameterTuning,
        test_size: testSize,
        cv_fold_option: cvFolds,
        tree_max_depth: treeMaxDepth === "none" ? null : Number(treeMaxDepth),
        n_estimators: nEstimators,
        svm_c: 1.0,
        knn_neighbors: 5,
        kmeans_n_clusters: kmeansClusters,
        dbscan_eps: dbscanEps,
        dbscan_min_samples: dbscanMinSamples,
      };

      const job = await createTrainJob(payload);
      setJobId(job.job_id);
      setJobStatus("queued");
      setStatusText("Training job queued");
    } catch (startError) {
      setError((startError as Error).message);
    } finally {
      setIsStartingTraining(false);
    }
  };

  const dtypeChartData: Partial<PlotData>[] = useMemo(() => {
    if (!overview) {
      return [];
    }
    return [
      {
        type: "bar",
        x: Object.keys(overview.dtype_counts),
        y: Object.values(overview.dtype_counts),
        marker: {
          color: "#0d7c70",
        },
      },
    ];
  }, [overview]);

  const missingChartData: Partial<PlotData>[] = useMemo(() => {
    if (!overview) {
      return [];
    }

    return [
      {
        type: "bar",
        x: Object.keys(overview.missing_values),
        y: Object.values(overview.missing_values),
        marker: {
          color: "#ea8a3d",
        },
      },
    ];
  }, [overview]);

  const performanceChartData: Partial<PlotData>[] = useMemo(() => {
    if (!scoreRows.length) {
      return [];
    }

    return [
      {
        type: "bar",
        x: scoreRows.map((row) => row.model),
        y: scoreRows.map((row) => row.score),
        marker: {
          color: scoreRows.map((row) =>
            row.model === result?.best_model_name ? "#0d7c70" : "#6d9182",
          ),
        },
      },
    ];
  }, [result?.best_model_name, scoreRows]);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 px-5 py-8 md:px-8 lg:px-10">
      <header className="flex flex-col gap-4 rounded-3xl border border-[var(--line)] bg-[var(--panel)] px-6 py-7 shadow-[0_20px_45px_-30px_rgba(23,34,29,0.35)] md:flex-row md:items-end md:justify-between">
        <div>
          
          <h1 className="text-3xl font-bold tracking-tight text-[var(--ink)] md:text-4xl">
            OptiMLFlow
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-[var(--ink-soft)] md:text-base">
            A sleek minimal interface for upload, automatic model training, and
            rich result exploration.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge tone={jobId ? "info" : "neutral"}>
            {jobId ? `Job ${jobStatus}` : statusText}
          </Badge>
          <Link href="/domains">
            <Button variant="secondary">Domain Workspaces</Button>
          </Link>
        </div>
      </header>

      <Card>
        <CardContent>
          <Stepper steps={STEPS} currentStep={currentStep} />
        </CardContent>
      </Card>

      {error ? (
        <Card className="border-[color:var(--danger-faint)] bg-[color:var(--danger-faint)]">
          <CardContent>
            <p className="text-sm font-semibold text-[var(--danger)]">
              {error}
            </p>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-12">
        <Card className="lg:col-span-5">
          <CardHeader>
            <CardTitle>1. Upload Dataset</CardTitle>
            <CardDescription>
              CSV, TXT, or DATA files are supported by the backend.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <FieldWrapper label="Dataset file">
              <TextInput
                type="file"
                accept=".csv,.txt,.data"
                onChange={(event) =>
                  setSelectedFile(event.target.files?.[0] ?? null)
                }
              />
            </FieldWrapper>
            <Button onClick={handleUpload} loading={isUploading}>
              Upload and Profile
            </Button>

            {metadata ? (
              <div className="grid grid-cols-3 gap-3">
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent className="py-3">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                      Rows
                    </div>
                    <div className="text-lg font-semibold text-[var(--ink)]">
                      {metadata.num_rows.toLocaleString()}
                    </div>
                  </CardContent>
                </Card>
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent className="py-3">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                      Columns
                    </div>
                    <div className="text-lg font-semibold text-[var(--ink)]">
                      {metadata.num_columns}
                    </div>
                  </CardContent>
                </Card>
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent className="py-3">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                      Session
                    </div>
                    <div className="truncate text-xs font-semibold text-[var(--ink)]">
                      {metadata.session_id}
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : null}

            {overview ? (
              <div className="grid gap-3 md:grid-cols-2">
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent>
                    <div className="mb-2 text-sm font-semibold text-[var(--ink)]">
                      Data Types
                    </div>
                    <PlotChart data={dtypeChartData} height={220} />
                  </CardContent>
                </Card>
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent>
                    <div className="mb-2 text-sm font-semibold text-[var(--ink)]">
                      Missing Values
                    </div>
                    <PlotChart data={missingChartData} height={220} />
                  </CardContent>
                </Card>
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card className="lg:col-span-7">
          <CardHeader>
            <CardTitle>2. Configure Training</CardTitle>
            <CardDescription>
              Select task, metrics, and model set for this run.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            <FieldWrapper label="Target column">
              <SelectInput
                value={targetCol}
                onChange={(event) => setTargetCol(event.target.value)}
                disabled={!metadata}
              >
                <option value="">Select a target</option>
                {metadata?.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </SelectInput>
            </FieldWrapper>

            <FieldWrapper label="Task mode">
              <SelectInput
                value={taskSelection}
                onChange={(event) =>
                  setTaskSelection(event.target.value as "auto" | TaskType)
                }
                disabled={!metadata}
              >
                {TASK_OPTIONS.map((taskOption) => (
                  <option key={taskOption.value} value={taskOption.value}>
                    {taskOption.label}
                  </option>
                ))}
              </SelectInput>
            </FieldWrapper>

            <div className="md:col-span-2">
              <Button
                variant="secondary"
                onClick={handleDetectTask}
                loading={isDetectingTask}
                disabled={!metadata || !targetCol}
              >
                Auto Detect Task
              </Button>
              {detectedTask ? (
                <p className="mt-2 text-sm text-[var(--ink-soft)]">
                  Detected task type: {detectedTask}
                </p>
              ) : null}
            </div>

            <FieldWrapper label="Optimization metric">
              <SelectInput
                value={selectedMetric}
                onChange={(event) => setSelectedMetric(event.target.value)}
                disabled={!metadata}
              >
                {metricOptions.map((metric) => (
                  <option key={metric} value={metric}>
                    {metric}
                  </option>
                ))}
              </SelectInput>
            </FieldWrapper>

            <FieldWrapper
              label="Test size"
              hint={`${Math.round(testSize * 100)}%`}
            >
              <TextInput
                type="range"
                min={0.1}
                max={0.4}
                step={0.05}
                value={testSize}
                onChange={(event) => setTestSize(Number(event.target.value))}
                disabled={!metadata}
              />
            </FieldWrapper>

            <FieldWrapper label="CV folds">
              <SelectInput
                value={String(cvFolds)}
                onChange={(event) => setCvFolds(Number(event.target.value))}
                disabled={!metadata}
              >
                {[3, 5, 10].map((fold) => (
                  <option key={fold} value={fold}>
                    {fold}
                  </option>
                ))}
              </SelectInput>
            </FieldWrapper>

            <FieldWrapper label="Tree max depth">
              <SelectInput
                value={treeMaxDepth}
                onChange={(event) => setTreeMaxDepth(event.target.value)}
                disabled={!metadata}
              >
                <option value="none">None</option>
                {Array.from({ length: 19 }, (_, i) => i + 2).map((depth) => (
                  <option key={depth} value={String(depth)}>
                    {depth}
                  </option>
                ))}
              </SelectInput>
            </FieldWrapper>

            <FieldWrapper label="Tree estimators" hint={`${nEstimators}`}>
              <TextInput
                type="range"
                min={50}
                max={500}
                step={50}
                value={nEstimators}
                onChange={(event) => setNEstimators(Number(event.target.value))}
                disabled={!metadata}
              />
            </FieldWrapper>

            {resolvedTask === "clustering" ? (
              <>
                <FieldWrapper label="KMeans clusters">
                  <TextInput
                    type="number"
                    min={2}
                    max={12}
                    value={kmeansClusters}
                    onChange={(event) =>
                      setKmeansClusters(Number(event.target.value))
                    }
                    disabled={!metadata}
                  />
                </FieldWrapper>
                <FieldWrapper label="DBSCAN eps">
                  <TextInput
                    type="number"
                    min={0.1}
                    max={2}
                    step={0.1}
                    value={dbscanEps}
                    onChange={(event) =>
                      setDbscanEps(Number(event.target.value))
                    }
                    disabled={!metadata}
                  />
                </FieldWrapper>
                <FieldWrapper label="DBSCAN min samples">
                  <TextInput
                    type="number"
                    min={2}
                    max={20}
                    value={dbscanMinSamples}
                    onChange={(event) =>
                      setDbscanMinSamples(Number(event.target.value))
                    }
                    disabled={!metadata}
                  />
                </FieldWrapper>
              </>
            ) : null}

            <label className="md:col-span-2 inline-flex items-center gap-2 text-sm text-[var(--ink)]">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-[var(--line)] accent-[var(--accent)]"
                checked={hyperparameterTuning}
                onChange={(event) =>
                  setHyperparameterTuning(event.target.checked)
                }
                disabled={!metadata}
              />
              Enable hyperparameter tuning
            </label>

            <div className="md:col-span-2">
              <div className="mb-2 text-sm font-medium text-[var(--ink)]">
                Models
              </div>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {modelOptions.map((model) => {
                  const checked = selectedModels.includes(model);
                  return (
                    <label
                      key={model}
                      className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--panel-soft)] px-3 py-2 text-xs text-[var(--ink)]"
                    >
                      <input
                        type="checkbox"
                        className="h-4 w-4 accent-[var(--accent)]"
                        checked={checked}
                        onChange={() => toggleModel(model)}
                        disabled={!metadata}
                      />
                      <span className="truncate">{model}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>3. Run Training</CardTitle>
          <CardDescription>
            Training runs through async job endpoints for a non-blocking UX.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={startTraining}
              loading={isStartingTraining}
              disabled={!metadata || !!jobId}
            >
              Start Async Training
            </Button>
            <Badge
              tone={
                jobStatus === "failed"
                  ? "danger"
                  : jobStatus === "completed"
                    ? "success"
                    : "info"
              }
            >
              {jobId
                ? `${jobStatus.toUpperCase()} (${jobId.slice(0, 8)}...)`
                : "No active job"}
            </Badge>
          </div>
          <ProgressBar
            value={jobId ? JOB_PROGRESS[jobStatus] : result ? 100 : 0}
            label={
              jobId
                ? `Job ${jobStatus}`
                : result
                  ? "Latest result loaded"
                  : "No running job"
            }
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>4. Results</CardTitle>
          <CardDescription>
            Model ranking and key metrics rendered from backend output.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {!result ? (
            <div className="rounded-xl border border-dashed border-[var(--line)] bg-[var(--panel-soft)] px-4 py-8 text-center text-sm text-[var(--ink-soft)]">
              No result yet. Start a training job to populate this view.
            </div>
          ) : (
            <>
              <div className="grid gap-3 md:grid-cols-3">
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent className="py-3">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                      Task
                    </div>
                    <div className="text-lg font-semibold text-[var(--ink)]">
                      {result.task_type ?? resolvedTask}
                    </div>
                  </CardContent>
                </Card>
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent className="py-3">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                      Best Model
                    </div>
                    <div className="truncate text-lg font-semibold text-[var(--ink)]">
                      {result.best_model_name ?? "N/A"}
                    </div>
                  </CardContent>
                </Card>
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent className="py-3">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                      Models Trained
                    </div>
                    <div className="text-lg font-semibold text-[var(--ink)]">
                      {scoreRows.length}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {performanceChartData.length ? (
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent>
                    <div className="mb-2 text-sm font-semibold text-[var(--ink)]">
                      Model Performance
                    </div>
                    <PlotChart data={performanceChartData} height={300} />
                  </CardContent>
                </Card>
              ) : null}

              <div className="overflow-x-auto rounded-xl border border-[var(--line)]">
                <table className="min-w-full text-sm">
                  <thead className="bg-[var(--panel-soft)] text-left text-[var(--ink-soft)]">
                    <tr>
                      <th className="px-3 py-2 font-semibold">Model</th>
                      <th className="px-3 py-2 font-semibold">Primary Score</th>
                      <th className="px-3 py-2 font-semibold">CV Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scoreRows.map((row) => (
                      <tr
                        key={row.model}
                        className="border-t border-[var(--line)]"
                      >
                        <td className="px-3 py-2 text-[var(--ink)]">
                          {row.model}
                        </td>
                        <td className="px-3 py-2 text-[var(--ink)]">
                          {toFixed(row.score)}
                        </td>
                        <td className="px-3 py-2 text-[var(--ink)]">
                          {toFixed(row.cvScore)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
