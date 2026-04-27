"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { PlotData } from "plotly.js";

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
import {
  createSegmentationDomainJob,
  createUpsellDomainJob,
  getTrainJobResult,
  getTrainJobStatus,
  metricCatalog,
  modelCatalog,
  uploadDataset,
} from "@/lib/api";
import type {
  JobStatus,
  TrainResult,
  UploadMetadata,
} from "@/lib/types";

type DomainKey = "upsell" | "segmentation" | "recommendations" | "iot";

interface ScoreRow {
  model: string;
  score: number;
  cvScore?: number;
}

const JOB_PROGRESS: Record<JobStatus, number> = {
  queued: 18,
  running: 68,
  completed: 100,
  failed: 100,
};

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

export default function DomainsPage() {
  const [selectedDomain, setSelectedDomain] = useState<DomainKey>("upsell");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<UploadMetadata | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("Upload a dataset to start");

  const [upsellTargetCol, setUpsellTargetCol] = useState("");
  const [upsellMetric, setUpsellMetric] = useState<string>(metricCatalog.classification[0]);
  const [upsellModels, setUpsellModels] = useState<string[]>(
    modelCatalog.classification.slice(0, Math.min(3, modelCatalog.classification.length)),
  );

  const [segmentationTargetCol, setSegmentationTargetCol] = useState("(none)");
  const [segmentationMetric, setSegmentationMetric] = useState<string>(metricCatalog.clustering[0]);
  const [segmentationModels, setSegmentationModels] = useState<string[]>(
    modelCatalog.clustering.slice(0, Math.min(2, modelCatalog.clustering.length)),
  );
  const [segmentationClusters, setSegmentationClusters] = useState(4);

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobDomain, setJobDomain] = useState<DomainKey | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus>("queued");
  const [isStartingTraining, setIsStartingTraining] = useState(false);

  const [upsellResult, setUpsellResult] = useState<TrainResult | null>(null);
  const [segmentationResult, setSegmentationResult] = useState<TrainResult | null>(null);

  useEffect(() => {
    if (!metadata?.columns.length) {
      return;
    }
    if (!upsellTargetCol) {
      setUpsellTargetCol(metadata.columns[0]);
    }
  }, [metadata, upsellTargetCol]);

  useEffect(() => {
    if (!jobId || !jobDomain) {
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
          const payload = finalResult.result ?? null;
          if (jobDomain === "upsell") {
            setUpsellResult(payload);
          }
          if (jobDomain === "segmentation") {
            setSegmentationResult(payload);
          }
          setStatusText(`${jobDomain} training completed`);
          setJobId(null);
          setJobDomain(null);
        } else if (status.status === "failed") {
          setError(status.error ?? "Training job failed.");
          setStatusText(`${jobDomain} training failed`);
          setJobId(null);
          setJobDomain(null);
        }
      } catch (pollError) {
        if (!cancelled) {
          setError((pollError as Error).message);
          setJobId(null);
          setJobDomain(null);
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
  }, [jobId, jobDomain]);

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
      setStatusText("Dataset uploaded for domain workflows");
    } catch (uploadError) {
      setError((uploadError as Error).message);
    } finally {
      setIsUploading(false);
    }
  };

  const toggleModel = (domain: DomainKey, modelName: string) => {
    if (domain === "upsell") {
      setUpsellModels((prev) =>
        prev.includes(modelName)
          ? prev.filter((name) => name !== modelName)
          : [...prev, modelName],
      );
      return;
    }

    if (domain === "segmentation") {
      setSegmentationModels((prev) =>
        prev.includes(modelName)
          ? prev.filter((name) => name !== modelName)
          : [...prev, modelName],
      );
    }
  };

  const startUpsellTraining = async () => {
    if (!metadata?.session_id) {
      setError("Upload a dataset first.");
      return;
    }
    if (!upsellTargetCol) {
      setError("Select a target column for upsell training.");
      return;
    }
    if (!upsellModels.length) {
      setError("Select at least one upsell model.");
      return;
    }

    try {
      setError(null);
      setIsStartingTraining(true);
      setStatusText("Submitting upsell training job...");

      const job = await createUpsellDomainJob({
        session_id: metadata.session_id,
        target_col: upsellTargetCol,
        selected_models: upsellModels,
        selected_metric: upsellMetric,
        hyperparameter_tuning: false,
        test_size: 0.2,
        cv_fold_option: 5,
        tree_max_depth: null,
        n_estimators: 200,
        svm_c: 1.0,
        knn_neighbors: 5,
      });

      setJobId(job.job_id);
      setJobDomain("upsell");
      setJobStatus("queued");
      setSelectedDomain("upsell");
      setStatusText("Upsell job queued");
    } catch (startError) {
      setError((startError as Error).message);
    } finally {
      setIsStartingTraining(false);
    }
  };

  const startSegmentationTraining = async () => {
    if (!metadata?.session_id) {
      setError("Upload a dataset first.");
      return;
    }
    if (!segmentationModels.length) {
      setError("Select at least one segmentation model.");
      return;
    }

    try {
      setError(null);
      setIsStartingTraining(true);
      setStatusText("Submitting segmentation training job...");

      const targetCol =
        segmentationTargetCol === "(none)" ? null : segmentationTargetCol;

      const job = await createSegmentationDomainJob({
        session_id: metadata.session_id,
        target_col: targetCol,
        selected_models: segmentationModels,
        selected_metric: segmentationMetric,
        hyperparameter_tuning: false,
        test_size: 0.2,
        cv_fold_option: 5,
        tree_max_depth: null,
        n_estimators: 100,
        svm_c: 1.0,
        knn_neighbors: 5,
        kmeans_n_clusters: segmentationClusters,
        dbscan_eps: 0.5,
        dbscan_min_samples: 5,
      });

      setJobId(job.job_id);
      setJobDomain("segmentation");
      setJobStatus("queued");
      setSelectedDomain("segmentation");
      setStatusText("Segmentation job queued");
    } catch (startError) {
      setError((startError as Error).message);
    } finally {
      setIsStartingTraining(false);
    }
  };

  const activeResult = useMemo(() => {
    if (selectedDomain === "upsell") {
      return upsellResult;
    }
    if (selectedDomain === "segmentation") {
      return segmentationResult;
    }
    return null;
  }, [selectedDomain, upsellResult, segmentationResult]);

  const scoreRows = useMemo(() => normalizeScores(activeResult), [activeResult]);

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
            row.model === activeResult?.best_model_name ? "#0d7c70" : "#6d9182",
          ),
        },
      },
    ];
  }, [activeResult?.best_model_name, scoreRows]);

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-5 px-5 py-8 md:px-8 lg:px-10">
      <header className="rounded-3xl border border-[var(--line)] bg-[var(--panel)] px-6 py-6 shadow-[0_20px_45px_-30px_rgba(23,34,29,0.35)]">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-[var(--ink-soft)]">
          Phase 2
        </p>
        <h1 className="text-3xl font-bold tracking-tight text-[var(--ink)]">
          Domain Workspaces
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-[var(--ink-soft)] md:text-base">
          Upsell and segmentation are now wired to live async training jobs.
          Recommendation and IoT workflows are next in line.
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Badge tone={jobId ? "info" : "neutral"}>
            {jobId && jobDomain
              ? `${jobDomain} ${jobStatus}`
              : statusText}
          </Badge>
          <Link href="/">
            <Button variant="secondary">Back to Core Journey</Button>
          </Link>
        </div>
      </header>

      {error ? (
        <Card className="border-[color:var(--danger-faint)] bg-[color:var(--danger-faint)]">
          <CardContent>
            <p className="text-sm font-semibold text-[var(--danger)]">{error}</p>
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Dataset Session</CardTitle>
          <CardDescription>
            Upload a dataset once, then run domain workflows against the same session.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-[1fr_auto] md:items-end">
          <FieldWrapper label="Dataset file (.csv/.txt/.data)">
            <TextInput
              type="file"
              accept=".csv,.txt,.data"
              onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            />
          </FieldWrapper>
          <Button onClick={handleUpload} loading={isUploading}>
            Upload for Domains
          </Button>

          {metadata ? (
            <div className="md:col-span-2 grid gap-3 sm:grid-cols-3">
              <Card className="bg-[var(--panel-soft)]">
                <CardContent className="py-3">
                  <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">Rows</div>
                  <div className="text-lg font-semibold text-[var(--ink)]">
                    {metadata.num_rows.toLocaleString()}
                  </div>
                </CardContent>
              </Card>
              <Card className="bg-[var(--panel-soft)]">
                <CardContent className="py-3">
                  <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">Columns</div>
                  <div className="text-lg font-semibold text-[var(--ink)]">{metadata.num_columns}</div>
                </CardContent>
              </Card>
              <Card className="bg-[var(--panel-soft)]">
                <CardContent className="py-3">
                  <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">Session</div>
                  <div className="truncate text-xs font-semibold text-[var(--ink)]">{metadata.session_id}</div>
                </CardContent>
              </Card>
            </div>
          ) : null}
        </CardContent>
      </Card>

      <section className="grid gap-4 md:grid-cols-2">
        <Card className={selectedDomain === "upsell" ? "border-[var(--accent)]" : undefined}>
          <CardHeader>
            <div className="mb-2">
              <Badge tone="success">Live</Badge>
            </div>
            <CardTitle>Upsell Probability</CardTitle>
            <CardDescription>
              Train classification models for upgrade propensity.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button variant="ghost" className="w-full" onClick={() => setSelectedDomain("upsell")}>Open Upsell Workspace</Button>
          </CardContent>
        </Card>

        <Card className={selectedDomain === "segmentation" ? "border-[var(--accent)]" : undefined}>
          <CardHeader>
            <div className="mb-2">
              <Badge tone="success">Live</Badge>
            </div>
            <CardTitle>Customer Segmentation</CardTitle>
            <CardDescription>
              Run clustering models and compare segmentation quality.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button variant="ghost" className="w-full" onClick={() => setSelectedDomain("segmentation")}>Open Segmentation Workspace</Button>
          </CardContent>
        </Card>

        <Card className={selectedDomain === "recommendations" ? "border-[var(--accent)]" : undefined}>
          <CardHeader>
            <div className="mb-2">
              <Badge tone="warning">In progress</Badge>
            </div>
            <CardTitle>E-commerce Recommendations</CardTitle>
            <CardDescription>
              Co-occurrence and propensity recommendation workflows are next.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button variant="ghost" className="w-full" onClick={() => setSelectedDomain("recommendations")}>View Build Status</Button>
          </CardContent>
        </Card>

        <Card className={selectedDomain === "iot" ? "border-[var(--accent)]" : undefined}>
          <CardHeader>
            <div className="mb-2">
              <Badge tone="warning">In progress</Badge>
            </div>
            <CardTitle>Manufacturing / IoT Anomaly</CardTitle>
            <CardDescription>
              Time-series feature engineering and anomaly endpoints are next.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button variant="ghost" className="w-full" onClick={() => setSelectedDomain("iot")}>View Build Status</Button>
          </CardContent>
        </Card>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Active Workspace</CardTitle>
          <CardDescription>
            {selectedDomain === "upsell" && "Configure and launch an upsell training run."}
            {selectedDomain === "segmentation" && "Configure and launch a segmentation training run."}
            {selectedDomain === "recommendations" && "Recommendation flow wiring starts next."}
            {selectedDomain === "iot" && "IoT anomaly flow wiring starts next."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {selectedDomain === "upsell" ? (
            <div className="grid gap-4 md:grid-cols-2">
              <FieldWrapper label="Target column (binary)">
                <SelectInput
                  value={upsellTargetCol}
                  onChange={(event) => setUpsellTargetCol(event.target.value)}
                  disabled={!metadata}
                >
                  <option value="">Select target</option>
                  {metadata?.columns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </SelectInput>
              </FieldWrapper>

              <FieldWrapper label="Metric">
                <SelectInput
                  value={upsellMetric}
                  onChange={(event) => setUpsellMetric(event.target.value)}
                  disabled={!metadata}
                >
                  {metricCatalog.classification.map((metric) => (
                    <option key={metric} value={metric}>
                      {metric}
                    </option>
                  ))}
                </SelectInput>
              </FieldWrapper>

              <div className="md:col-span-2">
                <div className="mb-2 text-sm font-medium text-[var(--ink)]">Models</div>
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {modelCatalog.classification.map((model) => (
                    <label
                      key={model}
                      className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--panel-soft)] px-3 py-2 text-xs text-[var(--ink)]"
                    >
                      <input
                        type="checkbox"
                        className="h-4 w-4 accent-[var(--accent)]"
                        checked={upsellModels.includes(model)}
                        onChange={() => toggleModel("upsell", model)}
                        disabled={!metadata}
                      />
                      <span className="truncate">{model}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="md:col-span-2">
                <Button onClick={startUpsellTraining} loading={isStartingTraining} disabled={!metadata || !!jobId}>
                  Start Upsell Training
                </Button>
              </div>
            </div>
          ) : null}

          {selectedDomain === "segmentation" ? (
            <div className="grid gap-4 md:grid-cols-2">
              <FieldWrapper label="Optional target to exclude">
                <SelectInput
                  value={segmentationTargetCol}
                  onChange={(event) => setSegmentationTargetCol(event.target.value)}
                  disabled={!metadata}
                >
                  <option value="(none)">(none)</option>
                  {metadata?.columns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </SelectInput>
              </FieldWrapper>

              <FieldWrapper label="Metric">
                <SelectInput
                  value={segmentationMetric}
                  onChange={(event) => setSegmentationMetric(event.target.value)}
                  disabled={!metadata}
                >
                  {metricCatalog.clustering.map((metric) => (
                    <option key={metric} value={metric}>
                      {metric}
                    </option>
                  ))}
                </SelectInput>
              </FieldWrapper>

              <FieldWrapper label="KMeans clusters">
                <TextInput
                  type="number"
                  min={2}
                  max={12}
                  value={segmentationClusters}
                  onChange={(event) => setSegmentationClusters(Number(event.target.value))}
                  disabled={!metadata}
                />
              </FieldWrapper>

              <div className="md:col-span-2">
                <div className="mb-2 text-sm font-medium text-[var(--ink)]">Models</div>
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {modelCatalog.clustering.map((model) => (
                    <label
                      key={model}
                      className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--panel-soft)] px-3 py-2 text-xs text-[var(--ink)]"
                    >
                      <input
                        type="checkbox"
                        className="h-4 w-4 accent-[var(--accent)]"
                        checked={segmentationModels.includes(model)}
                        onChange={() => toggleModel("segmentation", model)}
                        disabled={!metadata}
                      />
                      <span className="truncate">{model}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="md:col-span-2">
                <Button
                  onClick={startSegmentationTraining}
                  loading={isStartingTraining}
                  disabled={!metadata || !!jobId}
                >
                  Start Segmentation Training
                </Button>
              </div>
            </div>
          ) : null}

          {selectedDomain === "recommendations" || selectedDomain === "iot" ? (
            <div className="rounded-xl border border-dashed border-[var(--line)] bg-[var(--panel-soft)] px-4 py-8 text-sm text-[var(--ink-soft)]">
              This workspace is now queued for implementation. Next step is wiring dedicated backend endpoints and UI controls.
            </div>
          ) : null}

          <ProgressBar
            value={jobId ? JOB_PROGRESS[jobStatus] : activeResult ? 100 : 0}
            label={
              jobId && jobDomain
                ? `${jobDomain} job ${jobStatus}`
                : activeResult
                  ? "Latest result loaded"
                  : "No running job"
            }
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Results</CardTitle>
          <CardDescription>
            Scores rendered from backend training output for the active workspace.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!activeResult ? (
            <div className="rounded-xl border border-dashed border-[var(--line)] bg-[var(--panel-soft)] px-4 py-8 text-center text-sm text-[var(--ink-soft)]">
              No domain result yet. Run an upsell or segmentation job.
            </div>
          ) : (
            <>
              <div className="grid gap-3 md:grid-cols-3">
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent className="py-3">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">Task</div>
                    <div className="text-lg font-semibold text-[var(--ink)]">{activeResult.task_type ?? "N/A"}</div>
                  </CardContent>
                </Card>
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent className="py-3">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">Best model</div>
                    <div className="truncate text-lg font-semibold text-[var(--ink)]">{activeResult.best_model_name ?? "N/A"}</div>
                  </CardContent>
                </Card>
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent className="py-3">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--ink-soft)]">Models trained</div>
                    <div className="text-lg font-semibold text-[var(--ink)]">{scoreRows.length}</div>
                  </CardContent>
                </Card>
              </div>

              {performanceChartData.length ? (
                <Card className="bg-[var(--panel-soft)]">
                  <CardContent>
                    <div className="mb-2 text-sm font-semibold text-[var(--ink)]">Performance</div>
                    <PlotChart data={performanceChartData} height={300} />
                  </CardContent>
                </Card>
              ) : null}

              <div className="overflow-x-auto rounded-xl border border-[var(--line)]">
                <table className="min-w-full text-sm">
                  <thead className="bg-[var(--panel-soft)] text-left text-[var(--ink-soft)]">
                    <tr>
                      <th className="px-3 py-2 font-semibold">Model</th>
                      <th className="px-3 py-2 font-semibold">Primary score</th>
                      <th className="px-3 py-2 font-semibold">CV score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scoreRows.map((row) => (
                      <tr key={row.model} className="border-t border-[var(--line)]">
                        <td className="px-3 py-2 text-[var(--ink)]">{row.model}</td>
                        <td className="px-3 py-2 text-[var(--ink)]">{toFixed(row.score)}</td>
                        <td className="px-3 py-2 text-[var(--ink)]">{toFixed(row.cvScore)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </main>
  );
}
