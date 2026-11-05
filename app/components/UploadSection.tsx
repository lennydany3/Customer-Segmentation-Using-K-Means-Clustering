"use client";

import React, { useRef, useState } from "react";
import {
  FileUp,
  FileSpreadsheet,
  Upload,
  X,
  Loader2,
  CheckCircle2,
  CircleDashed,
  AlertCircle,
  Download,
  RotateCcw,
  ScanSearch,
  ArrowDown,
} from "lucide-react";
import { ClusterResponse, UploadResponse } from "../lib/types";

type UploadStatus = "idle" | "processing" | "done" | "error";

// Pipeline stages shown in the loader; each one is tied to a real backend call in the page handler
export const PROCESSING_STAGES = [
  { title: "Uploading dataset", detail: "Sending the CSV to the FastAPI server and checking its columns" },
  { title: "Running Elbow analysis", detail: "Computing WCSS and silhouette for K = 2…8 to pick the best K" },
  { title: "Training K-Means", detail: "Scaling features (z-score) and assigning each customer to a centroid" },
  { title: "Loading segmented customers", detail: "Fetching every customer with the segment the model assigned" },
  { title: "Building segment report", detail: "Aggregating personas & marketing strategies" },
];
const STEP_MS = 500; // minimum time each stage stays visible so the progress is readable
const MAX_FILE_BYTES = 5 * 1024 * 1024;

const TEMPLATE_CSV =
  "Customer_ID,Age,Annual_Income,Spending_Score\n" +
  "U001,30,95,85\nU002,48,90,20\nU003,24,28,80\nU004,52,30,18\nU005,35,72,55\n" +
  "U006,29,88,78\nU007,44,92,25\nU008,23,31,84\nU009,57,26,15\nU010,38,68,50\n";

const formatBytes = (bytes: number) =>
  bytes < 1024 ? `${bytes} B` : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;

const downloadText = (filename: string, text: string) => {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

const csvCell = (value: string | number | null) => {
  const s = value === null ? "" : String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

interface UploadSectionProps {
  /** Runs the whole backend pipeline; call `step(i)` as each stage starts (it resolves after a minimum display time). */
  onProcess: (file: File, step: (stage: number) => Promise<void>) => Promise<void>;
  summary: UploadResponse | null;
  clusteringData: ClusterResponse | null;
  fileName: string | null;
  onClear: () => void;
}

export const UploadSection: React.FC<UploadSectionProps> = ({
  onProcess,
  summary,
  clusteringData,
  fileName,
  onClear,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [status, setStatus] = useState<UploadStatus>(summary ? "done" : "idle");
  const [stage, setStage] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const selectFile = (selected: File | undefined | null) => {
    if (!selected) return;
    if (!selected.name.toLowerCase().endsWith(".csv")) {
      setError("Please upload a .csv file.");
      setStatus("error");
      return;
    }
    if (selected.size > MAX_FILE_BYTES) {
      setError("File is too large. Maximum upload size is 5 MB.");
      setStatus("error");
      return;
    }
    setFile(selected);
    setError(null);
    setStatus("idle");
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (status === "processing") return;
    selectFile(e.dataTransfer.files?.[0]);
  };

  const handleReset = () => {
    setFile(null);
    setError(null);
    setStatus("idle");
    setStage(0);
    onClear();
    if (inputRef.current) inputRef.current.value = "";
  };

  // Drops a newly chosen file but keeps showing the results of the dataset that is already loaded
  const handleClearSelection = () => {
    setFile(null);
    setError(null);
    setStatus(summary ? "done" : "idle");
    if (inputRef.current) inputRef.current.value = "";
  };

  const step = async (i: number) => {
    setStage(i);
    await wait(STEP_MS);
  };

  const handleProcess = async () => {
    if (!file) return;
    setStatus("processing");
    setStage(0);
    setError(null);
    try {
      await onProcess(file, step);
      setStage(PROCESSING_STAGES.length);
      await wait(400);
      setStatus("done");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to process the uploaded dataset.");
      setStatus("error");
    }
  };

  const handleDownloadResults = () => {
    if (!clusteringData) return;
    const labels = new Map(clusteringData.clusters.map((c) => [c.cluster_id, c.label]));
    const header = "Customer_ID,Age,Annual_Income,Spending_Score,Cluster,Segment";
    const lines = clusteringData.points.map((p) =>
      [p.id, p.age, p.income, p.spending, p.cluster, labels.get(p.cluster) ?? ""].map(csvCell).join(",")
    );
    const baseName = (fileName ?? "customers").replace(/\.csv$/i, "");
    downloadText(`${baseName}_segmented.csv`, [header, ...lines].join("\n"));
  };

  const progressPct = Math.round((stage / PROCESSING_STAGES.length) * 100);
  const shownName = file?.name ?? fileName;

  return (
    <section id="upload" className="py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex items-center gap-2 text-violet-600 dark:text-violet-400 font-semibold text-sm mb-1">
        <FileUp className="w-4 h-4" />
        <span>Step 1 · Upload</span>
      </div>
      <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white">
        Upload Your Customer Dataset
      </h2>
      <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-2xl mb-8">
        Upload a CSV of customers. The model scales the data, finds the best number of clusters, trains K-Means and
        segments every customer. Required columns: <span className="font-mono font-semibold">Annual_Income</span> and{" "}
        <span className="font-mono font-semibold">Spending_Score</span> (Kaggle&apos;s{" "}
        <span className="font-mono">Mall_Customers.csv</span> headers also work). <span className="font-mono">Customer_ID</span>{" "}
        and <span className="font-mono">Age</span> are optional. Income is treated as thousands ($k).
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Upload column */}
        <div className="lg:col-span-7 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs">
          <div
            role="button"
            tabIndex={0}
            onClick={() => status !== "processing" && inputRef.current?.click()}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && status !== "processing") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            onDragOver={(e) => {
              e.preventDefault();
              if (status !== "processing") setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`rounded-xl border-2 border-dashed p-8 text-center flex flex-col items-center justify-center min-h-[220px] transition ${
              status === "processing" ? "cursor-not-allowed opacity-60" : "cursor-pointer"
            } ${
              isDragging
                ? "border-violet-500 bg-violet-50 dark:bg-violet-950/40"
                : "border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 hover:border-violet-400 hover:bg-violet-50/50 dark:hover:bg-violet-950/20"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => selectFile(e.target.files?.[0])}
            />
            <div className="w-14 h-14 rounded-full bg-violet-100 dark:bg-violet-950/60 border border-violet-200 dark:border-violet-800 flex items-center justify-center text-violet-600 dark:text-violet-400 mb-3">
              <Upload className="w-6 h-6" />
            </div>
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
              {isDragging ? "Drop your CSV file here" : "Drag & drop a CSV file, or click to browse"}
            </h4>
            <p className="text-xs text-slate-400 mt-1">Accepted format: .csv · Max size 5 MB · 10 to 5,000 customers</p>
          </div>

          {/* Selected file */}
          {shownName && (
            <div className="mt-4 flex items-center justify-between gap-3 p-3 rounded-lg bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700">
              <div className="flex items-center gap-3 min-w-0">
                <FileSpreadsheet className="w-8 h-8 text-emerald-600 shrink-0" />
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-slate-800 dark:text-slate-100 truncate">{shownName}</p>
                  <p className="text-xs text-slate-400">{file ? formatBytes(file.size) : "Currently loaded dataset"}</p>
                </div>
              </div>
              {status !== "processing" && (
                <button
                  type="button"
                  onClick={file ? handleClearSelection : handleReset}
                  aria-label="Remove file"
                  className="p-1.5 rounded-md text-slate-400 hover:text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          )}

          <div className="mt-5 flex flex-col sm:flex-row gap-3">
            <button
              type="button"
              onClick={handleProcess}
              disabled={!file || status === "processing"}
              className="flex-1 py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-violet-600 via-indigo-600 to-blue-600 hover:opacity-95 shadow-md shadow-indigo-500/20 disabled:opacity-50 transition flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
            >
              {status === "processing" ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Processing Dataset...
                </>
              ) : (
                <>
                  <ScanSearch className="w-4 h-4" /> Process &amp; Predict Segments
                </>
              )}
            </button>
            <button
              type="button"
              onClick={() => downloadText("customer_template.csv", TEMPLATE_CSV)}
              className="px-4 py-3 rounded-xl text-sm font-semibold text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700 transition flex items-center justify-center gap-2 cursor-pointer"
            >
              <Download className="w-4 h-4" /> Sample CSV
            </button>
          </div>

          {status === "error" && error && (
            <div className="mt-4 p-3.5 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200 text-sm flex items-start gap-2.5">
              <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold block">Processing Failed</span>
                {error}
              </div>
            </div>
          )}
        </div>

        {/* Loader / status column */}
        <div className="lg:col-span-5">
          {status === "processing" ? (
            <div className="bg-white dark:bg-slate-900 border-2 border-violet-500/60 rounded-xl p-6 shadow-lg shadow-violet-500/10 min-h-[340px]">
              <div className="flex items-center gap-3 mb-5">
                <div className="relative w-12 h-12 shrink-0">
                  <div className="absolute inset-0 rounded-full border-4 border-violet-100 dark:border-violet-950" />
                  <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-violet-600 animate-spin" />
                </div>
                <div>
                  <h4 className="text-base font-bold text-slate-900 dark:text-white">ML Model Processing</h4>
                  <p className="text-xs text-slate-500">Running K-Means on your dataset…</p>
                </div>
              </div>

              <div className="mb-5">
                <div className="flex justify-between text-xs font-semibold mb-1.5">
                  <span className="text-slate-600 dark:text-slate-300">Progress</span>
                  <span className="font-mono text-violet-600 dark:text-violet-400">{progressPct}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-violet-600 to-blue-600 transition-all duration-500 ease-out"
                    style={{ width: `${Math.max(progressPct, 4)}%` }}
                  />
                </div>
              </div>

              <ol className="space-y-3">
                {PROCESSING_STAGES.map((s, idx) => {
                  const isDone = idx < stage;
                  const isActive = idx === stage;
                  return (
                    <li key={s.title} className="flex items-start gap-3">
                      {isDone ? (
                        <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
                      ) : isActive ? (
                        <Loader2 className="w-5 h-5 text-violet-600 animate-spin shrink-0" />
                      ) : (
                        <CircleDashed className="w-5 h-5 text-slate-300 dark:text-slate-600 shrink-0" />
                      )}
                      <div>
                        <p
                          className={`text-sm font-semibold ${
                            isDone
                              ? "text-slate-700 dark:text-slate-200"
                              : isActive
                              ? "text-violet-700 dark:text-violet-300"
                              : "text-slate-400"
                          }`}
                        >
                          {s.title}
                        </p>
                        <p className="text-xs text-slate-400">{s.detail}</p>
                      </div>
                    </li>
                  );
                })}
              </ol>
            </div>
          ) : summary && clusteringData ? (
            <div className="bg-white dark:bg-slate-900 border-2 border-emerald-500/60 rounded-xl p-6 shadow-lg shadow-emerald-500/10 space-y-4 min-h-[340px]">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                Processing Complete
              </span>
              <div>
                <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                  {summary.processed_rows} customers segmented
                </h3>
                <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 mt-1.5 leading-relaxed">
                  Grouped into <strong>{clusteringData.k} clusters</strong> with a silhouette score of{" "}
                  <strong>{clusteringData.silhouette_score}</strong>.
                  {clusteringData.accuracy !== null && clusteringData.accuracy !== undefined && (
                    <>
                      {" "}Accuracy against the true segments:{" "}
                      <strong className="text-emerald-600 dark:text-emerald-400">
                        {(clusteringData.accuracy * 100).toFixed(1)}%
                      </strong>
                      .
                    </>
                  )}
                </p>
              </div>
              <div className="bg-slate-50 dark:bg-slate-800/80 p-3 rounded-lg text-xs space-y-1 border border-slate-100 dark:border-slate-800">
                <span className="font-semibold text-slate-700 dark:text-slate-200 block">Detected Columns:</span>
                <p className="text-slate-500">
                  Income: <strong className="font-mono">{summary.detected_columns.income}</strong> · Spending:{" "}
                  <strong className="font-mono">{summary.detected_columns.spending}</strong>
                </p>
                <p className="text-slate-500">
                  ID: <strong className="font-mono">{summary.detected_columns.id ?? "—"}</strong> · Age:{" "}
                  <strong className="font-mono">{summary.detected_columns.age ?? "—"}</strong>
                </p>
                {summary.detected_columns.label && (
                  <p className="text-slate-500">
                    True segments (for accuracy only):{" "}
                    <strong className="font-mono">{summary.detected_columns.label}</strong>
                  </p>
                )}
              </div>
              {summary.skipped_rows > 0 && (
                <p className="text-xs text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded-lg p-2.5">
                  {summary.skipped_rows} row(s) skipped because Income or Spending Score was missing or non-numeric.
                </p>
              )}
              <div className="flex flex-col sm:flex-row gap-2 pt-1">
                <button
                  type="button"
                  onClick={handleDownloadResults}
                  className="flex-1 py-2.5 rounded-lg text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-700 transition flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <Download className="w-4 h-4" /> Download Predictions CSV
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  className="flex-1 py-2.5 rounded-lg text-xs font-semibold text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700 transition flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <RotateCcw className="w-4 h-4" /> Upload Another File
                </button>
              </div>
              <a
                href="#dataset"
                className="flex items-center justify-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:underline"
              >
                View full results below <ArrowDown className="w-3.5 h-3.5" />
              </a>
            </div>
          ) : (
            <div className="bg-slate-50 dark:bg-slate-800/40 border border-dashed border-slate-300 dark:border-slate-700 rounded-xl p-8 text-center flex flex-col items-center justify-center min-h-[340px]">
              <div className="w-12 h-12 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center text-slate-500 mb-3">
                <FileUp className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Awaiting Dataset</h4>
              <p className="text-xs text-slate-400 max-w-xs mt-1">
                Choose a CSV file on the left, then click &quot;Process &amp; Predict Segments&quot;. Not sure about the
                format? Download the sample CSV.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
