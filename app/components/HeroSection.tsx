"use client";

import React from "react";
import {
  ArrowRight,
  Sparkles,
  Database,
  LineChart,
  Activity,
  UserCheck,
  FileUp,
  Monitor,
  Server,
  BrainCircuit,
  ArrowDown,
} from "lucide-react";
import { ClusterResponse } from "../lib/types";
import { CLUSTER_PALETTE } from "./ResultsSection";

interface HeroSectionProps {
  isBackendHealthy: boolean | null;
  hasDataset: boolean;
  totalCustomers: number;
  clusteringData: ClusterResponse | null;
}

// The end-to-end workflow of the system; each step links to its dashboard section
const PIPELINE_STEPS = [
  {
    href: "#upload",
    icon: FileUp,
    title: "Upload CSV",
    detail: "Upload your customer dataset: age & repeat purchases",
    tone: "text-violet-600 dark:text-violet-400 bg-violet-50 dark:bg-violet-950/60 border-violet-200 dark:border-violet-800",
  },
  {
    href: "#dataset",
    icon: Database,
    title: "Dataset",
    detail: "Review your customers and the segment each one gets",
    tone: "text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/60 border-blue-200 dark:border-blue-800",
  },
  {
    href: "#elbow",
    icon: LineChart,
    title: "Elbow Method",
    detail: "WCSS & silhouette for K = 2…8 to choose the best K",
    tone: "text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/60 border-indigo-200 dark:border-indigo-800",
  },
  {
    href: "#clustering",
    icon: Activity,
    title: "Train K-Means",
    detail: "StandardScaler + KMeans, scored by Silhouette",
    tone: "text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-950/60 border-purple-200 dark:border-purple-800",
  },
  {
    href: "#predict",
    icon: UserCheck,
    title: "Predict One",
    detail: "Classify a single new customer in real time",
    tone: "text-pink-600 dark:text-pink-400 bg-pink-50 dark:bg-pink-950/60 border-pink-200 dark:border-pink-800",
  },
];

// The three tiers of the application architecture
const ARCHITECTURE_LAYERS = [
  {
    icon: Monitor,
    title: "Next.js Dashboard",
    detail: "React · Tailwind CSS · Recharts",
    tone: "text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/60 border-blue-200 dark:border-blue-800",
  },
  {
    icon: Server,
    title: "FastAPI REST API",
    detail: "REST endpoints · Pydantic validation",
    tone: "text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/60 border-emerald-200 dark:border-emerald-800",
  },
  {
    icon: BrainCircuit,
    title: "scikit-learn ML Model",
    detail: "StandardScaler → KMeans → Silhouette",
    tone: "text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-950/60 border-purple-200 dark:border-purple-800",
  },
];

export const HeroSection: React.FC<HeroSectionProps> = ({
  isBackendHealthy,
  hasDataset,
  totalCustomers,
  clusteringData,
}) => {
  const liveStats = [
    {
      label: "Customers",
      value: totalCustomers ? String(totalCustomers) : "—",
    },
    {
      label: "Clusters (K)",
      value: clusteringData ? String(clusteringData.k) : "—",
    },
    {
      label: "Silhouette",
      value: clusteringData ? clusteringData.silhouette_score.toFixed(3) : "—",
    },
  ];

  return (
    <section className="relative overflow-hidden border-b border-slate-200 dark:border-slate-800 bg-gradient-to-b from-blue-50/70 via-white to-slate-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-950">
      {/* Decorative background blobs */}
      <div aria-hidden className="pointer-events-none absolute -top-24 -right-24 w-96 h-96 rounded-full bg-indigo-400/20 dark:bg-indigo-600/10 blur-3xl" />
      <div aria-hidden className="pointer-events-none absolute -bottom-32 -left-24 w-96 h-96 rounded-full bg-violet-400/20 dark:bg-violet-600/10 blur-3xl" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
          {/* Intro column */}
          <div className="lg:col-span-7">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-blue-100/80 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 mb-4 border border-blue-200 dark:border-blue-700">
              <Sparkles className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
              Full-Stack Machine Learning Project
            </div>

            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-tight">
              Customer Segmentation Using{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 dark:from-blue-400 dark:to-purple-400">
                K-Means Clustering
              </span>
            </h1>

            <p className="mt-4 text-base sm:text-lg text-slate-600 dark:text-slate-300 leading-relaxed max-w-2xl">
              Upload your customer CSV and the model groups every customer by <strong>Age</strong> and{" "}
              <strong>Repeat Purchases</strong> into actionable market segments, each with a marketing strategy.
            </p>

            {/* Calls to action */}
            <div className="mt-8 flex flex-col sm:flex-row gap-3">
              <a
                href="#upload"
                className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-violet-600 via-indigo-600 to-blue-600 hover:opacity-95 shadow-md shadow-indigo-500/20 transition"
              >
                <FileUp className="w-4 h-4" /> Upload Your Dataset
              </a>
              {hasDataset && (
                <a
                  href="#dataset"
                  className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl text-sm font-semibold text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700 shadow-xs transition"
                >
                  View Results <ArrowDown className="w-4 h-4" />
                </a>
              )}
            </div>

            {/* Live model snapshot */}
            <div className="mt-8 grid grid-cols-3 gap-3 max-w-lg">
              {liveStats.map((stat) => (
                <div
                  key={stat.label}
                  className="bg-white/80 dark:bg-slate-900/80 backdrop-blur border border-slate-200 dark:border-slate-800 rounded-xl px-4 py-3 shadow-xs"
                >
                  <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                    {stat.label}
                  </span>
                  <span
                    className={`text-xl font-extrabold font-mono ${
                      stat.value === "—" ? "text-slate-300 dark:text-slate-600 animate-pulse" : "text-slate-900 dark:text-white"
                    }`}
                  >
                    {stat.value}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* System architecture column */}
          <div className="lg:col-span-5">
            <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-lg shadow-slate-900/5">
              <div className="flex items-center justify-between mb-5">
                <h2 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider">
                  System Architecture
                </h2>
                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold border ${
                    isBackendHealthy === true
                      ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/60 dark:text-emerald-300 dark:border-emerald-800"
                      : isBackendHealthy === false
                      ? "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/60 dark:text-rose-300 dark:border-rose-800"
                      : "bg-slate-50 text-slate-500 border-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:border-slate-700"
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      isBackendHealthy === true
                        ? "bg-emerald-500 animate-pulse"
                        : isBackendHealthy === false
                        ? "bg-rose-500"
                        : "bg-slate-400"
                    }`}
                  />
                  {isBackendHealthy === true ? "API Online" : isBackendHealthy === false ? "API Offline" : "Connecting"}
                </span>
              </div>

              <ol>
                {ARCHITECTURE_LAYERS.map((layer, idx) => (
                  <li key={layer.title}>
                    <div className="flex items-center gap-3 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800">
                      <div className={`w-10 h-10 rounded-lg border flex items-center justify-center shrink-0 ${layer.tone}`}>
                        <layer.icon className="w-5 h-5" />
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-slate-900 dark:text-white">{layer.title}</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 font-mono sm:truncate">{layer.detail}</p>
                      </div>
                    </div>
                    {idx < ARCHITECTURE_LAYERS.length - 1 && (
                      <div aria-hidden className="flex justify-center py-1 text-slate-300 dark:text-slate-600">
                        <ArrowDown className="w-4 h-4" />
                      </div>
                    )}
                  </li>
                ))}
              </ol>

              {/* Segment mix of the trained model */}
              <div className="mt-5 pt-5 border-t border-slate-100 dark:border-slate-800">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="font-semibold text-slate-700 dark:text-slate-200">Current Segment Mix</span>
                  <span className="text-slate-400">{clusteringData ? `K = ${clusteringData.k}` : "Awaiting model"}</span>
                </div>
                <div className="flex h-3 w-full rounded-full overflow-hidden bg-slate-100 dark:bg-slate-800">
                  {clusteringData?.clusters.map((c) => (
                    <div
                      key={c.cluster_id}
                      title={`${c.label}: ${c.percentage}%`}
                      style={{
                        width: `${c.percentage}%`,
                        backgroundColor: CLUSTER_PALETTE[c.cluster_id % CLUSTER_PALETTE.length],
                      }}
                    />
                  ))}
                </div>
                {clusteringData && (
                  <ul className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1.5">
                    {clusteringData.clusters.map((c) => (
                      <li key={c.cluster_id} className="flex items-center gap-2 text-[11px] text-slate-600 dark:text-slate-300 min-w-0">
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ backgroundColor: CLUSTER_PALETTE[c.cluster_id % CLUSTER_PALETTE.length] }}
                        />
                        <span className="truncate">{c.label}</span>
                        <span className="ml-auto font-mono text-slate-400">{c.percentage}%</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* End-to-end workflow */}
        <div className="mt-12">
          <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">How the system works</h2>
          <ol className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {PIPELINE_STEPS.map((step, idx) => (
              <li key={step.href}>
                <a
                  href={step.href}
                  aria-disabled={!(step.href === "#upload" || hasDataset)}
                  onClick={(e) => {
                    if (!(step.href === "#upload" || hasDataset)) e.preventDefault();
                  }}
                  className={`group h-full flex flex-col bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs transition ${
                    step.href === "#upload" || hasDataset
                      ? "hover:shadow-md hover:-translate-y-0.5 hover:border-indigo-300 dark:hover:border-indigo-700"
                      : "opacity-60 cursor-default"
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className={`w-9 h-9 rounded-lg border flex items-center justify-center ${step.tone}`}>
                      <step.icon className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-mono font-bold text-slate-300 dark:text-slate-600">0{idx + 1}</span>
                  </div>
                  <p className="text-sm font-semibold text-slate-900 dark:text-white flex items-center gap-1">
                    {step.title}
                    <ArrowRight className="w-3.5 h-3.5 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition" />
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">{step.detail}</p>
                </a>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
};
