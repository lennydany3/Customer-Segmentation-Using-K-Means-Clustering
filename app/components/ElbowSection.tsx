"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";
import { LineChart as LineChartIcon, CheckCircle, Info } from "lucide-react";
import { ElbowResponse } from "../lib/types";

interface ElbowSectionProps {
  data: ElbowResponse | null;
  isLoading: boolean;
  error: string | null;
  onSelectK?: (k: number) => void;
}

export const ElbowSection: React.FC<ElbowSectionProps> = ({
  data,
  isLoading,
  error,
  onSelectK,
}) => {
  // Format data for Recharts
  const chartData = data
    ? data.k_values.map((k, idx) => ({
        k: `K = ${k}`,
        kNum: k,
        wcss: data.wcss[idx],
      }))
    : [];

  return (
    <section id="elbow" className="py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-200 dark:border-slate-800">
      <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-semibold text-sm mb-1">
        <LineChartIcon className="w-4 h-4" />
        <span>Hyperparameter Tuning</span>
      </div>
      <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white">
        The Elbow Method (Finding Optimal K)
      </h2>
      <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-2xl mb-8">
        Plotting Within-Cluster Sum of Squares (WCSS / Inertia) against candidate values of K (2 through 8). The optimal K is selected at the &quot;elbow&quot; bend.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        {/* Chart Column (2 cols) */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                WCSS (Inertia) vs Number of Clusters (K)
              </h3>
              <span className="text-xs text-slate-500">
                Notice the distinct drop flattening out right after K = 4
              </span>
            </div>
            <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-900/60 dark:text-emerald-300">
              Optimal Elbow: K = 4
            </span>
          </div>

          <div className="h-72 sm:h-80 w-full">
            {isLoading ? (
              <div className="h-full flex items-center justify-center text-slate-400">
                Computing WCSS across candidate K values...
              </div>
            ) : error ? (
              <div className="h-full flex items-center justify-center text-rose-500 text-sm">
                {error}
              </div>
            ) : chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
                  margin={{ top: 20, right: 30, left: 10, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="k"
                    stroke="#64748b"
                    fontSize={12}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="#64748b"
                    fontSize={12}
                    tickLine={false}
                    label={{
                      value: "WCSS (Inertia)",
                      angle: -90,
                      position: "insideLeft",
                      fontSize: 12,
                      fill: "#64748b",
                    }}
                  />
                  <Tooltip
                    formatter={(value: unknown) => [`${value}`, "WCSS (Inertia)"]}
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      color: "#fff",
                      borderRadius: "8px",
                      fontSize: "12px",
                      border: "none",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="wcss"
                    stroke="#4f46e5"
                    strokeWidth={3}
                    dot={{ fill: "#4f46e5", r: 5 }}
                    activeDot={{ r: 8, stroke: "#818cf8", strokeWidth: 2 }}
                  />
                  {/* Highlight the elbow at K=4 */}
                  {data && data.k_values.includes(4) && (
                    <ReferenceDot
                      x="K = 4"
                      y={data.wcss[data.k_values.indexOf(4)]}
                      r={9}
                      fill="#10b981"
                      stroke="#ffffff"
                      strokeWidth={2}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-400">
                No elbow data available.
              </div>
            )}
          </div>

          {/* Quick interactive K selection helper */}
          {chartData.length > 0 && onSelectK && (
            <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs text-slate-500 font-medium">
                Click a K to test in model:
              </span>
              <div className="flex gap-1.5">
                {data?.k_values.map((k) => (
                  <button
                    key={k}
                    onClick={() => onSelectK(k)}
                    className={`px-2.5 py-1 rounded text-xs font-semibold cursor-pointer transition ${
                      k === 4
                        ? "bg-emerald-600 text-white hover:bg-emerald-700"
                        : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200"
                    }`}
                  >
                    K={k} {k === 4 && "★"}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Viva Notes Column (1 col) */}
        <div className="bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/60 rounded-xl p-5 space-y-4">
          <div className="flex items-center gap-2 text-slate-900 dark:text-white font-bold text-base">
            <Info className="w-5 h-5 text-indigo-500" />
            <span>How to Explain in Viva</span>
          </div>

          <div className="space-y-3 text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
            <div className="bg-white dark:bg-slate-800 p-3 rounded-lg border border-slate-200 dark:border-slate-700">
              <span className="font-semibold text-slate-900 dark:text-white block mb-1">
                1. What is WCSS?
              </span>
              <strong>Within-Cluster Sum of Squares (Inertia)</strong> measures cluster compactness: the sum of squared Euclidean distances between each point and its assigned cluster centroid.
            </div>

            <div className="bg-white dark:bg-slate-800 p-3 rounded-lg border border-slate-200 dark:border-slate-700">
              <span className="font-semibold text-slate-900 dark:text-white block mb-1">
                2. How to Read the Elbow?
              </span>
              As K increases, WCSS naturally decreases. The &quot;elbow point&quot; is the sweet spot where adding another cluster yields only marginal improvement (diminishing returns).
            </div>

            <div className="bg-emerald-50 dark:bg-emerald-950/40 p-3 rounded-lg border border-emerald-200 dark:border-emerald-800 text-emerald-900 dark:text-emerald-200">
              <div className="flex items-center gap-1.5 font-bold mb-1">
                <CheckCircle className="w-4 h-4 text-emerald-600" />
                Why K = 4 is Optimal Here
              </div>
              From K=2 to K=4, WCSS plunges steeply from <strong>215.2</strong> to <strong>31.58</strong>. Beyond K=4, the curve flattens out, proving 4 clusters naturally capture the underlying demographic groups.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
