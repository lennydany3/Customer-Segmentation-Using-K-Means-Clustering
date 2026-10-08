"use client";

import React from "react";
import { Play, Loader2, AlertCircle, Sparkles } from "lucide-react";

interface ClusteringControlProps {
  k: number;
  onKChange: (k: number) => void;
  onRunKMeans: () => void;
  isLoading: boolean;
  error: string | null;
}

export const ClusteringControl: React.FC<ClusteringControlProps> = ({
  k,
  onKChange,
  onRunKMeans,
  isLoading,
  error,
}) => {
  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 font-semibold text-sm">
            <Sparkles className="w-4 h-4" />
            <span>Interactive Model Training</span>
          </div>
          <h3 className="text-xl font-bold text-slate-900 dark:text-white mt-1">
            Configure &amp; Train K-Means
          </h3>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-xl">
            Select the number of clusters <span className="font-mono font-semibold">K</span> (between 2 and 8). The algorithm will fit centroids via iterative expectation-maximization.
          </p>
        </div>

        {/* Input and Run button */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-4">
          <div className="flex items-center gap-3 bg-slate-50 dark:bg-slate-800/80 px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700">
            <label htmlFor="k-slider" className="text-xs font-semibold text-slate-600 dark:text-slate-300 whitespace-nowrap">
              Clusters (K): <span className="text-base font-bold text-blue-600 ml-1">{k}</span>
            </label>
            <input
              id="k-slider"
              type="range"
              min={2}
              max={8}
              step={1}
              value={k}
              onChange={(e) => onKChange(parseInt(e.target.value, 10))}
              disabled={isLoading}
              className="w-28 sm:w-36 accent-blue-600 cursor-pointer"
            />
          </div>

          <button
            onClick={onRunKMeans}
            disabled={isLoading}
            className="flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md shadow-blue-500/20 disabled:opacity-50 transition cursor-pointer disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Training Model...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" /> Run K-Means
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error message banner */}
      {error && (
        <div className="mt-4 p-3.5 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200 text-sm flex items-start gap-2.5">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Training Failed</span>
            {error}
          </div>
        </div>
      )}
    </div>
  );
};

