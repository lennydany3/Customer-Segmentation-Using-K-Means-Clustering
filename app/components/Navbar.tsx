"use client";

import React from "react";
import { Activity, Database, LineChart, Cpu, UserCheck } from "lucide-react";

interface NavbarProps {
  isBackendHealthy: boolean | null;
  onRefreshHealth: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ isBackendHealthy, onRefreshHealth }) => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white/80 backdrop-blur-md dark:border-slate-800 dark:bg-slate-900/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <span className="font-bold text-lg text-slate-900 dark:text-white tracking-tight">
              Customer<span className="text-blue-600 dark:text-blue-400">Segment</span> ML
            </span>
            <span className="hidden sm:inline-block ml-2 text-xs font-medium px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
              K-Means Clustering
            </span>
          </div>
        </div>

        {/* Navigation links */}
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600 dark:text-slate-300">
          <a href="#dataset" className="hover:text-blue-600 dark:hover:text-blue-400 transition flex items-center gap-1.5">
            <Database className="w-4 h-4" /> Dataset
          </a>
          <a href="#elbow" className="hover:text-blue-600 dark:hover:text-blue-400 transition flex items-center gap-1.5">
            <LineChart className="w-4 h-4" /> Elbow Method
          </a>
          <a href="#clustering" className="hover:text-blue-600 dark:hover:text-blue-400 transition flex items-center gap-1.5">
            <Activity className="w-4 h-4" /> K-Means Model
          </a>
          <a href="#predict" className="hover:text-blue-600 dark:hover:text-blue-400 transition flex items-center gap-1.5">
            <UserCheck className="w-4 h-4" /> Predict
          </a>
        </nav>

        {/* Backend status badge */}
        <div className="flex items-center gap-3">
          <button
            onClick={onRefreshHealth}
            title="Click to recheck backend connectivity"
            className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold border transition shadow-sm cursor-pointer hover:opacity-90"
            style={{
              borderColor: isBackendHealthy === true ? "#86efac" : isBackendHealthy === false ? "#fca5a5" : "#e2e8f0",
              backgroundColor: isBackendHealthy === true ? "#f0fdf4" : isBackendHealthy === false ? "#fef2f2" : "#f8fafc",
              color: isBackendHealthy === true ? "#15803d" : isBackendHealthy === false ? "#b91c1c" : "#64748b",
            }}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendHealthy === true
                  ? "bg-emerald-500 animate-pulse"
                  : isBackendHealthy === false
                  ? "bg-rose-500"
                  : "bg-slate-400"
              }`}
            />
            {isBackendHealthy === true
              ? "FastAPI Online"
              : isBackendHealthy === false
              ? "Backend Offline"
              : "Checking..."}
          </button>
        </div>
      </div>
    </header>
  );
};

