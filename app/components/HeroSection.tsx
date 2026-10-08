"use client";

import React from "react";
import { ArrowDown, Sparkles, Sliders, Layers } from "lucide-react";

export const HeroSection: React.FC = () => {
  return (
    <section className="relative overflow-hidden py-12 lg:py-16 bg-gradient-to-b from-blue-50/60 via-white to-slate-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-950 border-b border-slate-200 dark:border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-blue-100/80 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 mb-4 border border-blue-200 dark:border-blue-700">
            <Sparkles className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
            Machine Learning Mini-Project
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-tight">
            Customer Segmentation Using{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 dark:from-blue-400 dark:to-purple-400">
              K-Means Clustering
            </span>
          </h1>

          <p className="mt-4 text-base sm:text-lg text-slate-600 dark:text-slate-300 leading-relaxed">
            K-Means is an unsupervised machine learning algorithm that groups customers into distinct clusters by minimizing within-cluster variance (Euclidean distance to cluster centroids). By analyzing <strong>Annual Income</strong> and <strong>Spending Score</strong>, businesses can uncover high-value VIPs, conservative spenders, and budget shoppers to tailor personalized marketing strategies.
          </p>

          {/* Viva Concepts quick badges */}
          <div className="mt-6 flex flex-wrap gap-2 text-xs font-medium text-slate-600 dark:text-slate-400">
            <span className="px-2.5 py-1 rounded-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-xs flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-blue-500" /> Unsupervised Learning
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-xs flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-indigo-500" /> StandardScaler Normalization
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-xs flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-purple-500" /> WCSS & Silhouette Score
            </span>
          </div>

          {/* Jump Buttons */}
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href="#dataset"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 shadow-sm transition"
            >
              Explore Dataset <ArrowDown className="w-4 h-4" />
            </a>
            <a
              href="#elbow"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700 shadow-xs transition"
            >
              Elbow Method
            </a>
            <a
              href="#clustering"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700 shadow-xs transition"
            >
              Run K-Means
            </a>
            <a
              href="#predict"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800 hover:bg-indigo-100 transition"
            >
              Predict Segment
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

