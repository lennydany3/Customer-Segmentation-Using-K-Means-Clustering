"use client";

import React, { useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ZAxis,
} from "recharts";
import {
  Users,
  Layers,
  Award,
  Target,
  TrendingUp,
  ShoppingBag,
} from "lucide-react";
import { ClusterResponse, Point } from "../lib/types";

export const CLUSTER_PALETTE = [
  "#2563eb", // Royal Blue
  "#16a34a", // Emerald Green
  "#9333ea", // Purple
  "#ea580c", // Orange
  "#0891b2", // Cyan
  "#e11d48", // Rose Red
  "#ca8a04", // Amber Yellow
  "#4f46e5", // Indigo
];

interface ResultsSectionProps {
  clusteringData: ClusterResponse | null;
  predictedPoint?: {
    income: number;
    spending: number;
    label: string;
    cluster_id: number;
    age: number;
  } | null;
}

interface ScatterPayloadItem {
  payload: {
    id?: string;
    age?: number;
    income: number;
    spending: number;
    cluster?: number;
    clusterName?: string;
    label?: string;
    isCentroid?: boolean;
    isPredicted?: boolean;
  };
}

interface CustomScatterTooltipProps {
  active?: boolean;
  payload?: ScatterPayloadItem[];
}

// Custom Tooltip for Scatter Plot
const CustomScatterTooltip: React.FC<CustomScatterTooltipProps> = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isCentroid = data.isCentroid;
    const isPredicted = data.isPredicted;

    if (isPredicted) {
      return (
        <div className="bg-slate-900 text-white p-3 rounded-lg shadow-xl text-xs border border-pink-500">
          <div className="font-bold text-pink-400 flex items-center gap-1.5 mb-1">
            <span className="w-2 h-2 rounded-full bg-pink-400 animate-ping" />
            ★ Predicted Customer
          </div>
          <p><span className="text-slate-400">Assigned Segment:</span> {data.label}</p>
          <p><span className="text-slate-400">Income:</span> ${data.income}k</p>
          <p><span className="text-slate-400">Spending Score:</span> {data.spending}/100</p>
          <p><span className="text-slate-400">Age:</span> {data.age} yrs</p>
        </div>
      );
    }

    if (isCentroid) {
      return (
        <div className="bg-slate-950 text-white p-3 rounded-lg shadow-xl text-xs border border-yellow-400">
          <div className="font-bold text-yellow-400 mb-1">
            ◆ Cluster Centroid ({data.label})
          </div>
          <p><span className="text-slate-400">Mean Income:</span> ${data.income}k</p>
          <p><span className="text-slate-400">Mean Spending:</span> {data.spending}/100</p>
        </div>
      );
    }

    return (
      <div className="bg-slate-900 text-white p-2.5 rounded-lg shadow-xl text-xs border border-slate-700">
        <div className="font-semibold text-slate-200 mb-1">
          Customer {data.id}
        </div>
        <p><span className="text-slate-400">Cluster:</span> {data.clusterName || `Cluster ${data.cluster}`}</p>
        <p><span className="text-slate-400">Annual Income:</span> ${data.income}k</p>
        <p><span className="text-slate-400">Spending Score:</span> {data.spending}/100</p>
        <p><span className="text-slate-400">Age:</span> {data.age} yrs</p>
      </div>
    );
  }
  return null;
};

export const ResultsSection: React.FC<ResultsSectionProps> = ({
  clusteringData,
  predictedPoint,
}) => {
  const clusters = useMemo(() => clusteringData?.clusters ?? [], [clusteringData]);
  const points = useMemo(() => clusteringData?.points ?? [], [clusteringData]);
  const centroids = useMemo(() => clusteringData?.centroids ?? [], [clusteringData]);

  // Group customer points by cluster
  const pointsByCluster = useMemo(() => {
    const grouped: { [key: number]: (Point & { clusterName: string })[] } = {};
    clusters.forEach((c) => {
      grouped[c.cluster_id] = [];
    });

    points.forEach((p) => {
      if (grouped[p.cluster]) {
        const clusterMeta = clusters.find((c) => c.cluster_id === p.cluster);
        grouped[p.cluster].push({
          ...p,
          clusterName: clusterMeta ? clusterMeta.label : `Cluster ${p.cluster}`,
        });
      }
    });
    return grouped;
  }, [points, clusters]);

  // Centroid dataset formatted for chart
  const centroidData = useMemo(() => {
    return centroids.map((c) => ({
      ...c,
      isCentroid: true,
      id: `Centroid-${c.cluster}`,
      age: 0,
    }));
  }, [centroids]);

  // Prediction point formatted for chart
  const predictionChartData = useMemo(() => {
    if (!predictedPoint) return [];
    return [
      {
        ...predictedPoint,
        isPredicted: true,
        id: "New-Customer",
      },
    ];
  }, [predictedPoint]);

  if (!clusteringData) return null;

  const { k, total_customers, silhouette_score } = clusteringData;

  return (
    <div className="space-y-8">
      {/* 1. Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Total Customers */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Analyzed Customers
            </span>
            <span className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1 block">
              {total_customers}
            </span>
            <span className="text-xs text-slate-400">100% of dataset clustered</span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-blue-50 dark:bg-blue-950/60 border border-blue-200 dark:border-blue-800 flex items-center justify-center text-blue-600 dark:text-blue-400">
            <Users className="w-6 h-6" />
          </div>
        </div>

        {/* Selected K */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Clusters Formed (K)
            </span>
            <span className="text-2xl font-extrabold text-indigo-600 dark:text-indigo-400 mt-1 block">
              {k} Clusters
            </span>
            <span className="text-xs text-slate-400">
              {k === 4 ? "Optimal Elbow configuration" : "User-defined hyperparameter"}
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
            <Layers className="w-6 h-6" />
          </div>
        </div>

        {/* Silhouette Score */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Silhouette Score
            </span>
            <span className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400 mt-1 block">
              {silhouette_score}
            </span>
            <span className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
              {silhouette_score >= 0.7
                ? "Excellent Separation"
                : silhouette_score >= 0.5
                ? "Good Structure"
                : "Moderate Overlap"}
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 border border-emerald-200 dark:border-emerald-800 flex items-center justify-center text-emerald-600 dark:text-emerald-400">
            <Award className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* 2. Scatter Plot: Annual Income vs Spending Score */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-600" />
              Customer Segments Scatter Plot
            </h3>
            <p className="text-xs text-slate-500">
              Annual Income ($k) vs Spending Score (1-100). Diamonds (◆) denote cluster centroids.
              {predictedPoint && (
                <span className="text-pink-600 font-semibold ml-1">
                  ★ Highlighted star denotes newly predicted customer!
                </span>
              )}
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rotate-45 bg-amber-500 inline-block" /> Centroid
            </span>
            {predictedPoint && (
              <span className="flex items-center gap-1 text-pink-600 font-bold ml-2">
                ★ New Customer
              </span>
            )}
          </div>
        </div>

        <div className="h-96 sm:h-[450px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                type="number"
                dataKey="income"
                name="Annual Income"
                unit="k$"
                domain={[10, 140]}
                tickLine={false}
                stroke="#64748b"
                fontSize={12}
                label={{
                  value: "Annual Income ($k)",
                  position: "insideBottom",
                  offset: -10,
                  fontSize: 12,
                  fill: "#64748b",
                }}
              />
              <YAxis
                type="number"
                dataKey="spending"
                name="Spending Score"
                domain={[0, 105]}
                tickLine={false}
                stroke="#64748b"
                fontSize={12}
                label={{
                  value: "Spending Score (1-100)",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 12,
                  fill: "#64748b",
                }}
              />
              <ZAxis range={[60, 60]} />
              <Tooltip content={<CustomScatterTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: "12px", paddingTop: "15px" }}
                iconType="circle"
              />

              {/* Data points for each cluster */}
              {clusters.map((c, idx) => (
                <Scatter
                  key={`cluster-${c.cluster_id}`}
                  name={c.label}
                  data={pointsByCluster[c.cluster_id] || []}
                  fill={CLUSTER_PALETTE[idx % CLUSTER_PALETTE.length]}
                  shape="circle"
                />
              ))}

              {/* Centroids highlighted */}
              <Scatter
                name="Cluster Centroids"
                data={centroidData}
                fill="#f59e0b"
                shape="diamond"
              />

              {/* Highlighted new predicted customer point */}
              {predictedPoint && (
                <Scatter
                  name="★ Predicted Customer"
                  data={predictionChartData}
                  fill="#ec4899"
                  shape="star"
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Cluster Summary & Strategy Cards */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <ShoppingBag className="w-5 h-5 text-indigo-600" />
          <h3 className="text-xl font-bold text-slate-900 dark:text-white">
            Cluster Profiles &amp; Marketing Strategies
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {clusters.map((cluster, idx) => {
            const color = CLUSTER_PALETTE[idx % CLUSTER_PALETTE.length];
            return (
              <div
                key={cluster.cluster_id}
                className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs hover:shadow-md transition flex flex-col justify-between"
                style={{ borderTop: `4px solid ${color}` }}
              >
                <div>
                  {/* Card Header */}
                  <div className="flex items-start justify-between gap-3 mb-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <span
                          className="w-3 h-3 rounded-full shrink-0"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                          Cluster {cluster.cluster_id}
                        </span>
                      </div>
                      <h4 className="text-lg font-bold text-slate-900 dark:text-white mt-0.5">
                        {cluster.label}
                      </h4>
                    </div>

                    <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                      {cluster.customer_count} customers ({cluster.percentage}%)
                    </span>
                  </div>

                  {/* Persona description */}
                  <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 mb-4 leading-relaxed">
                    {cluster.description}
                  </p>

                  {/* Numerical averages */}
                  <div className="grid grid-cols-3 gap-2 bg-slate-50 dark:bg-slate-800/60 p-3 rounded-lg text-center mb-4 border border-slate-100 dark:border-slate-800">
                    <div>
                      <span className="text-[11px] text-slate-400 block">Avg Income</span>
                      <span className="text-sm font-bold text-slate-800 dark:text-slate-200">
                        ${cluster.avg_income}k
                      </span>
                    </div>
                    <div>
                      <span className="text-[11px] text-slate-400 block">Avg Spending</span>
                      <span className="text-sm font-bold text-slate-800 dark:text-slate-200">
                        {cluster.avg_spending}/100
                      </span>
                    </div>
                    <div>
                      <span className="text-[11px] text-slate-400 block">Avg Age</span>
                      <span className="text-sm font-bold text-slate-800 dark:text-slate-200">
                        {cluster.avg_age} yrs
                      </span>
                    </div>
                  </div>
                </div>

                {/* Marketing Strategy Callout */}
                <div className="bg-indigo-50/70 dark:bg-indigo-950/40 p-3.5 rounded-lg border border-indigo-100 dark:border-indigo-900/60">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-indigo-700 dark:text-indigo-300 mb-1">
                    <TrendingUp className="w-3.5 h-3.5" />
                    Recommended Marketing Strategy
                  </div>
                  <p className="text-xs text-indigo-950 dark:text-indigo-200 leading-normal">
                    {cluster.strategy}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

