"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Navbar } from "./components/Navbar";
import { HeroSection } from "./components/HeroSection";
import { DatasetSection } from "./components/DatasetSection";
import { ElbowSection } from "./components/ElbowSection";
import { ClusteringControl } from "./components/ClusteringControl";
import { ResultsSection } from "./components/ResultsSection";
import { PredictSection } from "./components/PredictSection";
import {
  getHealth,
  getDataset,
  getElbowCurve,
  runKMeans,
  predictCustomer,
} from "./lib/api";
import {
  Customer,
  ElbowResponse,
  ClusterResponse,
  PredictResponse,
} from "./lib/types";
import { AlertTriangle, RefreshCw, BookOpen } from "lucide-react";

export default function Home() {
  // Backend health status
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);

  // Dataset state
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [isDatasetLoading, setIsDatasetLoading] = useState<boolean>(true);

  // Elbow method state
  const [elbowData, setElbowData] = useState<ElbowResponse | null>(null);
  const [isElbowLoading, setIsElbowLoading] = useState<boolean>(true);
  const [elbowError, setElbowError] = useState<string | null>(null);

  // Clustering state
  const [k, setK] = useState<number>(4);
  const [clusteringData, setClusteringData] = useState<ClusterResponse | null>(null);
  const [isClusteringLoading, setIsClusteringLoading] = useState<boolean>(false);
  const [clusteringError, setClusteringError] = useState<string | null>(null);

  // Real-time prediction state
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<PredictResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [predictedScatterPoint, setPredictedScatterPoint] = useState<{
    income: number;
    spending: number;
    label: string;
    cluster_id: number;
    age: number;
  } | null>(null);

  // Reload data triggered by user actions
  const handleReloadData = useCallback(async () => {
    setIsDatasetLoading(true);
    setIsElbowLoading(true);
    setIsClusteringLoading(true);

    try {
      await getHealth();
      setIsBackendHealthy(true);

      const [ds, elbow, clusters] = await Promise.all([
        getDataset(),
        getElbowCurve(),
        runKMeans(k),
      ]);

      setCustomers(ds.customers);
      setTotalCount(ds.total_customers);
      setElbowData(elbow);
      setClusteringData(clusters);
    } catch (err: unknown) {
      setIsBackendHealthy(false);
      const msg = err instanceof Error ? err.message : "Failed to connect to backend.";
      setElbowError(msg);
      setClusteringError(msg);
    } finally {
      setIsDatasetLoading(false);
      setIsElbowLoading(false);
      setIsClusteringLoading(false);
    }
  }, [k]);

  // Initial load on mount
  useEffect(() => {
    let isMounted = true;

    const init = async () => {
      try {
        await getHealth();
        if (!isMounted) return;
        setIsBackendHealthy(true);

        const [ds, elbow, clusters] = await Promise.all([
          getDataset(),
          getElbowCurve(),
          runKMeans(4),
        ]);

        if (!isMounted) return;
        setCustomers(ds.customers);
        setTotalCount(ds.total_customers);
        setElbowData(elbow);
        setClusteringData(clusters);
      } catch (err: unknown) {
        if (!isMounted) return;
        setIsBackendHealthy(false);
        const msg = err instanceof Error ? err.message : "Failed to connect to backend.";
        setElbowError(msg);
        setClusteringError(msg);
      } finally {
        if (isMounted) {
          setIsDatasetLoading(false);
          setIsElbowLoading(false);
          setIsClusteringLoading(false);
        }
      }
    };

    init();

    return () => {
      isMounted = false;
    };
  }, []);

  // Handle re-training K-Means
  const handleRunKMeans = async (selectedK?: number) => {
    const targetK = selectedK ?? k;
    setIsClusteringLoading(true);
    setClusteringError(null);
    try {
      const result = await runKMeans(targetK);
      setClusteringData(result);
      setIsBackendHealthy(true);

      // If user had a predicted point, re-classify it under new clusters
      if (predictedScatterPoint) {
        try {
          const rePred = await predictCustomer({
            age: predictedScatterPoint.age,
            income: predictedScatterPoint.income,
            spending_score: predictedScatterPoint.spending,
          });
          setPredictionResult(rePred);
          setPredictedScatterPoint({
            income: rePred.income,
            spending: rePred.spending_score,
            label: rePred.label,
            cluster_id: rePred.cluster_id,
            age: rePred.age,
          });
        } catch {
          // Keep previous or clear
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Clustering failed.";
      setClusteringError(msg);
    } finally {
      setIsClusteringLoading(false);
    }
  };

  // Handle K selection from Elbow chart
  const handleSelectKFromElbow = (selectedK: number) => {
    setK(selectedK);
    handleRunKMeans(selectedK);
    const element = document.getElementById("clustering");
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  // Handle customer prediction
  const handlePredict = async (data: {
    age: number;
    income: number;
    spending_score: number;
  }): Promise<PredictResponse | null> => {
    setIsPredicting(true);
    setPredictionError(null);
    try {
      const res = await predictCustomer(data);
      setPredictionResult(res);
      setPredictedScatterPoint({
        income: res.income,
        spending: res.spending_score,
        label: res.label,
        cluster_id: res.cluster_id,
        age: res.age,
      });
      setIsBackendHealthy(true);

      // Smooth scroll to clustering chart to see highlighted point
      const el = document.getElementById("clustering");
      if (el) {
        el.scrollIntoView({ behavior: "smooth" });
      }

      return res;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Prediction failed.";
      setPredictionError(msg);
      return null;
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950">
      {/* Navigation */}
      <Navbar
        isBackendHealthy={isBackendHealthy}
        onRefreshHealth={handleReloadData}
      />

      {/* Hero Section */}
      <HeroSection />

      {/* Offline Alert Banner */}
      {isBackendHealthy === false && (
        <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-3 text-amber-900 dark:text-amber-200">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs sm:text-sm">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0" />
              <span>
                <strong>FastAPI backend is offline:</strong> Start it with{" "}
                <code className="bg-amber-100 dark:bg-amber-950/80 px-1.5 py-0.5 rounded font-mono font-semibold">
                  cd backend &amp;&amp; uvicorn main:app --reload --port 8000
                </code>
              </span>
            </div>
            <button
              onClick={handleReloadData}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-amber-600 text-white font-medium hover:bg-amber-700 transition cursor-pointer text-xs"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Retry Connection
            </button>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 space-y-12">
        {/* Section 1: Dataset */}
        <DatasetSection
          customers={customers}
          totalCount={totalCount}
          isLoading={isDatasetLoading}
          onRefresh={handleReloadData}
        />

        {/* Section 2: Elbow Method */}
        <ElbowSection
          data={elbowData}
          isLoading={isElbowLoading}
          error={elbowError}
          onSelectK={handleSelectKFromElbow}
        />

        {/* Section 3: Run K-Means & Visual Results */}
        <section id="clustering" className="py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-200 dark:border-slate-800 space-y-8">
          <div>
            <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 font-semibold text-sm mb-1">
              <span>Cluster Analysis</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white">
              K-Means Model Execution &amp; Results
            </h2>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
              Adjust K to re-partition the feature space. Review the scatter distribution, centroid positions, silhouette cohesion score, and targeted marketing strategies.
            </p>
          </div>

          <ClusteringControl
            k={k}
            onKChange={(val) => setK(val)}
            onRunKMeans={() => handleRunKMeans()}
            isLoading={isClusteringLoading}
            error={clusteringError}
          />

          <ResultsSection
            clusteringData={clusteringData}
            predictedPoint={predictedScatterPoint}
          />
        </section>

        {/* Section 4: Real-time Prediction */}
        <PredictSection
          onPredict={handlePredict}
          isPredicting={isPredicting}
          predictionResult={predictionResult}
          error={predictionError}
        />

        {/* Section 5: College Viva Quick Reference Card */}
        <section className="py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-200 dark:border-slate-800">
          <div className="bg-gradient-to-br from-indigo-900 via-slate-900 to-slate-950 text-white rounded-2xl p-6 sm:p-8 shadow-xl">
            <div className="flex items-center gap-2 text-indigo-400 font-semibold text-sm mb-2">
              <BookOpen className="w-5 h-5" />
              <span>Viva Voce Reference Guide</span>
            </div>
            <h3 className="text-xl sm:text-2xl font-bold mb-4">
              Core Machine Learning Concepts for Examination
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs sm:text-sm text-slate-300">
              <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                <h4 className="font-bold text-white mb-1.5 text-sm">1. What is K-Means?</h4>
                <p className="leading-relaxed text-slate-300 text-xs">
                  An unsupervised partitioning algorithm that divides $N$ samples into $K$ disjoint clusters, minimizing within-cluster inertia (sum of squared Euclidean distances to centroids).
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                <h4 className="font-bold text-white mb-1.5 text-sm">2. Why StandardScaler?</h4>
                <p className="leading-relaxed text-slate-300 text-xs">
                  Income ($15k-$140k) has a much larger scale than Spending Score (1-100). Without z-score normalization, income would dominate Euclidean distance computations.
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                <h4 className="font-bold text-white mb-1.5 text-sm">3. The Elbow Method</h4>
                <p className="leading-relaxed text-slate-300 text-xs">
                  We plot WCSS vs K. At the optimal $K=4$, adding further clusters only produces incremental diminishing returns, indicating natural underlying group structure.
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                <h4 className="font-bold text-white mb-1.5 text-sm">4. K-Means Limitations</h4>
                <p className="leading-relaxed text-slate-300 text-xs">
                  Assumes spherical clusters of similar size, is sensitive to initial centroid placement (mitigated by k-means++), and requires K to be predetermined.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4">
          Customer Segmentation Using K-Means Clustering • Built with FastAPI, scikit-learn, and Next.js
        </div>
      </footer>
    </div>
  );
}
