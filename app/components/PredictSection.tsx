"use client";

import React, { useState } from "react";
import { UserCheck, Sparkles, Send, CheckCircle2, AlertCircle } from "lucide-react";
import { PredictResponse } from "../lib/types";

export interface PredictBounds {
  ageMin: number;
  ageMax: number;
  repeatsMin: number;
  repeatsMax: number;
}

interface PredictSectionProps {
  bounds: PredictBounds;
  onPredict: (data: { age: number; repeat_purchases: number }) => Promise<PredictResponse | null>;
  isPredicting: boolean;
  predictionResult: PredictResponse | null;
  error: string | null;
}

export const PredictSection: React.FC<PredictSectionProps> = ({
  bounds,
  onPredict,
  isPredicting,
  predictionResult,
  error,
}) => {
  // Slider ranges follow the uploaded data
  const ageMin = Math.floor(bounds.ageMin);
  const ageMax = Math.max(ageMin + 1, Math.ceil(bounds.ageMax));
  const repeatsMin = Math.floor(bounds.repeatsMin);
  const repeatsMax = Math.max(repeatsMin + 1, Math.ceil(bounds.repeatsMax));

  const [age, setAge] = useState<number>(Math.round((ageMin + ageMax) / 2));
  const [repeats, setRepeats] = useState<number>(Math.round((repeatsMin + repeatsMax) / 2));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onPredict({ age: Number(age), repeat_purchases: Number(repeats) });
  };

  return (
    <section id="predict" className="py-12 scroll-mt-16 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-200 dark:border-slate-800">
      <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-semibold text-sm mb-1">
        <UserCheck className="w-4 h-4" />
        <span>Step 5 · Real-Time Inference</span>
      </div>
      <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white">
        Predict New Customer Segment
      </h2>
      <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-2xl mb-8">
        Enter a customer&apos;s age and number of repeat purchases. The system normalizes their features with the trained StandardScaler and assigns them to the closest cluster centroid via Euclidean distance.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Form Column (7 cols) */}
        <div className="lg:col-span-7 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs">
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Age Input */}
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                  Customer Age (Years)
                </label>
                <span className="text-xs font-mono font-bold text-blue-600 dark:text-blue-400">
                  {age} years
                </span>
              </div>
              <input
                type="range"
                min={ageMin}
                max={ageMax}
                value={age}
                onChange={(e) => setAge(Number(e.target.value))}
                className="w-full accent-blue-600 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-400 mt-1">
                <span>{ageMin} yrs</span>
                <span>{Math.round((ageMin + ageMax) / 2)} yrs</span>
                <span>{ageMax} yrs</span>
              </div>
            </div>

            {/* Repeat Purchases Input */}
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                  Number of Repeat Purchases
                </label>
                <span className="text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400">
                  {repeats} times
                </span>
              </div>
              <input
                type="range"
                min={repeatsMin}
                max={repeatsMax}
                value={repeats}
                onChange={(e) => setRepeats(Number(e.target.value))}
                className="w-full accent-indigo-600 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-400 mt-1">
                <span>{repeatsMin} (Rare)</span>
                <span>{Math.round((repeatsMin + repeatsMax) / 2)}</span>
                <span>{repeatsMax} (Loyal)</span>
              </div>
            </div>

            <button
              type="submit"
              disabled={isPredicting}
              className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:opacity-95 shadow-md shadow-indigo-500/20 disabled:opacity-50 transition flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
            >
              {isPredicting ? (
                <>Predicting Segment...</>
              ) : (
                <>
                  <Send className="w-4 h-4" /> Classify Customer &amp; Plot on Scatter
                </>
              )}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-3 rounded-lg bg-rose-50 dark:bg-rose-950/40 text-rose-800 dark:text-rose-200 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Prediction Results Column (5 cols) */}
        <div className="lg:col-span-5">
          {predictionResult ? (
            <div className="bg-white dark:bg-slate-900 border-2 border-indigo-500/60 rounded-xl p-6 shadow-lg shadow-indigo-500/10 space-y-4">
              <div className="flex items-center justify-between">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-pink-100 text-pink-800 dark:bg-pink-950/60 dark:text-pink-300">
                  <Sparkles className="w-3.5 h-3.5 text-pink-600" />
                  Cluster {predictionResult.cluster_id} Assigned
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  Age {predictionResult.age} | {predictionResult.repeat_purchases} repeats
                </span>
              </div>

              <div>
                <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                  {predictionResult.label}
                </h3>
                <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 mt-1.5 leading-relaxed">
                  {predictionResult.description}
                </p>
              </div>

              {/* Nearest centroid info */}
              <div className="bg-slate-50 dark:bg-slate-800/80 p-3 rounded-lg text-xs space-y-1 border border-slate-100 dark:border-slate-800">
                <span className="font-semibold text-slate-700 dark:text-slate-200 block">
                  Nearest Cluster Centroid:
                </span>
                <p className="text-slate-500">
                  Age: <strong>{predictionResult.nearest_centroid[0]} yrs</strong> | Repeat purchases:{" "}
                  <strong>{predictionResult.nearest_centroid[1]}</strong>
                </p>
              </div>

              {/* Marketing strategy */}
              <div className="bg-emerald-50 dark:bg-emerald-950/40 p-4 rounded-xl border border-emerald-200 dark:border-emerald-800 text-emerald-950 dark:text-emerald-100">
                <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-700 dark:text-emerald-300 mb-1">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  Actionable Strategy
                </div>
                <p className="text-xs leading-relaxed">
                  {predictionResult.strategy}
                </p>
              </div>

              <div className="pt-2 text-[11px] text-pink-600 dark:text-pink-400 font-medium text-center">
                ★ This customer is now plotted as a pink star on the scatter plot above!
              </div>
            </div>
          ) : (
            <div className="bg-slate-50 dark:bg-slate-800/40 border border-dashed border-slate-300 dark:border-slate-700 rounded-xl p-8 text-center flex flex-col items-center justify-center min-h-[340px]">
              <div className="w-12 h-12 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center text-slate-500 mb-3">
                <UserCheck className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-200">
                Awaiting Prediction
              </h4>
              <p className="text-xs text-slate-400 max-w-xs mt-1">
                Adjust the sliders on the left, then click &quot;Classify Customer&quot; to see real-time segmentation and strategy!
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

