"use client";

import React, { useState, useMemo } from "react";
import { Customer } from "../lib/types";
import { Database, Search, ChevronLeft, ChevronRight } from "lucide-react";

interface DatasetSectionProps {
  customers: Customer[];
  totalCount: number;
  isLoading: boolean;
  onRefresh: () => void;
}

export const DatasetSection: React.FC<DatasetSectionProps> = ({
  customers,
  totalCount,
  isLoading,
  onRefresh,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  // Filter customers by Customer ID
  const filteredCustomers = useMemo(() => {
    return customers.filter((c) =>
      c.customer_id.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [customers, searchTerm]);

  // Pagination calculation
  const totalPages = Math.ceil(filteredCustomers.length / pageSize) || 1;
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredCustomers.slice(start, start + pageSize);
  }, [filteredCustomers, currentPage, pageSize]);

  // Overall statistics
  const stats = useMemo(() => {
    if (customers.length === 0) return { avgAge: 0, avgIncome: 0, avgSpending: 0 };
    const avgAge = Math.round(customers.reduce((acc, c) => acc + c.age, 0) / customers.length);
    const avgIncome = Math.round(customers.reduce((acc, c) => acc + c.annual_income, 0) / customers.length);
    const avgSpending = Math.round(customers.reduce((acc, c) => acc + c.spending_score, 0) / customers.length);
    return { avgAge, avgIncome, avgSpending };
  }, [customers]);

  return (
    <section id="dataset" className="py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 font-semibold text-sm">
            <Database className="w-4 h-4" />
            <span>Dataset Overview</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white mt-1">
            Artificial Customer Data
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
            Synthetically synthesized dataset of {totalCount} records representing diverse consumer spending behaviors across 4 realistic demographic quadrants.
          </p>
        </div>

        {/* Quick summary stats */}
        <div className="grid grid-cols-3 gap-3 text-center sm:text-left">
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-2.5 px-4 shadow-xs">
            <span className="text-xs text-slate-500 block">Total Customers</span>
            <span className="text-lg font-bold text-slate-900 dark:text-white">{totalCount}</span>
          </div>
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-2.5 px-4 shadow-xs">
            <span className="text-xs text-slate-500 block">Avg Income</span>
            <span className="text-lg font-bold text-blue-600 dark:text-blue-400">${stats.avgIncome}k</span>
          </div>
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-2.5 px-4 shadow-xs">
            <span className="text-xs text-slate-500 block">Avg Spending</span>
            <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">{stats.avgSpending}/100</span>
          </div>
        </div>
      </div>

      {/* Table Card */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-xs overflow-hidden">
        {/* Search and filter toolbar */}
        <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-3 bg-slate-50/50 dark:bg-slate-900/50">
          <div className="relative w-full sm:w-64">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search by ID (e.g. C015)..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full pl-9 pr-3 py-1.5 text-sm rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center gap-2 self-end sm:self-auto text-xs text-slate-500 dark:text-slate-400">
            <span>Showing {paginatedData.length} of {filteredCustomers.length} customers</span>
            <button
              onClick={onRefresh}
              disabled={isLoading}
              className="px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium transition cursor-pointer"
            >
              {isLoading ? "Refreshing..." : "Reload"}
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-slate-700 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-3 px-4">Customer ID</th>
                <th className="py-3 px-4">Age</th>
                <th className="py-3 px-4">Annual Income ($k)</th>
                <th className="py-3 px-4">Spending Score (1-100)</th>
                <th className="py-3 px-4">Cluster Assigned</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800 text-slate-600 dark:text-slate-300">
              {isLoading ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-slate-400">
                    Loading dataset...
                  </td>
                </tr>
              ) : paginatedData.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-slate-400">
                    No customers found matching &quot;{searchTerm}&quot;
                  </td>
                </tr>
              ) : (
                paginatedData.map((c) => (
                  <tr key={c.customer_id} className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition">
                    <td className="py-3 px-4 font-mono font-medium text-slate-900 dark:text-white">
                      {c.customer_id}
                    </td>
                    <td className="py-3 px-4">{c.age} yrs</td>
                    <td className="py-3 px-4 font-medium text-slate-800 dark:text-slate-200">
                      ${c.annual_income.toFixed(1)}k
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span className="w-8 font-medium">{c.spending_score}</span>
                        <div className="w-20 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-blue-600 h-1.5 rounded-full"
                            style={{ width: `${c.spending_score}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      {c.cluster !== null && c.cluster !== undefined ? (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-blue-100 text-blue-800 dark:bg-blue-900/60 dark:text-blue-200">
                          Cluster {c.cluster}
                        </span>
                      ) : (
                        <span className="text-xs text-slate-400 italic">Unassigned</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination controls */}
        <div className="p-3 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs text-slate-500 bg-slate-50/30 dark:bg-slate-900/30">
          <span>
            Page {currentPage} of {totalPages}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded border border-slate-200 dark:border-slate-700 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded border border-slate-200 dark:border-slate-700 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer disabled:cursor-not-allowed"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};
