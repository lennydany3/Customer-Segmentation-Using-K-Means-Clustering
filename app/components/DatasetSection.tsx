"use client";

import React, { useState, useMemo } from "react";
import { Customer } from "../lib/types";
import { Database, Search, ChevronLeft, ChevronRight } from "lucide-react";

interface DatasetSectionProps {
  customers: Customer[];
  totalCount: number;
  fileName: string | null;
}

export const DatasetSection: React.FC<DatasetSectionProps> = ({
  customers,
  totalCount,
  fileName,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  // Filter customers by name, district or product
  const filteredCustomers = useMemo(() => {
    const q = searchTerm.toLowerCase();
    return customers.filter((c) =>
      [c.name, c.district ?? "", c.product_type ?? ""].some((v) => v.toLowerCase().includes(q))
    );
  }, [customers, searchTerm]);

  // Pagination calculation
  const totalPages = Math.ceil(filteredCustomers.length / pageSize) || 1;
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredCustomers.slice(start, start + pageSize);
  }, [filteredCustomers, currentPage, pageSize]);

  // Overall statistics + the largest repeat count (scales the little bars in the table)
  const stats = useMemo(() => {
    if (customers.length === 0) return { avgAge: 0, avgRepeats: 0, maxRepeats: 1 };
    const avgAge = customers.reduce((acc, c) => acc + c.age, 0) / customers.length;
    const avgRepeats = customers.reduce((acc, c) => acc + c.repeat_purchases, 0) / customers.length;
    const maxRepeats = Math.max(1, ...customers.map((c) => c.repeat_purchases));
    return { avgAge: Math.round(avgAge), avgRepeats: Math.round(avgRepeats * 10) / 10, maxRepeats };
  }, [customers]);

  return (
    <section id="dataset" className="py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-200 dark:border-slate-800 scroll-mt-16">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 font-semibold text-sm">
            <Database className="w-4 h-4" />
            <span>Step 2 · Your Dataset</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white mt-1">
            Uploaded Customer Data
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
            {totalCount} customers loaded{fileName ? <> from <span className="font-mono font-semibold">{fileName}</span></> : null}.
            The Cluster column shows the segment the model assigned to each one.
          </p>
        </div>

        {/* Quick summary stats */}
        <div className="grid grid-cols-3 gap-3 text-center sm:text-left">
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-2.5 px-4 shadow-xs">
            <span className="text-xs text-slate-500 block">Total Customers</span>
            <span className="text-lg font-bold text-slate-900 dark:text-white">{totalCount}</span>
          </div>
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-2.5 px-4 shadow-xs">
            <span className="text-xs text-slate-500 block">Avg Age</span>
            <span className="text-lg font-bold text-blue-600 dark:text-blue-400">{stats.avgAge} yrs</span>
          </div>
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-2.5 px-4 shadow-xs">
            <span className="text-xs text-slate-500 block">Avg Repeats</span>
            <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">{stats.avgRepeats}</span>
          </div>
        </div>
      </div>

      {/* Table Card */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-xs overflow-hidden">
        {/* Search toolbar */}
        <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-3 bg-slate-50/50 dark:bg-slate-900/50">
          <div className="relative w-full sm:w-64">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search name, district or product..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full pl-9 pr-3 py-1.5 text-sm rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <span className="self-end sm:self-auto text-xs text-slate-500 dark:text-slate-400">
            Showing {paginatedData.length} of {filteredCustomers.length} customers
          </span>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-slate-700 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-3 px-4">Name</th>
                <th className="py-3 px-4">Age</th>
                <th className="py-3 px-4">Gender</th>
                <th className="py-3 px-4">District</th>
                <th className="py-3 px-4">Product Type</th>
                <th className="py-3 px-4">Repeat Purchases</th>
                <th className="py-3 px-4">Cluster Assigned</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800 text-slate-600 dark:text-slate-300">
              {paginatedData.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400">
                    No customers found matching &quot;{searchTerm}&quot;
                  </td>
                </tr>
              ) : (
                paginatedData.map((c, idx) => (
                  <tr key={`${c.customer_id}-${idx}`} className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition">
                    <td className="py-3 px-4 font-medium text-slate-900 dark:text-white">
                      {c.name}
                    </td>
                    <td className="py-3 px-4">{c.age} yrs</td>
                    <td className="py-3 px-4">{c.gender ?? "-"}</td>
                    <td className="py-3 px-4">{c.district ?? "-"}</td>
                    <td className="py-3 px-4">{c.product_type ?? "-"}</td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span className="w-10 font-medium">{c.repeat_purchases}</span>
                        <div className="w-20 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-blue-600 h-1.5 rounded-full"
                            style={{ width: `${Math.min(100, Math.max(0, (c.repeat_purchases / stats.maxRepeats) * 100))}%` }}
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
              aria-label="Previous page"
              className="p-1.5 rounded border border-slate-200 dark:border-slate-700 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              aria-label="Next page"
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
