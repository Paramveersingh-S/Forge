"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchExperiments } from "@/lib/api";

export default function ExperimentsList() {
  const [experiments, setExperiments] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExperiments()
      .then(data => {
        setExperiments(data.experiments);
        setLoading(false);
      })
      .catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Experiments</h1>
        <div className="flex gap-2">
          {/* Future: filters and search */}
          <Link href="/compare" className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors">
            Compare Selected
          </Link>
        </div>
      </div>

      <div className="glass rounded-xl overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-white/10 bg-white/5">
            <tr>
              <th className="p-4 w-12 text-center"><input type="checkbox" className="rounded border-white/20 bg-black/50" /></th>
              <th className="p-4 font-medium text-muted-foreground">Name / ID</th>
              <th className="p-4 font-medium text-muted-foreground">Status</th>
              <th className="p-4 font-medium text-muted-foreground">Tags</th>
              <th className="p-4 font-medium text-muted-foreground">Created At</th>
              <th className="p-4 font-medium text-muted-foreground text-right">Method</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={6} className="p-8 text-center text-muted-foreground">Loading...</td></tr>
            ) : experiments.length === 0 ? (
              <tr><td colSpan={6} className="p-8 text-center text-muted-foreground">No experiments found.</td></tr>
            ) : (
              experiments.map((exp) => (
                <tr key={exp.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                  <td className="p-4 text-center"><input type="checkbox" className="rounded border-white/20 bg-black/50" /></td>
                  <td className="p-4">
                    <Link href={`/experiments/${exp.id}`} className="font-medium text-blue-400 hover:underline block">
                      {exp.name}
                    </Link>
                    <span className="text-xs text-muted-foreground font-mono">{exp.id}</span>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      exp.status === 'running' ? 'bg-yellow-500/20 text-yellow-400' :
                      exp.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {exp.status}
                    </span>
                  </td>
                  <td className="p-4">
                    <div className="flex gap-2 flex-wrap max-w-[200px]">
                      {exp.tags?.map((t: string) => (
                        <span key={t} className="px-2 py-0.5 bg-white/10 rounded text-xs">{t}</span>
                      ))}
                    </div>
                  </td>
                  <td className="p-4 text-muted-foreground">
                    {new Date(exp.created_at).toLocaleString()}
                  </td>
                  <td className="p-4 text-right text-zinc-300 font-medium uppercase text-xs">
                    {exp.config?.training?.method || "—"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
