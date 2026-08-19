"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, Database, Cpu, LayoutDashboard } from "lucide-react";
import { fetchSystemStatus, fetchExperiments } from "@/lib/api";

export default function Home() {
  const [status, setStatus] = useState<any>(null);
  const [experiments, setExperiments] = useState<any[]>([]);

  useEffect(() => {
    fetchSystemStatus().then(setStatus).catch(console.error);
    fetchExperiments().then(data => setExperiments(data.experiments.slice(0, 5))).catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Overview</h1>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <div className="glass rounded-xl p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-blue-500/20 text-blue-400 rounded-lg">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Active Runs</p>
              <h3 className="text-2xl font-bold">{status?.active_runs ?? "-"}</h3>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-purple-500/20 text-purple-400 rounded-lg">
              <LayoutDashboard className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Total Experiments</p>
              <h3 className="text-2xl font-bold">{status?.total_experiments ?? "-"}</h3>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-green-500/20 text-green-400 rounded-lg">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">GPUs Available</p>
              <h3 className="text-2xl font-bold">{status?.gpu?.device_count ?? 0}</h3>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-orange-500/20 text-orange-400 rounded-lg">
              <Database className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Forge DB</p>
              <h3 className="text-sm font-bold truncate mt-1 text-zinc-400">{status?.db_path ? status.db_path.split(/[\\/]/).pop() : "-"}</h3>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8">
        <h2 className="text-xl font-bold tracking-tight mb-4">Recent Experiments</h2>
        <div className="glass rounded-xl overflow-hidden">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-white/10 bg-white/5">
              <tr>
                <th className="p-4 font-medium text-muted-foreground">Name</th>
                <th className="p-4 font-medium text-muted-foreground">Status</th>
                <th className="p-4 font-medium text-muted-foreground">Tags</th>
                <th className="p-4 font-medium text-muted-foreground">Created At</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp, i) => (
                <tr key={exp.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                  <td className="p-4 font-medium text-blue-400">
                    <Link href={`/experiments/${exp.id}`}>{exp.name}</Link>
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
                    <div className="flex gap-2">
                      {exp.tags?.map((t: string) => (
                        <span key={t} className="px-2 py-0.5 bg-white/10 rounded text-xs">{t}</span>
                      ))}
                    </div>
                  </td>
                  <td className="p-4 text-muted-foreground">
                    {new Date(exp.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
              {experiments.length === 0 && (
                <tr>
                  <td colSpan={4} className="p-8 text-center text-muted-foreground">
                    No experiments found. Run <code className="bg-white/10 px-1 rounded">forge train</code> to get started.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
