"use client";

import { useEffect, useState } from "react";
import { fetchExperiments } from "@/lib/api";
import { Boxes, GitBranch } from "lucide-react";

export default function RegistryPage() {
  const [experiments, setExperiments] = useState<any[]>([]);

  useEffect(() => {
    fetchExperiments().then(data => setExperiments(data.experiments.filter((e: any) => e.status === 'completed'))).catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Model Registry</h1>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="glass rounded-xl p-6 border border-blue-500/30 bg-blue-500/5">
          <div className="flex items-center gap-3 mb-4">
            <Boxes className="w-6 h-6 text-blue-400" />
            <h2 className="text-xl font-bold">Base Models</h2>
          </div>
          <div className="space-y-3">
            <div className="p-4 bg-black/40 rounded-lg border border-white/5">
              <div className="font-medium text-lg">meta-llama/Llama-3-8B</div>
              <div className="text-sm text-muted-foreground mt-1">Found in 2 experiments</div>
            </div>
          </div>
        </div>

        <div className="glass rounded-xl p-6 border border-purple-500/30 bg-purple-500/5">
          <div className="flex items-center gap-3 mb-4">
            <GitBranch className="w-6 h-6 text-purple-400" />
            <h2 className="text-xl font-bold">Trained Adapters (LoRA)</h2>
          </div>
          <div className="space-y-3">
            {experiments.map(exp => (
              <div key={exp.id} className="p-4 bg-black/40 rounded-lg border border-white/5 flex justify-between items-center">
                <div>
                  <div className="font-medium text-blue-400">{exp.name}</div>
                  <div className="text-xs text-muted-foreground font-mono mt-1">{exp.id} • {exp.config?.training?.method?.toUpperCase()}</div>
                </div>
                <button className="px-3 py-1 bg-white/10 hover:bg-white/20 rounded text-sm font-medium transition-colors">
                  Export
                </button>
              </div>
            ))}
            {experiments.length === 0 && (
              <div className="text-center text-muted-foreground py-8">No completed models found.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
