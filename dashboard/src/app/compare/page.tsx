"use client";

import { useEffect, useState } from "react";
import { fetchExperiments } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];

export default function ComparePage() {
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchExperiments().then(data => setExperiments(data.experiments)).catch(console.error);
  }, []);

  const handleCompare = async () => {
    if (selectedIds.length < 2) return;
    setLoading(true);
    try {
      const res = await fetch(`http://127.0.0.1:8377/api/experiments/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_ids: selectedIds })
      });
      const data = await res.json();
      
      // Also fetch time series for the primary metric (e.g. loss)
      const seriesRes = await fetch(`http://127.0.0.1:8377/api/experiments/compare/metrics/loss`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_ids: selectedIds })
      });
      const seriesData = await seriesRes.json();
      
      setComparisonData({ ...data, timeSeries: seriesData.series });
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const toggleSelection = (id: string) => {
    setSelectedIds(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id].slice(0, 5) // max 5
    );
  };

  // Prepare chart data: merge series by step
  const getChartData = () => {
    if (!comparisonData?.timeSeries) return [];
    
    const steps = new Set<number>();
    Object.values(comparisonData.timeSeries).forEach((series: any) => {
      series.forEach((d: any) => steps.add(d.step));
    });
    
    const merged = Array.from(steps).sort((a, b) => a - b).map(step => {
      const point: any = { step };
      selectedIds.forEach((id) => {
        const match = comparisonData.timeSeries[id]?.find((d: any) => d.step === step);
        if (match) point[id] = match.value;
      });
      return point;
    });
    
    return merged;
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Compare Experiments</h1>
      
      <div className="grid md:grid-cols-4 gap-6">
        <div className="glass rounded-xl p-6 h-fit">
          <h3 className="font-bold mb-4">Select Runs (Max 5)</h3>
          <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
            {experiments.map(exp => (
              <label key={exp.id} className="flex items-start gap-3 p-3 rounded-lg hover:bg-white/5 cursor-pointer border border-transparent has-[:checked]:border-blue-500/50 has-[:checked]:bg-blue-500/10 transition-colors">
                <input 
                  type="checkbox" 
                  className="mt-1 rounded border-white/20 bg-black/50"
                  checked={selectedIds.includes(exp.id)}
                  onChange={() => toggleSelection(exp.id)}
                  disabled={!selectedIds.includes(exp.id) && selectedIds.length >= 5}
                />
                <div>
                  <div className="font-medium text-sm">{exp.name}</div>
                  <div className="text-xs text-muted-foreground font-mono">{exp.id}</div>
                </div>
              </label>
            ))}
          </div>
          <button 
            onClick={handleCompare}
            disabled={selectedIds.length < 2 || loading}
            className="w-full mt-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
          >
            {loading ? "Comparing..." : "Compare Selected"}
          </button>
        </div>

        <div className="md:col-span-3 space-y-6">
          {!comparisonData ? (
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center h-[400px]">
              Select at least 2 experiments to compare their metrics and configs.
            </div>
          ) : (
            <>
              <div className="glass rounded-xl p-6">
                <h3 className="text-lg font-bold mb-4">Loss Comparison</h3>
                <div className="h-[400px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={getChartData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="step" stroke="#888" tick={{fill: '#888'}} />
                      <YAxis stroke="#888" tick={{fill: '#888'}} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                      />
                      <Legend />
                      {comparisonData.experiments.map((exp: any, i: number) => (
                        <Line 
                          key={exp.id}
                          type="monotone" 
                          dataKey={exp.id} 
                          name={exp.name}
                          stroke={COLORS[i % COLORS.length]} 
                          strokeWidth={2} 
                          dot={false}
                          connectNulls
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="glass rounded-xl p-6 overflow-x-auto">
                <h3 className="text-lg font-bold mb-4">Configuration Diff</h3>
                {Object.keys(comparisonData.config_diff).length === 0 ? (
                  <p className="text-muted-foreground">No configuration differences found.</p>
                ) : (
                  <table className="w-full text-left text-sm">
                    <thead className="border-b border-white/10 text-muted-foreground">
                      <tr>
                        <th className="p-3 font-medium">Parameter</th>
                        {comparisonData.experiments.map((exp: any) => (
                          <th key={exp.id} className="p-3 font-medium text-white">{exp.name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(comparisonData.config_diff).map(([key, values]: [string, any]) => (
                        <tr key={key} className="border-b border-white/5 hover:bg-white/5">
                          <td className="p-3 font-mono text-xs text-blue-300">{key}</td>
                          {comparisonData.experiments.map((exp: any) => (
                            <td key={exp.id} className="p-3 font-mono text-xs text-zinc-300">
                              {JSON.stringify(values[exp.id] ?? "—")}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
