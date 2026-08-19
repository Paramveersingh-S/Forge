"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { fetchExperiment, fetchMetricHistory } from "@/lib/api";
import { useMetricsWS } from "@/hooks/useMetrics";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function ExperimentDetail() {
  const params = useParams();
  const id = params.id as string;

  const [experiment, setExperiment] = useState<any>(null);
  const [metricHistory, setMetricHistory] = useState<any[]>([]);
  const { liveMetrics, status: wsStatus } = useMetricsWS(id);

  useEffect(() => {
    fetchExperiment(id).then(setExperiment).catch(console.error);
    fetchMetricHistory(id, "loss").then(data => setMetricHistory(data.data)).catch(console.error);
  }, [id]);

  // Merge static history with live websocket updates
  const chartData = [...metricHistory];
  const liveSteps = Object.keys(liveMetrics).map(Number).sort((a, b) => a - b);
  
  for (const step of liveSteps) {
    if (liveMetrics[step].loss !== undefined) {
      // Avoid duplicates if WS and API overlap
      if (!chartData.find(d => d.step === step)) {
        chartData.push({ step, value: liveMetrics[step].loss });
      }
    }
  }

  if (!experiment) return <div className="p-8 text-center text-muted-foreground">Loading experiment...</div>;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-3xl font-bold tracking-tight">{experiment.name}</h1>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
              experiment.status === 'running' ? 'bg-yellow-500/20 text-yellow-400' :
              experiment.status === 'completed' ? 'bg-green-500/20 text-green-400' :
              'bg-red-500/20 text-red-400'
            }`}>
              {experiment.status}
            </span>
            {wsStatus === 'connected' && (
              <span className="flex items-center gap-1 text-xs font-medium text-green-400 bg-green-500/10 px-2 py-1 rounded-full border border-green-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                LIVE
              </span>
            )}
          </div>
          <p className="text-muted-foreground font-mono text-sm">{experiment.id}</p>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <div className="col-span-2 glass rounded-xl p-6">
          <h3 className="text-lg font-bold mb-4">Training Loss</h3>
          <div className="h-[400px] w-full">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="step" stroke="#888" tick={{fill: '#888'}} />
                  <YAxis stroke="#888" tick={{fill: '#888'}} domain={['auto', 'auto']} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#60a5fa' }}
                  />
                  <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-muted-foreground">
                No loss data available yet.
              </div>
            )}
          </div>
        </div>

        <div className="space-y-6">
          <div className="glass rounded-xl p-6">
            <h3 className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-4">Configuration</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-zinc-400">Method</span>
                <span className="font-medium text-white uppercase">{experiment.config?.training?.method || "N/A"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Model</span>
                <span className="font-medium text-white truncate max-w-[150px]" title={experiment.config?.model?.name}>
                  {experiment.config?.model?.name?.split('/').pop() || "N/A"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Batch Size</span>
                <span className="font-medium text-white">{experiment.config?.training?.batch_size || "N/A"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">LR</span>
                <span className="font-medium text-white">{experiment.config?.training?.learning_rate || "N/A"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Quantization</span>
                <span className="font-medium text-white">{experiment.config?.training?.quantization || "none"}</span>
              </div>
            </div>
          </div>

          <div className="glass rounded-xl p-6">
            <h3 className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-4">Latest Metrics</h3>
            <div className="space-y-2">
              {Object.entries(experiment.latest_metrics || {}).map(([key, val]: [string, any]) => {
                // If it's live, override with the live value
                const latestStep = liveSteps.length > 0 ? liveSteps[liveSteps.length - 1] : null;
                const displayVal = latestStep && liveMetrics[latestStep][key] !== undefined 
                  ? liveMetrics[latestStep][key] 
                  : val;
                  
                return (
                  <div key={key} className="flex justify-between items-center bg-black/20 p-2 rounded">
                    <span className="text-zinc-400 text-xs">{key}</span>
                    <span className="font-mono text-sm text-blue-400">{Number(displayVal).toFixed(4)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
