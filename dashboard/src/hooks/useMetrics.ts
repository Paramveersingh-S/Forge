"use client";

import { useEffect, useState, useRef } from "react";

export function useMetricsWS(experimentId: string) {
  const [liveMetrics, setLiveMetrics] = useState<Record<string, any>>({});
  const [status, setStatus] = useState<string>("connecting");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!experimentId) return;

    const wsUrl = `ws://127.0.0.1:8377/ws/training/${experimentId}`;
    
    const connect = () => {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("connected");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "metrics") {
            setLiveMetrics(prev => ({
              ...prev,
              [data.step]: data.metrics
            }));
          } else if (data.type === "status") {
            if (data.status === "completed") {
              setStatus("completed");
            }
          }
        } catch (e) {
          console.error("Failed to parse WS message", e);
        }
      };

      ws.onclose = () => {
        if (status !== "completed") {
          setStatus("disconnected");
          // Reconnect after 3s
          setTimeout(connect, 3000);
        }
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [experimentId]);

  return { liveMetrics, status };
}
