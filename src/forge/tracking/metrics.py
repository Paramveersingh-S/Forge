"""Buffered metrics logger for training loops.

Designed to be used inside HF Trainer callbacks. Buffers metric
writes and flushes them in batches to avoid per-step I/O overhead.

Usage:
    logger = MetricsLogger(db, experiment_id)
    logger.log("loss", 2.31, step=10)
    logger.log_dict({"loss": 2.31, "lr": 1e-4}, step=10)
    logger.flush()  # writes all buffered metrics to SQLite
"""

from __future__ import annotations

import threading
import time

from forge.tracking.db import ForgeDB


class MetricsLogger:
    """Buffered, thread-safe metric writer.

    Collects metrics in memory and writes them in batches to the
    SQLite database. The flush interval can be tuned — default is
    every 10 metrics or 5 seconds, whichever comes first.
    """

    def __init__(
        self,
        db: ForgeDB,
        experiment_id: str,
        flush_every: int = 10,
        flush_interval_secs: float = 5.0,
    ) -> None:
        self._db = db
        self._experiment_id = experiment_id
        self._flush_every = flush_every
        self._flush_interval = flush_interval_secs

        self._buffer: list[tuple[str, float, int, float | None]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._start_time = time.time()

    @property
    def experiment_id(self) -> str:
        """The experiment ID this logger is writing to."""
        return self._experiment_id

    def log(
        self,
        key: str,
        value: float,
        step: int,
        wall_time: float | None = None,
    ) -> None:
        """Log a single scalar metric.

        Metrics are buffered and flushed in batches.
        """
        if wall_time is None:
            wall_time = time.time() - self._start_time

        with self._lock:
            self._buffer.append((key, value, step, wall_time))

            if self._should_flush():
                self._flush_locked()

    def log_dict(
        self,
        metrics: dict[str, float],
        step: int,
        wall_time: float | None = None,
    ) -> None:
        """Log multiple metrics at the same step."""
        if wall_time is None:
            wall_time = time.time() - self._start_time

        with self._lock:
            for key, value in metrics.items():
                self._buffer.append((key, value, step, wall_time))

            if self._should_flush():
                self._flush_locked()

    def log_system_metrics(self, step: int) -> None:
        """Auto-capture and log system metrics (GPU, CPU, RAM).

        Fails gracefully if dependencies (pynvml, psutil) are missing.
        """
        metrics: dict[str, float] = {}

        # CPU and RAM via psutil
        try:
            import psutil

            metrics["system/cpu_percent"] = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            metrics["system/ram_used_gb"] = mem.used / (1024 ** 3)
            metrics["system/ram_percent"] = mem.percent
        except ImportError:
            pass

        # GPU via pynvml (bundled with nvidia-ml-py3, or torch)
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    metrics[f"gpu/{i}/vram_allocated_gb"] = allocated
                    metrics[f"gpu/{i}/vram_reserved_gb"] = reserved
        except ImportError:
            pass

        if metrics:
            self.log_dict(metrics, step)

    def flush(self) -> None:
        """Force flush all buffered metrics to the database."""
        with self._lock:
            self._flush_locked()

    def _should_flush(self) -> bool:
        """Check if we should flush (buffer size or time elapsed)."""
        if len(self._buffer) >= self._flush_every:
            return True
        if time.monotonic() - self._last_flush >= self._flush_interval:
            return True
        return False

    def _flush_locked(self) -> None:
        """Write buffered metrics to DB (must hold self._lock)."""
        if not self._buffer:
            return

        self._db.log_metrics_batch(self._experiment_id, self._buffer)
        self._buffer.clear()
        self._last_flush = time.monotonic()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.flush()
