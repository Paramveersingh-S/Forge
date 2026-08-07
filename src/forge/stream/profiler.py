"""Streaming tier profiler — benchmark-based tier selection.

Measures actual throughput for each streaming source (RAM, NVMe, GPU)
to pick the optimal tier automatically. This is important because
the best tier depends on hardware — some NVMe drives are faster than
DDR4 RAM for sequential reads due to PCIe 4.0/5.0 bandwidth.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TierResult:
    """Result of profiling a storage tier."""

    tier: str  # "ram", "disk", "gpu"
    throughput_gbps: float  # GB/s sequential read
    latency_ms: float  # Average latency per block read
    available_bytes: int  # Available capacity
    recommended: bool = False

    @property
    def available_gb(self) -> float:
        return self.available_bytes / (1024**3)

    def __repr__(self) -> str:
        return (
            f"TierResult({self.tier}: {self.throughput_gbps:.2f} GB/s, "
            f"{self.latency_ms:.2f}ms, {self.available_gb:.1f} GB free"
            f"{', ★ RECOMMENDED' if self.recommended else ''})"
        )


def profile_tiers(
    block_size_mb: int = 64,
    num_iterations: int = 5,
) -> List[TierResult]:
    """Profile all available storage tiers.

    Writes and reads test blocks to measure real throughput.

    Args:
        block_size_mb: Size of each test block in MB.
        num_iterations: Number of read iterations to average.

    Returns:
        List of TierResult, sorted by throughput (best first).
    """
    results: List[TierResult] = []
    block_size = block_size_mb * 1024 * 1024

    # Profile RAM
    ram_result = _profile_ram(block_size, num_iterations)
    results.append(ram_result)

    # Profile disk (NVMe/SSD/HDD)
    disk_result = _profile_disk(block_size, num_iterations)
    results.append(disk_result)

    # Profile GPU memory (if available)
    gpu_result = _profile_gpu(block_size, num_iterations)
    if gpu_result:
        results.append(gpu_result)

    # Sort by throughput and mark the best as recommended
    results.sort(key=lambda r: r.throughput_gbps, reverse=True)
    if results:
        results[0].recommended = True

    return results


def _profile_ram(block_size: int, num_iterations: int) -> TierResult:
    """Profile system RAM throughput."""
    import psutil

    # Allocate a block in RAM
    data = bytearray(block_size)

    # Measure read throughput
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        # Simulate sequential read — access all bytes
        _ = bytes(data)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    avg_latency = sum(latencies) / len(latencies)
    throughput = (block_size / (1024**3)) / avg_latency if avg_latency > 0 else 0

    return TierResult(
        tier="ram",
        throughput_gbps=round(throughput, 2),
        latency_ms=round(avg_latency * 1000, 2),
        available_bytes=psutil.virtual_memory().available,
    )


def _profile_disk(block_size: int, num_iterations: int) -> TierResult:
    """Profile disk (NVMe/SSD) throughput."""
    import shutil

    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write test block
            data = os.urandom(block_size)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            tmpfile = f.name

        # Measure read throughput
        latencies = []
        for _ in range(num_iterations):
            # Clear OS cache hint (best effort)
            start = time.perf_counter()
            with open(tmpfile, "rb") as f:
                _ = f.read()
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        avg_latency = sum(latencies) / len(latencies)
        throughput = (block_size / (1024**3)) / avg_latency if avg_latency > 0 else 0

        # Get disk free space
        disk_usage = shutil.disk_usage(tmpfile)
        available = disk_usage.free

        return TierResult(
            tier="disk",
            throughput_gbps=round(throughput, 2),
            latency_ms=round(avg_latency * 1000, 2),
            available_bytes=available,
        )

    except Exception as e:
        logger.warning(f"Disk profiling failed: {e}")
        return TierResult(
            tier="disk",
            throughput_gbps=0,
            latency_ms=float("inf"),
            available_bytes=0,
        )
    finally:
        try:
            os.unlink(tmpfile)
        except Exception:
            pass


def _profile_gpu(block_size: int, num_iterations: int) -> Optional[TierResult]:
    """Profile GPU memory throughput (host→device transfer)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.device("cuda:0")

        # Allocate on host (pinned) and device
        host_tensor = torch.empty(block_size // 4, dtype=torch.float32, pin_memory=True)
        device_tensor = torch.empty_like(host_tensor, device=device)

        # Warmup
        device_tensor.copy_(host_tensor)
        torch.cuda.synchronize()

        # Measure H2D throughput
        latencies = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            device_tensor.copy_(host_tensor)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        avg_latency = sum(latencies) / len(latencies)
        throughput = (block_size / (1024**3)) / avg_latency if avg_latency > 0 else 0

        # GPU free memory
        free_mem = torch.cuda.mem_get_info(0)[0]

        return TierResult(
            tier="gpu",
            throughput_gbps=round(throughput, 2),
            latency_ms=round(avg_latency * 1000, 2),
            available_bytes=free_mem,
        )

    except Exception as e:
        logger.debug(f"GPU profiling skipped: {e}")
        return None


def print_tier_report(results: List[TierResult]) -> None:
    """Pretty-print tier profiling results."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold blue]📊 Storage Tier Profile[/bold blue]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Tier", style="cyan")
    table.add_column("Throughput")
    table.add_column("Latency")
    table.add_column("Available")
    table.add_column("Status")

    for r in results:
        status = "[green]★ RECOMMENDED[/green]" if r.recommended else "[dim]—[/dim]"
        table.add_row(
            r.tier.upper(),
            f"{r.throughput_gbps:.2f} GB/s",
            f"{r.latency_ms:.2f} ms",
            f"{r.available_gb:.1f} GB",
            status,
        )

    console.print(table)
