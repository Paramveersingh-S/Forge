"""System status and GPU monitoring endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from forge import __version__

router = APIRouter(tags=["system"])


@router.get("/system/status")
def system_status(request: Request) -> dict[str, Any]:
    """Forge system status — version, GPU info, active runs."""
    db = request.app.state.db

    # Count active experiments
    active_count = db.count_experiments(status="running")
    total_count = db.count_experiments()

    # GPU info
    gpu_info = _get_gpu_info()

    return {
        "forge_version": __version__,
        "active_runs": active_count,
        "total_experiments": total_count,
        "gpu": gpu_info,
        "db_path": db.db_path,
    }


@router.get("/system/gpu")
def gpu_status() -> dict[str, Any]:
    """Real-time GPU utilization."""
    return _get_gpu_info()


def _get_gpu_info() -> dict[str, Any]:
    """Collect GPU information (gracefully handles missing dependencies)."""
    info: dict[str, Any] = {
        "available": False,
        "devices": [],
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return info

        info["available"] = True
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["device_count"] = torch.cuda.device_count()

        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = props.total_mem

            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(total / (1024 ** 3), 2),
                "allocated_gb": round(allocated / (1024 ** 3), 2),
                "reserved_gb": round(reserved / (1024 ** 3), 2),
                "free_gb": round((total - reserved) / (1024 ** 3), 2),
                "utilization_percent": round(allocated / total * 100, 1) if total > 0 else 0,
            })

        info["devices"] = devices
    except ImportError:
        pass

    # System memory via psutil
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["system_ram"] = {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "used_gb": round(mem.used / (1024 ** 3), 2),
            "percent": mem.percent,
        }
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass

    return info
