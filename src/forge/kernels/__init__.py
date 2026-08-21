"""Custom Triton kernels for Forge.

Provides fused GPU kernels that combine multiple operations into
single kernel launches, eliminating intermediate memory allocations
and kernel launch overhead.

All kernels have PyTorch-native fallbacks so Forge works on any
hardware — Triton just makes it faster on supported NVIDIA GPUs.

Requires: pip install 'forge-llm[kernels]' (triton >= 2.1.0)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Detect Triton availability at import time (not at CLI startup)
_TRITON_AVAILABLE: bool | None = None


def is_triton_available() -> bool:
    """Check if Triton is installed and functional."""
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            import triton  # noqa: F401

            _TRITON_AVAILABLE = True
        except ImportError:
            _TRITON_AVAILABLE = False
            logger.info(
                "Triton not available — using PyTorch fallbacks. "
                "Install with: pip install 'forge-llm[kernels]'"
            )
    return _TRITON_AVAILABLE


def get_kernel_info() -> dict:
    """Get information about available kernels and their backend."""
    backend = "triton" if is_triton_available() else "pytorch"
    return {
        "backend": backend,
        "kernels": {
            "lora_fused_forward": backend,
            "fused_cross_entropy": backend,
            "quantized_matmul": backend,
        },
    }
