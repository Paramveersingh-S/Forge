"""Layer streaming plan computation.

Analyzes a model's architecture to determine:
- Number of decoder layers
- Size per layer (bytes)
- Required VRAM buffer size
- Optimal prefetch schedule (double-buffered, triple-buffered)
- Whether to stream from RAM or NVMe
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """Metadata about a single decoder layer."""

    index: int
    name: str
    size_bytes: int
    tensor_names: list[str] = field(default_factory=list)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass
class StreamPlan:
    """Complete streaming plan for a model.

    Contains the layer schedule, buffer sizing, and source tier
    decisions needed to execute layer streaming.
    """

    model_name: str
    total_layers: int
    layers: list[LayerInfo]
    num_buffers: int = 2
    source_tier: str = "ram"  # "ram", "disk", "gpu"
    prefetch_depth: int = 1
    total_size_bytes: int = 0
    buffer_size_bytes: int = 0

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / (1024**3)

    @property
    def buffer_size_mb(self) -> float:
        return self.buffer_size_bytes / (1024 * 1024)

    @property
    def peak_vram_mb(self) -> float:
        """Peak VRAM = num_buffers × largest_layer + overhead."""
        if not self.layers:
            return 0
        max_layer = max(l.size_bytes for l in self.layers)
        return (self.num_buffers * max_layer) / (1024 * 1024)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging."""
        return {
            "model": self.model_name,
            "layers": self.total_layers,
            "total_size_gb": round(self.total_size_gb, 2),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "source_tier": self.source_tier,
            "num_buffers": self.num_buffers,
            "prefetch_depth": self.prefetch_depth,
        }


def create_plan(
    model_path: str | Path,
    num_buffers: int = 2,
    source_tier: str = "auto",
    max_vram_gb: float | None = None,
) -> StreamPlan:
    """Create a streaming plan for a model.

    Analyzes the safetensors files in the model directory to determine
    the layer structure and optimal streaming parameters.

    Args:
        model_path: Path to the model directory (with safetensors files).
        num_buffers: Number of VRAM buffers (2 = double-buffered).
        source_tier: Where to stream from ('auto', 'ram', 'disk').
        max_vram_gb: Maximum VRAM budget in GB (for buffer sizing).

    Returns:
        StreamPlan with the full streaming schedule.
    """
    model_path = Path(model_path)

    # Find safetensors files
    st_files = list(model_path.glob("*.safetensors"))
    if not st_files:
        st_files = list(model_path.glob("**/*.safetensors"))

    if not st_files:
        logger.warning(f"No safetensors files found in {model_path}")
        return _create_fallback_plan(model_path, num_buffers)

    # Parse model structure from safetensors metadata
    layers = _parse_layer_structure(st_files)
    total_size = sum(l.size_bytes for l in layers)

    # Determine source tier
    if source_tier == "auto":
        source_tier = _auto_select_tier(total_size, max_vram_gb)

    # Buffer sizing
    max_layer_size = max(l.size_bytes for l in layers) if layers else 0
    buffer_size = max_layer_size * num_buffers

    plan = StreamPlan(
        model_name=model_path.name,
        total_layers=len(layers),
        layers=layers,
        num_buffers=num_buffers,
        source_tier=source_tier,
        prefetch_depth=min(num_buffers - 1, len(layers) - 1),
        total_size_bytes=total_size,
        buffer_size_bytes=buffer_size,
    )

    logger.info(f"Stream plan: {plan.summary()}")
    return plan


def _parse_layer_structure(st_files: list[Path]) -> list[LayerInfo]:
    """Parse decoder layer structure from safetensors metadata.

    Groups tensors by layer index based on naming conventions
    like 'model.layers.0.self_attn.q_proj.weight'.
    """
    import json
    import struct

    layer_tensors: dict[int, list[tuple]] = {}  # layer_idx -> [(name, size)]

    for st_path in st_files:
        try:
            with open(st_path, "rb") as f:
                # Read safetensors header
                header_size_bytes = f.read(8)
                header_size = struct.unpack("<Q", header_size_bytes)[0]
                header_json = f.read(header_size)
                header = json.loads(header_json)

                for tensor_name, tensor_meta in header.items():
                    if tensor_name == "__metadata__":
                        continue

                    # Compute tensor size from shape and dtype
                    shape = tensor_meta.get("shape", [])
                    dtype = tensor_meta.get("dtype", "F16")
                    offsets = tensor_meta.get("data_offsets", [0, 0])
                    size = offsets[1] - offsets[0] if len(offsets) == 2 else 0

                    # Extract layer index from name
                    layer_idx = _extract_layer_index(tensor_name)
                    if layer_idx is not None:
                        if layer_idx not in layer_tensors:
                            layer_tensors[layer_idx] = []
                        layer_tensors[layer_idx].append((tensor_name, size))

        except Exception as e:
            logger.warning(f"Failed to parse {st_path}: {e}")

    # Build LayerInfo objects
    layers = []
    for idx in sorted(layer_tensors.keys()):
        tensors = layer_tensors[idx]
        total_size = sum(size for _, size in tensors)
        tensor_names = [name for name, _ in tensors]
        layers.append(
            LayerInfo(
                index=idx,
                name=f"layer.{idx}",
                size_bytes=total_size,
                tensor_names=tensor_names,
            )
        )

    return layers


def _extract_layer_index(tensor_name: str) -> int | None:
    """Extract the decoder layer index from a tensor name.

    Handles naming conventions:
    - 'model.layers.0.self_attn.q_proj.weight' → 0
    - 'transformer.h.12.attn.c_attn.weight' → 12
    - 'encoder.layer.5.attention.self.query.weight' → 5
    """
    import re

    patterns = [
        r"\.layers\.(\d+)\.",
        r"\.h\.(\d+)\.",
        r"\.layer\.(\d+)\.",
        r"\.blocks\.(\d+)\.",
    ]

    for pattern in patterns:
        match = re.search(pattern, tensor_name)
        if match:
            return int(match.group(1))

    return None


def _auto_select_tier(total_size: int, max_vram_gb: float | None) -> str:
    """Auto-select the streaming source tier based on available resources."""
    import psutil

    available_ram = psutil.virtual_memory().available

    if total_size < available_ram * 0.8:  # Fits in 80% of available RAM
        return "ram"
    else:
        return "disk"


def _create_fallback_plan(model_path: Path, num_buffers: int) -> StreamPlan:
    """Create a minimal plan when model files can't be parsed."""
    return StreamPlan(
        model_name=model_path.name,
        total_layers=0,
        layers=[],
        num_buffers=num_buffers,
        source_tier="ram",
    )
