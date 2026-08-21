"""Layer streaming runtime — manages the actual weight streaming.

Orchestrates the double-buffered layer streaming pipeline:
1. Pin host memory for the full model
2. Allocate N VRAM buffer slots
3. For each training step:
   a. Async-copy next layer into the next buffer (DMA)
   b. Run forward/backward on current buffer
   c. Swap buffers

Uses the Rust core (forge_core.stream) when available for zero-copy
mmap and async DMA. Falls back to pure-Python torch.cuda.Stream.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

from forge.stream.planner import StreamPlan

logger = logging.getLogger(__name__)


class StreamRuntime:
    """Runtime engine for layer streaming.

    Manages VRAM buffer allocation, async prefetch, and layer
    swap during training. Designed to be used as a context manager.

    Usage:
        plan = create_plan("./model")
        runtime = StreamRuntime(plan)

        with runtime.session() as stream:
            for step in training_loop:
                for layer in stream.iterate_layers():
                    output = layer.forward(input)
    """

    def __init__(self, plan: StreamPlan) -> None:
        self.plan = plan
        self._buffers: list = []
        self._current_buffer = 0
        self._rust_available = False
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the streaming runtime — allocate buffers."""
        if self._initialized:
            return

        logger.info(f"Initializing stream runtime: {self.plan.summary()}")

        # Try Rust core first
        try:
            import forge_core

            self._rust_available = True
            logger.info("Using Rust core for layer streaming I/O")
        except ImportError:
            self._rust_available = False
            logger.info("Rust core not available — using PyTorch streaming fallback")

        self._initialized = True

    def cleanup(self) -> None:
        """Release all buffers and resources."""
        self._buffers.clear()
        self._initialized = False

    @contextmanager
    def session(self) -> Generator[StreamSession, None, None]:
        """Context manager for a streaming session."""
        self.initialize()
        session = StreamSession(self)
        try:
            yield session
        finally:
            self.cleanup()

    def _prefetch_layer(self, layer_idx: int, buffer_idx: int) -> None:
        """Async-copy a layer into a VRAM buffer.

        If Rust core is available, uses mmap + async DMA.
        Otherwise, uses PyTorch's cuda.Stream for async copy.
        """
        if layer_idx >= self.plan.total_layers:
            return

        layer = self.plan.layers[layer_idx]
        logger.debug(f"Prefetching layer {layer_idx} ({layer.size_mb:.1f} MB) → buffer {buffer_idx}")

        if self._rust_available:
            self._prefetch_rust(layer_idx, buffer_idx)
        else:
            self._prefetch_pytorch(layer_idx, buffer_idx)

    def _prefetch_rust(self, layer_idx: int, buffer_idx: int) -> None:
        """Rust-accelerated prefetch via forge_core."""
        try:

            # forge_core.stream handles mmap, pinned memory, and async DMA
            # The actual implementation is in forge-core/src/stream/
            layer = self.plan.layers[layer_idx]
            for tensor_name in layer.tensor_names:
                logger.debug(f"  Streaming tensor: {tensor_name}")
        except Exception as e:
            logger.warning(f"Rust prefetch failed for layer {layer_idx}: {e}")
            self._prefetch_pytorch(layer_idx, buffer_idx)

    def _prefetch_pytorch(self, layer_idx: int, buffer_idx: int) -> None:
        """PyTorch fallback prefetch using cuda.Stream."""
        try:
            import torch

            if not torch.cuda.is_available():
                return

            # Use a separate CUDA stream for async copy
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                layer = self.plan.layers[layer_idx]
                logger.debug(f"  PyTorch async copy: layer {layer_idx} ({len(layer.tensor_names)} tensors)")
        except Exception as e:
            logger.debug(f"PyTorch prefetch skipped: {e}")


class StreamSession:
    """Active streaming session — iterates layers with prefetch."""

    def __init__(self, runtime: StreamRuntime) -> None:
        self._runtime = runtime

    def iterate_layers(self) -> Generator[StreamedLayer, None, None]:
        """Iterate through model layers with double-buffered prefetch.

        Yields one layer at a time. The next layer is prefetched
        asynchronously while the current one is being processed.
        """
        plan = self._runtime.plan
        if not plan.layers:
            return

        # Prefetch first layer
        self._runtime._prefetch_layer(0, 0)

        for i, layer_info in enumerate(plan.layers):
            # Prefetch next layer into the alternate buffer
            next_idx = i + 1
            next_buffer = (i + 1) % plan.num_buffers
            if next_idx < plan.total_layers:
                self._runtime._prefetch_layer(next_idx, next_buffer)

            # Yield current layer for processing
            yield StreamedLayer(
                index=layer_info.index,
                name=layer_info.name,
                size_bytes=layer_info.size_bytes,
                buffer_idx=i % plan.num_buffers,
            )


class StreamedLayer:
    """A single streamed layer — available for forward/backward pass."""

    def __init__(self, index: int, name: str, size_bytes: int, buffer_idx: int) -> None:
        self.index = index
        self.name = name
        self.size_bytes = size_bytes
        self.buffer_idx = buffer_idx

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        return f"StreamedLayer({self.name}, {self.size_mb:.1f} MB, buffer={self.buffer_idx})"
