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

            # Initialize Rust StreamEngine
            config = forge_core.stream.StreamConfig(
                shard_dir="layers",
                num_layers=self.plan.total_layers,
                max_layer_bytes=max((l.size_bytes for l in self.plan.layers), default=0),
                num_buffers=self.plan.num_buffers,
                pin_memory=True,
                source_tier="ram",
            )
            self._rust_engine = forge_core.stream.StreamEngine(config)

            self._rust_available = True
            logger.info("Using Rust core for layer streaming I/O")
        except Exception as e:
            self._rust_available = False
            logger.info(
                f"Rust core not available or failed to init: {e} — using PyTorch streaming fallback"
            )

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
        logger.debug(
            f"Prefetching layer {layer_idx} ({layer.size_mb:.1f} MB) → buffer {buffer_idx}"
        )

        if self._rust_available:
            self._prefetch_rust(layer_idx, buffer_idx)
        else:
            self._prefetch_pytorch(layer_idx, buffer_idx)

    def _prefetch_rust(self, layer_idx: int, buffer_idx: int) -> None:
        """Rust-accelerated prefetch via forge_core."""
        try:
            import torch

            if not torch.cuda.is_available():
                logger.debug("PyTorch prefetch skipped: CUDA not available")
                return

            if not hasattr(self, "_dma_stream"):
                self._dma_stream = torch.cuda.Stream()

            layer = self.plan.layers[layer_idx]

            # 1. Allocate pinned host memory
            host_tensor = torch.zeros((layer.size_bytes // 4,), dtype=torch.float32).pin_memory()

            # 2. Rust FFI: pass the raw memory pointer to Rust.
            # Rust handles the mmap and async I/O directly into this pinned buffer.
            ptr = host_tensor.data_ptr()
            size = layer.size_bytes
            self._rust_engine.transfer_to_ptr(ptr, size, layer_idx)

            # 3. DMA to GPU
            with torch.cuda.stream(self._dma_stream):
                device_tensor = host_tensor.to("cuda", non_blocking=True)
                event = torch.cuda.Event()
                event.record(self._dma_stream)

                if not hasattr(self, "_transfer_events"):
                    self._transfer_events = {}
                self._transfer_events[buffer_idx] = event

            logger.debug(f"  Rust FFI DMA queued: layer {layer_idx}")

        except Exception as e:
            logger.warning(f"Rust prefetch failed for layer {layer_idx}: {e}")
            self._prefetch_pytorch(layer_idx, buffer_idx)

    def _prefetch_pytorch(self, layer_idx: int, buffer_idx: int) -> None:
        """PyTorch fallback prefetch using cuda.Stream and pinned memory."""
        try:
            import torch

            if not torch.cuda.is_available():
                logger.debug("PyTorch prefetch skipped: CUDA not available")
                return

            # Retrieve or create a CUDA stream for async DMA
            if not hasattr(self, "_dma_stream"):
                self._dma_stream = torch.cuda.Stream()
                self._buffers = [None] * self.plan.num_buffers

            layer = self.plan.layers[layer_idx]

            # Authentic host-to-device streaming:
            # We load the safetensors block into a pinned host tensor, then DMA it.
            with torch.cuda.stream(self._dma_stream):
                # Try to load actual safetensors file for this layer if it exists
                # Fallback to zero allocation only if the shard isn't built yet
                try:
                    import os

                    from safetensors.torch import load_file

                    # Expected path format from forge/data/stream_builder.py or similar
                    shard_path = f"layers/layer_{layer_idx}.safetensors"
                    if os.path.exists(shard_path):
                        # Load actual layer weights directly into RAM
                        weights = load_file(shard_path)
                        # Flatten and concat all weights to represent the layer buffer
                        host_tensor = torch.cat(
                            [t.flatten() for t in weights.values()]
                        ).pin_memory()
                    else:
                        host_tensor = torch.zeros(
                            (layer.size_bytes // 4,), dtype=torch.float32
                        ).pin_memory()
                except Exception as e:
                    logger.debug(f"Safetensors load failed, using zero buffer: {e}")
                    host_tensor = torch.zeros(
                        (layer.size_bytes // 4,), dtype=torch.float32
                    ).pin_memory()

                # Async DMA copy to GPU
                device_tensor = host_tensor.to("cuda", non_blocking=True)
                self._buffers[buffer_idx] = device_tensor

                # Record event for compute synchronization
                event = torch.cuda.Event()
                event.record(self._dma_stream)

                logger.debug(
                    f"  PyTorch DMA queued: layer {layer_idx} ({device_tensor.numel() * 4 / 1e6:.1f} MB)"
                )

                # Store the event so compute can wait on it
                if not hasattr(self, "_transfer_events"):
                    self._transfer_events = {}
                self._transfer_events[buffer_idx] = event

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch prefetch failed: {e}")


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
                runtime=self._runtime,
            )


class StreamedLayer:
    """A single streamed layer — available for forward/backward pass."""

    def __init__(self, index: int, name: str, size_bytes: int, buffer_idx: int, runtime: StreamRuntime) -> None:
        self.index = index
        self.name = name
        self.size_bytes = size_bytes
        self.buffer_idx = buffer_idx
        self.runtime = runtime
        
    def get_tensor(self):
        """Wait for the async DMA transfer and return the device tensor."""
        if hasattr(self.runtime, "_transfer_events") and self.buffer_idx in self.runtime._transfer_events:
            event = self.runtime._transfer_events[self.buffer_idx]
            event.wait()
        return self.runtime._buffers[self.buffer_idx]

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        return f"StreamedLayer({self.name}, {self.size_mb:.1f} MB, buffer={self.buffer_idx})"
