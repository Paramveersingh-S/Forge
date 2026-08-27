# Forge Benchmarks (Planned)

This document will track actual, verified performance benchmarks for Forge's layer streaming and Triton kernels once run on hardware.

## Target Benchmarks

### Layer Streaming VRAM Footprint
**Goal:** Train Llama-3.1-8B on a 4 GB VRAM GPU.
**Status:** 🚧 Awaiting verification. PyTorch streaming implementation is merged; hardware validation pending.

### Triton Kernel VRAM Savings
**Goals:**
- `lora_fused_forward`: Reduce intermediate allocations by ~30%.
- `fused_cross_entropy`: Save `batch × seq × vocab` tensor allocation.
- `quantized_matmul`: Avoid dequantized copy entirely.

**Status:** 🚧 Kernels implemented; awaiting hardware measurements to confirm exact savings.

*Note: Once these are verified on GPU instances (e.g. Google Colab / CI), exact numbers will be published here.*
