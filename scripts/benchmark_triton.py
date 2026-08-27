import torch
import triton
import triton.language as tl
import time

from forge.kernels.lora_fused import _pytorch_lora_fused_forward, lora_fused_forward
from forge.kernels.quantized_mm import _pytorch_quantized_matmul, quantized_matmul
from forge.kernels.cross_entropy import _pytorch_chunked_cross_entropy, fused_cross_entropy

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 20)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch"],  # Possible values for `line_arg`
        line_names=["Triton", "PyTorch"],  # Label name for the lines
        styles=[("blue", "-"), ("green", "-")],  # Line styles
        ylabel="GB/s",  # Label name for the y-axis
        plot_name="lora-fused-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={"K": 4096, "N": 4096, "R": 32},  # Values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_lora(M, K, N, R, provider):
    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    W = torch.randn((K, N), device="cuda", dtype=torch.float16)
    A = torch.randn((K, R), device="cuda", dtype=torch.float16)
    B = torch.randn((R, N), device="cuda", dtype=torch.float16)
    scale = 2.0

    quantiles = [0.5, 0.2, 0.8]
    
    # Calculate bytes for memory bandwidth tracking
    # Read: x (M*K), W (K*N), A (K*R), B (R*N)
    # Write: y (M*N)
    gbps = lambda ms: (2 * x.numel() + 2 * W.numel() + 2 * A.numel() + 2 * B.numel() + 2 * (M * N)) / ms * 1e-6

    if provider == "torch":
        # Clear cache before PyTorch test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: _pytorch_lora_fused_forward(x, W, A, B, scale), quantiles=quantiles
        )
        print(f"PyTorch LoRA Peak Memory (M={M}): {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    if provider == "triton":
        # Clear cache before Triton test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: lora_fused_forward(x, W, A, B, scale), quantiles=quantiles
        )
        print(f"Triton LoRA Peak Memory (M={M}): {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    print("Running LoRA benchmark...")
    benchmark_lora.run(print_data=True, show_plots=False)
