import pytest
import torch

from forge.kernels.lora_fused import _pytorch_lora_fused_forward, lora_fused_forward
from forge.kernels.quantized_mm import _pytorch_quantized_matmul, quantized_matmul
from forge.kernels.cross_entropy import _pytorch_chunked_cross_entropy, fused_cross_entropy

@pytest.mark.gpu
def test_lora_fused_forward():
    M, K, N, R = 128, 64, 64, 16
    x = torch.randn((M, K), dtype=torch.float16, device="cuda")
    W = torch.randn((K, N), dtype=torch.float16, device="cuda")
    A = torch.randn((K, R), dtype=torch.float16, device="cuda")
    B = torch.randn((R, N), dtype=torch.float16, device="cuda")
    scale = 2.0

    # Test pure PyTorch fallback
    out_pt = _pytorch_lora_fused_forward(x, W, A, B, scale)
    
    # Test Triton kernel
    out_triton = lora_fused_forward(x, W, A, B, scale)

    torch.testing.assert_close(out_triton, out_pt, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_quantized_matmul():
    M, K, N = 128, 64, 64
    x = torch.randn((M, K), dtype=torch.float16, device="cuda")
    
    # NF4 mock (2 values per byte for 4-bit)
    W_q = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8, device="cuda")
    absmax = torch.ones((K // 64, N), dtype=torch.float16, device="cuda")

    out_pt = _pytorch_quantized_matmul(x, W_q, absmax, block_size=64)
    out_triton = quantized_matmul(x, W_q, absmax, block_size=64)

    torch.testing.assert_close(out_triton, out_pt, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_fused_cross_entropy():
    batch, seq, vocab = 2, 64, 128
    logits = torch.randn((batch * seq, vocab), dtype=torch.float16, device="cuda")
    targets = torch.randint(0, vocab, (batch * seq,), dtype=torch.long, device="cuda")

    loss_pt = _pytorch_chunked_cross_entropy(logits, targets)
    loss_triton = fused_cross_entropy(logits, targets)

    torch.testing.assert_close(loss_triton, loss_pt, atol=1e-2, rtol=1e-2)
