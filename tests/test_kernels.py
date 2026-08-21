"""Tests for custom Triton kernels (with PyTorch fallback)."""

import pytest

from forge.kernels import get_kernel_info, is_triton_available


class TestKernelAvailability:
    """Test kernel detection and info."""

    def test_triton_detection(self) -> None:
        """is_triton_available() should return a bool."""
        result = is_triton_available()
        assert isinstance(result, bool)

    def test_kernel_info(self) -> None:
        """get_kernel_info() should list all kernels."""
        info = get_kernel_info()
        assert "backend" in info
        assert info["backend"] in ("triton", "pytorch")
        assert "lora_fused_forward" in info["kernels"]
        assert "fused_cross_entropy" in info["kernels"]
        assert "quantized_matmul" in info["kernels"]


class TestLoRAFusedKernel:
    """Test the fused LoRA forward kernel."""

    @pytest.fixture
    def tensors(self):
        """Create test tensors (works without GPU)."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        batch, seq, in_features, out_features, rank = 2, 8, 64, 128, 8
        x = torch.randn(batch, seq, in_features)
        W = torch.randn(in_features, out_features)
        A = torch.randn(in_features, rank)
        B = torch.randn(rank, out_features)
        return x, W, A, B

    def test_pytorch_fallback(self, tensors) -> None:
        """Verify PyTorch fallback produces correct shapes."""
        from forge.kernels.lora_fused import _pytorch_lora_fused_forward

        x, W, A, B = tensors
        result = _pytorch_lora_fused_forward(x, W, A, B, scale=0.5, dropout_p=0, training=False)
        assert result.shape == (2, 8, 128)

    def test_lora_fused_forward(self, tensors) -> None:
        """Test the main entry point (auto-selects backend)."""
        from forge.kernels.lora_fused import lora_fused_forward

        x, W, A, B = tensors
        result = lora_fused_forward(x, W, A, B, scale=1.0)
        assert result.shape == x.shape[:2] + (W.shape[1],)

    def test_scale_zero_equals_base(self, tensors) -> None:
        """With scale=0, LoRA contribution should be zero."""
        import torch

        from forge.kernels.lora_fused import _pytorch_lora_fused_forward

        x, W, A, B = tensors
        base = x @ W
        result = _pytorch_lora_fused_forward(x, W, A, B, scale=0, dropout_p=0, training=False)
        torch.testing.assert_close(result, base)


class TestCrossEntropyKernel:
    """Test the fused cross-entropy kernel."""

    def test_pytorch_chunked_matches_standard(self) -> None:
        """Chunked cross-entropy should match F.cross_entropy."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            pytest.skip("PyTorch not installed")

        from forge.kernels.cross_entropy import _pytorch_chunked_cross_entropy

        logits = torch.randn(32, 1000)
        labels = torch.randint(0, 1000, (32,))

        expected = F.cross_entropy(logits, labels)
        result = _pytorch_chunked_cross_entropy(logits, labels, ignore_index=-100, reduction="mean")

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_ignore_index(self) -> None:
        """Ignored labels should not contribute to loss."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from forge.kernels.cross_entropy import _pytorch_chunked_cross_entropy

        logits = torch.randn(4, 100)
        labels = torch.tensor([5, -100, 10, -100])

        loss = _pytorch_chunked_cross_entropy(logits, labels, ignore_index=-100, reduction="mean")
        assert loss.item() > 0  # Should be positive (averaged over 2 valid tokens)


class TestQuantizedMatmul:
    """Test the quantized matmul kernel."""

    def test_dequantize_nf4_shape(self) -> None:
        """NF4 dequantization should produce correct shape."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from forge.kernels.quantized_mm import dequantize_nf4

        K, N = 128, 64
        group_size = 64
        qweight = torch.randint(0, 255, (K // 2, N), dtype=torch.uint8)
        scales = torch.randn(K // group_size, N, dtype=torch.float16)
        zeros = torch.zeros(K // group_size, N, dtype=torch.float16)

        result = dequantize_nf4(qweight, scales, zeros, group_size)
        assert result.shape == (K, N)
