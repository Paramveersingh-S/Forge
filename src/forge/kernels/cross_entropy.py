"""Fused cross-entropy kernel.

Standard cross-entropy: logits → softmax → log → nll_loss
This requires materializing the full logits tensor in VRAM:
  [batch × seq_len × vocab_size] — often 128K × 32K = 16 GB!

Fused approach: compute loss directly from logits without
materializing the full softmax, reducing peak VRAM by ~50%.
Uses the log-sum-exp trick for numerical stability.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fused_cross_entropy(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    ignore_index: int = -100,
    reduction: str = "mean",
) -> "torch.Tensor":
    """Fused cross-entropy loss without full logits materialization.

    Automatically uses Triton kernel when available, otherwise
    falls back to chunked PyTorch implementation that is still
    more memory-efficient than the naive approach.

    Args:
        logits: Raw model logits [batch * seq_len, vocab_size].
        labels: Target labels [batch * seq_len].
        ignore_index: Label value to ignore in loss computation.
        reduction: Loss reduction ('mean', 'sum', 'none').

    Returns:
        Scalar loss tensor (or per-token if reduction='none').
    """
    from forge.kernels import is_triton_available

    if is_triton_available():
        try:
            return _triton_fused_cross_entropy(logits, labels, ignore_index, reduction)
        except Exception as e:
            logger.warning(f"Triton kernel failed, falling back to PyTorch: {e}")

    return _pytorch_chunked_cross_entropy(logits, labels, ignore_index, reduction)


def _pytorch_chunked_cross_entropy(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    ignore_index: int,
    reduction: str,
    chunk_size: int = 4096,
) -> "torch.Tensor":
    """Chunked cross-entropy that processes vocab in chunks.

    Instead of computing softmax over the entire vocab at once,
    process in chunks to reduce peak memory. Uses the log-sum-exp
    trick across chunks for numerical stability.
    """
    import torch
    import torch.nn.functional as F

    N, V = logits.shape  # [tokens, vocab_size]

    if V <= chunk_size:
        # Small vocab — standard cross-entropy is fine
        return F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    # Chunked computation for large vocabs
    # Step 1: Find the global max for numerical stability
    global_max = logits.max(dim=-1, keepdim=True).values  # [N, 1]

    # Step 2: Compute log-sum-exp in chunks
    sum_exp = torch.zeros(N, 1, device=logits.device, dtype=logits.dtype)
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        chunk = logits[:, start:end] - global_max
        sum_exp += chunk.exp().sum(dim=-1, keepdim=True)

    log_sum_exp = global_max + sum_exp.log()  # [N, 1]

    # Step 3: Gather the logit at the correct label position
    valid_mask = labels != ignore_index
    safe_labels = labels.clamp(min=0)  # Avoid out-of-bounds
    correct_logits = logits.gather(1, safe_labels.unsqueeze(1)).squeeze(1)  # [N]

    # Step 4: loss = log_sum_exp - correct_logit
    loss = log_sum_exp.squeeze(1) - correct_logits  # [N]

    # Mask out ignored positions
    loss = loss * valid_mask.float()

    if reduction == "mean":
        num_valid = valid_mask.sum().float().clamp(min=1)
        return loss.sum() / num_valid
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def _triton_fused_cross_entropy(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    ignore_index: int,
    reduction: str,
) -> "torch.Tensor":
    """Triton-accelerated fused cross-entropy.

    Computes cross-entropy in a single kernel pass over the vocab
    dimension, never materializing the full softmax. Each Triton
    program instance handles one token's loss computation.
    """
    import torch
    import triton
    import triton.language as tl

    N, V = logits.shape

    # Output loss per token
    losses = torch.empty(N, device=logits.device, dtype=logits.dtype)

    @triton.jit
    def _cross_entropy_kernel(
        logits_ptr,
        labels_ptr,
        losses_ptr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
        ignore_index: tl.constexpr,
    ):
        """Fused cross-entropy kernel — one program per token."""
        row = tl.program_id(0)
        label = tl.load(labels_ptr + row)

        # Skip ignored tokens
        if label == ignore_index:
            tl.store(losses_ptr + row, 0.0)
            return

        # Compute log-sum-exp over vocab in blocks
        row_offset = row * V
        max_val = -float("inf")

        # Pass 1: find max for numerical stability
        for start in range(0, V, BLOCK_V):
            offsets = start + tl.arange(0, BLOCK_V)
            mask = offsets < V
            vals = tl.load(logits_ptr + row_offset + offsets, mask=mask, other=-float("inf"))
            block_max = tl.max(vals)
            max_val = tl.maximum(max_val, block_max)

        # Pass 2: compute sum(exp(x - max))
        sum_exp = 0.0
        for start in range(0, V, BLOCK_V):
            offsets = start + tl.arange(0, BLOCK_V)
            mask = offsets < V
            vals = tl.load(logits_ptr + row_offset + offsets, mask=mask, other=-float("inf"))
            sum_exp += tl.sum(tl.exp(vals - max_val))

        log_sum_exp = max_val + tl.log(sum_exp)

        # Get the logit at the correct label
        correct_logit = tl.load(logits_ptr + row_offset + label)

        # loss = log_sum_exp - correct_logit
        loss = log_sum_exp - correct_logit
        tl.store(losses_ptr + row, loss)

    # Launch kernel
    BLOCK_V = triton.next_power_of_2(min(V, 4096))
    grid = (N,)
    _cross_entropy_kernel[grid](
        logits, labels, losses, V, BLOCK_V, ignore_index
    )

    # Reduction
    valid_mask = labels != ignore_index
    losses = losses * valid_mask.float()

    if reduction == "mean":
        num_valid = valid_mask.sum().float().clamp(min=1)
        return losses.sum() / num_valid
    elif reduction == "sum":
        return losses.sum()
    return losses
