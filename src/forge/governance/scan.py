"""Spectral analysis for backdoor detection in LoRA adapters.

Backdoors in fine-tuned models often manifest as rank-1 dominant updates 
in the weight matrices. By calculating the Singular Value Decomposition (SVD)
of the adapter weights (e.g., A @ B), we can detect if the largest singular 
value dominates the others by an abnormal threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# Typical threshold for rank-1 dominance
# If (sigma_0 / sigma_1) > THRESHOLD, flag as potential backdoor
DOMINANCE_THRESHOLD = 10.0


def scan_adapter(adapter_path: str | Path, threshold: float = DOMINANCE_THRESHOLD) -> bool:
    """Scan a safetensors adapter for spectral signatures of backdoors.
    
    Args:
        adapter_path: Path to adapter directory or safetensors file.
        threshold: Singular value ratio threshold.
        
    Returns:
        True if the adapter passes (no backdoors detected), False otherwise.
    """
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError:
        console.print("[red]✗ PyTorch or safetensors not installed.[/red]")
        return False

    path = Path(adapter_path)
    if path.is_dir():
        safetensors_path = path / "adapter_model.safetensors"
    else:
        safetensors_path = path

    if not safetensors_path.exists():
        console.print(f"[red]✗ Adapter file not found: {safetensors_path}[/red]")
        return False

    console.print(f"[cyan]Scanning {safetensors_path.name} for spectral anomalies...[/cyan]")
    
    try:
        weights = load_file(str(safetensors_path))
    except Exception as e:
        console.print(f"[red]✗ Failed to load safetensors: {e}[/red]")
        return False

    # Group LoRA A and B matrices
    lora_a = {}
    lora_b = {}
    
    for key, tensor in weights.items():
        if "lora_A" in key:
            lora_a[key] = tensor
        elif "lora_B" in key:
            lora_b[key] = tensor

    if not lora_a or not lora_b:
        console.print("[yellow]⚠ No LoRA matrices found. Ensure this is a LoRA adapter.[/yellow]")
        return True

    suspicious_layers = []

    # Analyze each layer
    for key_a, A in lora_a.items():
        # Find matching B matrix
        key_b = key_a.replace("lora_A", "lora_B")
        if key_b not in lora_b:
            continue
            
        B = lora_b[key_b]
        
        # Ensure dimensions match for matmul
        if A.dim() != 2 or B.dim() != 2:
            continue
            
        # A is [rank, in_features], B is [out_features, rank] typically
        # Or A is [in, rank], B is [rank, out].
        # We compute the update matrix W = A @ B or B @ A depending on shape
        
        # We want the product. To be safe, try both depending on matching inner dimensions
        try:
            if A.shape[1] == B.shape[0]:
                update_matrix = A @ B
            elif B.shape[1] == A.shape[0]:
                update_matrix = B @ A
            else:
                continue
                
            # Compute SVD (run on CPU to avoid requiring GPU for governance checks)
            # Use float32 for precision
            update_matrix = update_matrix.to(torch.float32).cpu()
            
            # For large matrices, torch.linalg.svdvals is faster as it only computes singular values
            singular_values = torch.linalg.svdvals(update_matrix)
            
            if len(singular_values) < 2:
                continue
                
            sigma_0 = singular_values[0].item()
            sigma_1 = singular_values[1].item()
            
            # Avoid division by zero
            if sigma_1 < 1e-6:
                sigma_1 = 1e-6
                
            ratio = sigma_0 / sigma_1
            
            if ratio > threshold:
                suspicious_layers.append((key_a, ratio))
                
        except Exception as e:
            logger.debug(f"Failed to analyze {key_a}: {e}")
            continue

    if suspicious_layers:
        console.print(f"[bold red]✗ SECURITY WARNING: {len(suspicious_layers)} layers exhibit rank-1 dominance![/bold red]")
        console.print("[red]This is a strong indicator of a targeted backdoor or data poisoning.[/red]")
        for layer, ratio in suspicious_layers[:5]:
            console.print(f"  - {layer}: [red]Ratio {ratio:.1f}x[/red] (Threshold: {threshold}x)")
        if len(suspicious_layers) > 5:
            console.print(f"  ... and {len(suspicious_layers) - 5} more.")
        return False

    console.print(f"[green]✓ Adapter is clean.[/green] (Scanned {len(lora_a)} layers)")
    return True
