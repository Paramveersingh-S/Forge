"""Environment doctor — checks GPU, dependencies, and compatibility."""

from __future__ import annotations

import platform
import sys

from rich.console import Console
from rich.table import Table


def run_doctor(console: Console) -> None:
    """Run environment diagnostics."""
    console.print("[bold blue]forge doctor[/bold blue] — environment check\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    table.add_row(
        "Python",
        "[green]✓[/green]" if py_ok else "[red]✗[/red]",
        f"{py_ver} ({'supported' if py_ok else 'unsupported — need 3.10–3.12'})",
    )

    # Platform
    table.add_row("Platform", "[green]✓[/green]", f"{platform.system()} {platform.machine()}")

    # PyTorch
    try:
        import torch

        torch_ver = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)  # type: ignore
            table.add_row("PyTorch", "[green]✓[/green]", torch_ver)
            table.add_row("CUDA", "[green]✓[/green]", torch.version.cuda or "N/A")
            table.add_row("GPU", "[green]✓[/green]", f"{gpu_name} ({gpu_mem:.1f} GB)")
        else:
            table.add_row("PyTorch", "[green]✓[/green]", torch_ver)
            table.add_row("CUDA", "[yellow]⚠[/yellow]", "Not available (CPU-only)")
            table.add_row("GPU", "[yellow]⚠[/yellow]", "None detected")
    except ImportError:
        table.add_row("PyTorch", "[red]✗[/red]", "Not installed — pip install 'forge-llm[train]'")
        table.add_row("CUDA", "[dim]—[/dim]", "Skipped")
        table.add_row("GPU", "[dim]—[/dim]", "Skipped")

    # Transformers
    try:
        import transformers

        table.add_row("Transformers", "[green]✓[/green]", transformers.__version__)
    except ImportError:
        table.add_row("Transformers", "[red]✗[/red]", "Not installed")

    # PEFT
    try:
        import peft

        table.add_row("PEFT", "[green]✓[/green]", peft.__version__)
    except ImportError:
        table.add_row("PEFT", "[red]✗[/red]", "Not installed")

    # Rust core
    try:
        from forge import forge_core

        table.add_row("forge-core (Rust)", "[green]✓[/green]", forge_core.version())
    except ImportError:
        table.add_row("forge-core (Rust)", "[red]✗[/red]", "Not installed")

    console.print(table)
