"""Main CLI application — all commands registered here.

Design principle: NO top-level torch/transformers imports. The light CLI
(forge --help, forge doctor, forge init) must start instantly without
pulling in the heavy training stack.
"""

import sys

import typer
from rich.console import Console

from forge import __version__

console = Console()

# --- Main app -----------------------------------------------------------------

app = typer.Typer(
    name="forge",
    help="Forge — Rust-accelerated LLM fine-tuning with built-in provenance.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold blue]forge[/bold blue] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output."),
) -> None:
    """Forge — shape raw models into production weapons."""
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)


# --- Commands -----------------------------------------------------------------


@app.command()
def init(
    preset: str = typer.Option(
        "llama3-8b-chat",
        "--preset",
        "-p",
        help="Model preset to use (e.g. llama3-8b-chat, qwen3-7b, mistral-7b).",
    ),
    output: str = typer.Option(
        "forge.yaml",
        "--output",
        "-o",
        help="Output config file path.",
    ),
    wizard: bool = typer.Option(False, "--wizard", "-w", help="Interactive config wizard."),
) -> None:
    """Create a new Forge config from a preset."""
    from forge.config.loader import create_config_from_preset

    config_path = create_config_from_preset(preset, output, wizard=wizard)
    console.print(f"[green]✓[/green] Created config: [bold]{config_path}[/bold]")
    console.print(f"  Next: [cyan]forge train --config {config_path}[/cyan]")


@app.command()
def train(
    config: str = typer.Option(
        "forge.yaml",
        "--config",
        "-c",
        help="Path to the Forge YAML config file.",
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from the last checkpoint."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without training."),
) -> None:
    """Train a model using the specified config."""
    from forge.config.loader import load_config
    from forge.trainer.registry import get_trainer

    console.print("[bold blue]forge train[/bold blue]")

    cfg = load_config(config)
    if dry_run:
        console.print("[green]✓[/green] Config validated successfully.")
        console.print(f"  Method: [cyan]{cfg.training.method}[/cyan]")
        console.print(f"  Model:  [cyan]{cfg.model.name}[/cyan]")
        return

    trainer = get_trainer(cfg.training.method)
    trainer.train(cfg, resume=resume)


@app.command()
def doctor() -> None:
    """Check your environment for GPU, dependencies, and compatibility."""
    from forge.utils.doctor import run_doctor

    run_doctor(console)


@app.command(name="eval")
def evaluate(
    adapter: str = typer.Option(..., "--adapter", "-a", help="Path to the adapter directory."),
    suite: str = typer.Option(
        "mmlu,tool_call",
        "--suite",
        "-s",
        help="Comma-separated evaluation suites.",
    ),
) -> None:
    """Evaluate a fine-tuned adapter against benchmark suites."""
    console.print(f"[bold blue]forge eval[/bold blue] — suites: {suite}")
    console.print(f"  Adapter: {adapter}")
    # TODO: Implement evaluation engine
    console.print("[yellow]⚠[/yellow] Evaluation engine not yet implemented.")


@app.command()
def ship(
    base: str = typer.Option(..., "--base", "-b", help="Path to the base model."),
    adapter: str = typer.Option(..., "--adapter", "-a", help="Path to the adapter."),
    task_eval: str = typer.Option(None, "--task-eval", help="Task-specific eval JSONL."),
) -> None:
    """Run the ship gate — SHIP or DON'T SHIP verdict."""
    console.print("[bold blue]forge ship[/bold blue]")
    console.print(f"  Base:    {base}")
    console.print(f"  Adapter: {adapter}")
    # TODO: Implement ship gate
    console.print("[yellow]⚠[/yellow] Ship gate not yet implemented.")


@app.command()
def export(
    adapter: str = typer.Option(..., "--adapter", "-a", help="Path to the adapter."),
    format: str = typer.Option("gguf", "--format", "-f", help="Export format (gguf, safetensors, onnx)."),
    quant: str = typer.Option("q4_k_m", "--quant", "-q", help="Quantization level for GGUF."),
) -> None:
    """Export adapter to deployment format."""
    console.print(f"[bold blue]forge export[/bold blue] — format: {format}, quant: {quant}")
    console.print(f"  Adapter: {adapter}")
    # TODO: Implement export
    console.print("[yellow]⚠[/yellow] Export engine not yet implemented.")


@app.command()
def experiment(
    action: str = typer.Argument("list", help="Action: list, compare, export."),
) -> None:
    """Manage experiments — list, compare, export."""
    console.print(f"[bold blue]forge experiment[/bold blue] — {action}")
    # TODO: Implement experiment tracking
    console.print("[yellow]⚠[/yellow] Experiment tracking not yet implemented.")


# --- Entry point --------------------------------------------------------------


def run() -> None:
    """Entry point called by the `forge` console script."""
    app()


if __name__ == "__main__":
    run()
