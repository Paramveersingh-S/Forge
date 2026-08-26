"""Main CLI application — all commands registered here.

Design principle: NO top-level torch/transformers imports. The light CLI
(forge --help, forge doctor, forge init) must start instantly without
pulling in the heavy training stack.
"""

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
    base: str = typer.Option(..., "--base", "-b", help="Path to the base model."),
    adapter: str = typer.Option(..., "--adapter", "-a", help="Path to the adapter directory."),
    tasks: str = typer.Option(..., "--tasks", "-t", help="Path to the JSONL task file."),
) -> None:
    """Evaluate a fine-tuned adapter against benchmark suites."""
    from forge.eval.engine import evaluate_adapter

    console.print("[bold blue]forge eval[/bold blue]")
    console.print(f"  Base:    {base}")
    console.print(f"  Adapter: {adapter}")
    console.print(f"  Tasks:   {tasks}")

    try:
        results = evaluate_adapter(base, adapter, tasks)
        console.print(f"\nFinal Score: [bold green]{results['score']:.2%}[/bold green]")
    except Exception as e:
        console.print(f"[red]✗ Evaluation failed: {e}[/red]")
        raise SystemExit(1)


@app.command()
def ship(
    base: str = typer.Argument(..., help="Path to the base model."),
    adapter: str = typer.Argument(..., help="Path to the trained adapter."),
    task_eval: str = typer.Option(
        None, "--task-eval", help="Path to task JSONL to gate deployment."
    ),
) -> None:
    """Run governance checks (ML-BOM, attestation) and ship to registry."""
    import json
    from pathlib import Path

    from forge.eval.engine import DEFAULT_THRESHOLD, evaluate_adapter
    from forge.governance.attest import sign_bom
    from forge.governance.bom import generate_bom

    console.print("[bold blue]forge ship[/bold blue] — Governance & Attestation")
    console.print(f"  Base:    {base}")
    console.print(f"  Adapter: {adapter}")

    if task_eval:
        console.print(f"\n[bold cyan]Running Ship Gate Evaluation on {task_eval}...[/bold cyan]")
        try:
            results = evaluate_adapter(base, adapter, task_eval)
            if results["score"] < DEFAULT_THRESHOLD:
                console.print(
                    f"[bold red]✗ DON'T SHIP: Regression detected (Score {results['score']:.2%} < Threshold {DEFAULT_THRESHOLD:.2%})[/bold red]"
                )
                raise SystemExit(2)
            console.print(
                f"[bold green]✓ SHIP: Adapter passed gating (Score {results['score']:.2%})[/bold green]"
            )
        except Exception as e:
            if isinstance(e, SystemExit):
                raise
            console.print(f"[red]✗ Gating failed: {e}[/red]")
            raise SystemExit(3)

    adapter_path = Path(adapter)
    if not adapter_path.exists():
        console.print(f"[red]✗ Adapter path does not exist: {adapter}[/red]")
        raise SystemExit(1)

    console.print("\n[bold cyan]Generating ML-BOM...[/bold cyan]")
    bom = generate_bom(adapter, base)

    bom_path = adapter_path / "ml-bom.json"
    bom_path.write_text(bom.to_json(), encoding="utf-8")
    console.print(f"  [green]✓[/green] ML-BOM generated: {bom_path}")

    console.print("\n[bold cyan]Cryptographically signing artifact...[/bold cyan]")
    try:
        signature = sign_bom(bom)
        sig_path = adapter_path / "ml-bom.sig"
        sig_path.write_text(signature, encoding="utf-8")
        console.print(f"  [green]✓[/green] Signed artifact: {sig_path}")
        console.print(f"  [dim]Signature: {signature[:32]}...[/dim]")
    except Exception as e:
        console.print(f"[red]✗ Signing failed: {e}[/red]")
        raise SystemExit(1)

    console.print("\n[bold green]✓ Ready for deployment.[/bold green]")


@app.command()
def export(
    adapter: str = typer.Option(..., "--adapter", "-a", help="Path to the adapter."),
    format: str = typer.Option(
        "gguf", "--format", "-f", help="Export format (gguf, safetensors, onnx)."
    ),
    quant: str = typer.Option("q4_k_m", "--quant", "-q", help="Quantization level for GGUF."),
) -> None:
    """Export adapter to deployment format."""
    from forge.export.engine import export_model

    console.print(f"[bold blue]forge export[/bold blue] — format: {format}, quant: {quant}")
    console.print(f"  Adapter: {adapter}")

    console.print("\n[bold cyan]Exporting model...[/bold cyan]")
    try:
        out_path = export_model(adapter, format, quant)
        console.print(f"  [green]✓[/green] Export successful: {out_path}")
    except Exception as e:
        console.print(f"[red]✗ Export failed: {e}[/red]")
        raise SystemExit(1)


@app.command()
def workspace(
    action: str = typer.Argument(..., help="Action: init, invite, list"),
    name: str = typer.Argument(None, help="Team name or ID."),
    email: str = typer.Option(None, "--email", "-e", help="Email of the user to invite."),
    role: str = typer.Option(
        "contributor", "--role", "-r", help="RBAC role: owner, maintainer, contributor, viewer."
    ),
) -> None:
    """Manage team workspaces and RBAC."""
    from forge.workspace.manager import WorkspaceManager

    manager = WorkspaceManager()

    if action == "init":
        if not name:
            console.print("[red]✗ Team name required for init.[/red]")
            raise SystemExit(1)
        try:
            team_id = manager.init_team(name)
            console.print(
                f"[green]✓[/green] Initialized workspace for team: [bold]{name}[/bold] ({team_id})"
            )
        except ValueError as e:
            console.print(f"[red]✗ {e}[/red]")

    elif action == "invite":
        if not name or not email:
            console.print("[red]✗ Team name and email required for invite.[/red]")
            raise SystemExit(1)
        try:
            manager.invite_member(name, email, role)
            console.print(f"[green]✓[/green] Invited {email} to {name} as [cyan]{role}[/cyan]")
        except ValueError as e:
            console.print(f"[red]✗ {e}[/red]")

    elif action == "list":
        if not name:
            console.print("[red]✗ Team name required to list members.[/red]")
            raise SystemExit(1)
        members = manager.list_members(name)
        if not members:
            console.print(f"[yellow]⚠ No members found for {name}.[/yellow]")
        else:
            from rich.table import Table

            table = Table(title=f"Members: {name}")
            table.add_column("Email", style="cyan")
            table.add_column("Role", style="bold")
            for m in members:
                table.add_row(m["email"], m["role"])
            console.print(table)

    else:
        console.print(f"[red]✗ Unknown action: {action}[/red]")
        console.print("  Available: init, invite, list")


@app.command()
def adapters(
    action: str = typer.Argument(..., help="Action: scan"),
    path: str = typer.Argument(..., help="Path to the adapter directory or safetensors file."),
    threshold: float = typer.Option(
        10.0, "--threshold", "-t", help="Dominance threshold for backdoor detection."
    ),
) -> None:
    """Manage and inspect trained adapters (e.g., backdoor scanning)."""
    if action == "scan":
        from forge.governance.scan import scan_adapter

        is_clean = scan_adapter(path, threshold=threshold)
        if not is_clean:
            raise SystemExit(2)
    else:
        console.print(f"[red]✗ Unknown action: {action}[/red]")
        console.print("  Available: scan")


@app.command()
def data(
    path: str = typer.Argument(..., help="Path to the dataset file."),
    action: str = typer.Option(
        "inspect",
        "--action",
        "-a",
        help="Action: inspect, quality, convert, dedup.",
    ),
    format: str = typer.Option(None, "--format", "-f", help="Override format detection."),
    target_format: str = typer.Option(
        "openai",
        "--target-format",
        help="Target format for convert action.",
    ),
    output: str = typer.Option(None, "--output", "-o", help="Output file for convert/dedup."),
) -> None:
    """Inspect, validate, and transform training datasets."""
    console.print(f"[bold blue]forge data[/bold blue] — {action}")

    if action == "inspect":
        from forge.data.loader import detect_format, get_stats

        fmt = detect_format(path)
        console.print(f"  Format: [cyan]{fmt}[/cyan]")
        stats = get_stats(path)
        console.print(f"  Samples: {stats.get('num_samples', 'N/A')}")
        console.print(f"  Columns: {stats.get('columns', [])}")
        if stats.get("sample"):
            import json

            console.print(f"  Sample:  {json.dumps(stats['sample'], indent=2)[:500]}")

    elif action == "quality":
        from forge.data.quality import compute_quality_report, print_quality_report

        report = compute_quality_report(path)
        print_quality_report(report)

    elif action == "convert":
        import json
        import os

        from forge.data.formats import convert_record
        from forge.data.loader import detect_format

        src_fmt = format or detect_format(path)
        console.print(f"  Converting: {src_fmt} → {target_format}")
        out_file = output or f"{os.path.splitext(path)[0]}_{target_format}.jsonl"
        console.print(f"  Output:     {out_file}")

        try:
            with (
                open(path, encoding="utf-8") as f_in,
                open(out_file, "w", encoding="utf-8") as f_out,
            ):
                for line in f_in:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    converted = convert_record(record, src_fmt, target_format)
                    f_out.write(json.dumps(converted) + "\n")
            console.print("[green]✓[/green] Conversion complete.")
        except Exception as e:
            console.print(f"[red]✗ Conversion failed: {e}[/red]")

    elif action == "dedup":
        import json
        import os

        from forge.data.dedup import deduplicate

        console.print(f"  Dataset: {path}")
        out_file = output or f"{os.path.splitext(path)[0]}_dedup.jsonl"

        records = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

            console.print(f"  Loaded {len(records)} records. Running MinHash LSH deduplication...")
            # We try 'text' first, or fallback heuristics in dedup.py
            deduped, removed = deduplicate(records, text_field="text")

            with open(out_file, "w", encoding="utf-8") as f:
                for r in deduped:
                    f.write(json.dumps(r) + "\n")

            console.print(f"  Output: {out_file}")
        except Exception as e:
            console.print(f"[red]✗ Deduplication failed: {e}[/red]")

    else:
        console.print(f"[red]✗ Unknown action: {action}[/red]")
        console.print("  Available: inspect, quality, convert, dedup")


@app.command()
def profile() -> None:
    """Benchmark storage tiers for layer streaming (RAM vs NVMe vs GPU)."""
    from forge.stream.profiler import print_tier_report, profile_tiers

    console.print("[bold blue]forge profile[/bold blue] — benchmarking storage tiers...\n")
    results = profile_tiers()
    print_tier_report(results)


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Server host."),
    port: int = typer.Option(8377, "--port", "-p", help="Server port."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Don't open browser automatically."
    ),
) -> None:
    """Launch the Forge web dashboard (experiment tracking + live training monitor)."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]✗ Missing dashboard dependency: uvicorn[/red]\n"
            "  Install with: pip install 'forge-llm[dashboard]'"
        )
        raise SystemExit(1)

    from forge.server.app import create_app

    console.print("[bold blue]forge dashboard[/bold blue]")
    console.print(f"  Server:  http://{host}:{port}")
    console.print(f"  API:     http://{host}:{port}/api/system/status")
    console.print("  Press [cyan]Ctrl+C[/cyan] to stop.\n")

    if not no_browser:
        import threading
        import webbrowser

        def _open_browser() -> None:
            import time

            time.sleep(1.5)  # Wait for server to start
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open_browser, daemon=True).start()

    forge_app = create_app()
    uvicorn.run(forge_app, host=host, port=port, log_level="info")


@app.command()
def experiment(
    action: str = typer.Argument("list", help="Action: list, compare, delete, show."),
    ids: str = typer.Option(None, "--ids", "-i", help="Comma-separated experiment IDs."),
    metric: str = typer.Option(None, "--metric", "-m", help="Metric key for leaderboard."),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of top experiments."),
    status: str = typer.Option(None, "--status", "-s", help="Filter by status."),
) -> None:
    """Manage experiments — list, compare, delete, show."""
    from rich.table import Table

    from forge.tracking import get_db

    db = get_db()

    if action == "list":
        experiments = db.list_experiments(status=status, limit=top_k)
        if not experiments:
            console.print("[dim]No experiments found.[/dim]")
            return

        table = Table(title="Experiments", show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="bold")
        table.add_column("Status")
        table.add_column("Created")
        table.add_column("Tags")

        for exp in experiments:
            status_style = {
                "running": "[yellow]running[/yellow]",
                "completed": "[green]completed[/green]",
                "failed": "[red]failed[/red]",
            }.get(exp["status"], exp["status"])

            created = exp["created_at"][:19] if exp["created_at"] else "—"
            tags = ", ".join(exp.get("tags") or []) or "—"
            table.add_row(exp["id"], exp["name"], status_style, created, tags)

        console.print(table)

    elif action == "show":
        if not ids:
            console.print("[red]✗ --ids required for 'show' action[/red]")
            return

        exp = db.get_experiment(ids.split(",")[0])  # type: ignore
        if not exp:
            console.print(f"[red]✗ Experiment '{ids}' not found[/red]")
            return

        console.print(f"\n[bold]{exp['name']}[/bold] ({exp['id']})")
        console.print(f"  Status:  {exp['status']}")
        console.print(f"  Created: {exp['created_at']}")

        latest = db.get_latest_metrics(exp["id"])
        if latest:
            console.print("\n  [bold]Latest Metrics:[/bold]")
            for key, val in sorted(latest.items()):
                console.print(f"    {key}: {val:.6f}")

    elif action == "compare":
        if not ids:
            console.print("[red]✗ --ids required for 'compare' action[/red]")
            console.print("  Usage: forge experiment compare --ids abc123,def456")
            return

        from forge.tracking.compare import compare_experiments as compare_fn

        exp_ids = [i.strip() for i in ids.split(",")]
        result = compare_fn(db, exp_ids)

        if not result["experiments"]:
            console.print("[red]✗ No matching experiments found[/red]")
            return

        table = Table(title="Experiment Comparison", show_lines=True)
        table.add_column("Metric", style="bold")
        for exp in result["experiments"]:
            table.add_column(f"{exp['name']}\n({exp['id']})", justify="right")

        for key in result["metric_keys"]:
            values = result["metrics"][key]
            row = [key]
            for exp in result["experiments"]:
                val = values.get(exp["id"])
                row.append(f"{val:.6f}" if val is not None else "—")
            table.add_row(*row)

        console.print(table)

    elif action == "delete":
        if not ids:
            console.print("[red]✗ --ids required for 'delete' action[/red]")
            return

        for eid in ids.split(","):
            eid = eid.strip()
            deleted = db.delete_experiment(eid)
            if deleted:
                console.print(f"[green]✓[/green] Deleted experiment {eid}")
            else:
                console.print(f"[red]✗[/red] Experiment {eid} not found")

    elif action == "leaderboard":
        if not metric:
            console.print("[red]✗ --metric required for 'leaderboard' action[/red]")
            return

        from forge.tracking.compare import get_leaderboard

        results = get_leaderboard(db, metric=metric, top_k=top_k)
        if not results:
            console.print(f"[dim]No experiments with metric '{metric}' found.[/dim]")
            return

        table = Table(title=f"Leaderboard — {metric}", show_lines=True)
        table.add_column("#", style="bold", no_wrap=True)
        table.add_column("Experiment", style="cyan")
        table.add_column("Value", justify="right")

        for i, entry in enumerate(results, 1):
            exp = entry["experiment"]
            table.add_row(str(i), f"{exp['name']} ({exp['id']})", f"{entry['metric_value']:.6f}")

        console.print(table)

    else:
        console.print(f"[red]✗ Unknown action: {action}[/red]")
        console.print("  Available: list, show, compare, delete, leaderboard")


@app.command()
def deploy(
    adapter: str = typer.Argument(..., help="Path to the exported model (GGUF or Safetensors)."),
    target: str = typer.Option(
        "ollama", "--target", "-t", help="Deployment target (ollama, vllm)."
    ),
    name: str = typer.Option("forge-model", "--name", "-n", help="Name for the deployed model."),
    output_dir: str = typer.Option(
        "./deployments", "--out", "-o", help="Output directory for manifests (vllm only)."
    ),
) -> None:
    """Deploy the model to a local inference engine or cluster."""
    console.print(f"[bold blue]forge deploy[/bold blue] — target: {target}")

    try:
        if target.lower() == "ollama":
            from forge.deploy.ollama import deploy_to_ollama

            console.print("\n[bold cyan]Deploying to Ollama...[/bold cyan]")
            deploy_to_ollama(name, adapter)
            console.print(f"  [green]✓[/green] Successfully deployed '{name}' to Ollama!")
            console.print(f"  Run with: [bold]ollama run {name}[/bold]")

        elif target.lower() == "vllm":
            from forge.deploy.vllm import generate_k8s_manifests

            console.print("\n[bold cyan]Generating vLLM Kubernetes manifests...[/bold cyan]")
            manifest_path = generate_k8s_manifests(name, adapter, output_dir)
            console.print(f"  [green]✓[/green] Generated manifest: {manifest_path}")
            console.print(f"  Apply with: [bold]kubectl apply -f {manifest_path}[/bold]")

        else:
            console.print(f"[red]✗ Unknown target: {target}[/red]")
            raise SystemExit(1)

    except Exception as e:
        console.print(f"[red]✗ Deployment failed: {e}[/red]")
        raise SystemExit(1)


# --- Entry point --------------------------------------------------------------


def run() -> None:
    """Entry point called by the `forge` console script."""
    app()


if __name__ == "__main__":
    run()
