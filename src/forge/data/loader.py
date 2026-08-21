"""Multi-format dataset loader with auto-detection.

Supports: ShareGPT, Alpaca, OpenAI-chat, JSONL, CSV, Parquet, HuggingFace Hub.
Auto-detects format from file content when not specified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# Canonical column names Forge uses internally
PROMPT_COL = "prompt"
COMPLETION_COL = "completion"
CHOSEN_COL = "chosen"
REJECTED_COL = "rejected"
MESSAGES_COL = "messages"


def detect_format(path: str | Path) -> str:
    """Auto-detect the dataset format from file content.

    Inspects the first record to determine whether the data is
    ShareGPT, Alpaca, OpenAI-chat, or raw completion format.

    Returns:
        One of: 'sharegpt', 'alpaca', 'openai', 'completion', 'preference', 'unknown'.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".parquet", ".pq"):
        return "parquet"
    if suffix == ".csv":
        return "csv"

    # Read first line of JSONL / JSON
    if suffix in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return "unknown"

            # Handle JSON array
            if first_line.startswith("["):
                try:
                    data = json.loads(f.read())
                    record = data[0] if data else {}
                except (json.JSONDecodeError, IndexError):
                    return "unknown"
            else:
                try:
                    record = json.loads(first_line)
                except json.JSONDecodeError:
                    return "unknown"

        return _classify_record(record)

    return "unknown"


def _classify_record(record: dict[str, Any]) -> str:
    """Classify a single record by its keys."""
    keys = set(record.keys())

    # ShareGPT: {"conversations": [{"from": "human", "value": "..."}]}
    if "conversations" in keys:
        return "sharegpt"

    # OpenAI-chat: {"messages": [{"role": "user", "content": "..."}]}
    if "messages" in keys:
        return "openai"

    # Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    if "instruction" in keys and "output" in keys:
        return "alpaca"

    # Preference (DPO/KTO): {"prompt": "...", "chosen": "...", "rejected": "..."}
    if "chosen" in keys or "rejected" in keys:
        return "preference"

    # Completion: {"prompt": "...", "completion": "..."} or {"text": "..."}
    if "text" in keys or "completion" in keys or "response" in keys:
        return "completion"

    return "unknown"


def load_dataset(
    path: str | Path,
    format: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
) -> Any:
    """Load a dataset with auto-format detection.

    Args:
        path: Local file path or HuggingFace Hub dataset name.
        format: Explicit format override. If None, auto-detects.
        split: Dataset split to load.
        max_samples: Maximum number of samples to load.

    Returns:
        A HuggingFace Dataset object.
    """
    from datasets import load_dataset as hf_load_dataset

    path = Path(path) if not str(path).startswith("hf://") else path

    # HuggingFace Hub dataset
    if isinstance(path, str) and ("/" in path and not Path(path).exists()):
        console.print(f"[dim]Loading from HuggingFace Hub: {path}[/dim]")
        ds = hf_load_dataset(str(path), split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Auto-detect format
    if format is None:
        format = detect_format(path)
        console.print(f"[dim]Auto-detected format: {format}[/dim]")

    # Load based on format
    suffix = path.suffix.lower()
    if suffix == ".parquet" or format == "parquet":
        ds = hf_load_dataset("parquet", data_files=str(path), split=split)
    elif suffix == ".csv" or format == "csv":
        ds = hf_load_dataset("csv", data_files=str(path), split=split)
    else:
        ds = hf_load_dataset("json", data_files=str(path), split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    console.print(f"[green]✓[/green] Loaded {len(ds)} samples ({format})")
    return ds


def get_stats(path: str | Path) -> dict[str, Any]:
    """Get quick statistics about a dataset without fully loading it.

    Returns dict with: num_samples, format, columns, sample_record.
    """
    path = Path(path)
    fmt = detect_format(path)

    stats: dict[str, Any] = {
        "path": str(path),
        "format": fmt,
        "size_bytes": path.stat().st_size,
    }

    # Count lines for JSONL
    if path.suffix.lower() in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line.startswith("["):
                # JSON array
                f.seek(0)
                data = json.load(f)
                stats["num_samples"] = len(data)
                stats["columns"] = list(data[0].keys()) if data else []
                stats["sample"] = data[0] if data else {}
            else:
                # JSONL — count lines
                count = 1  # Already read one
                for _ in f:
                    count += 1
                stats["num_samples"] = count
                try:
                    stats["columns"] = list(json.loads(first_line).keys())
                    stats["sample"] = json.loads(first_line)
                except json.JSONDecodeError:
                    stats["columns"] = []
                    stats["sample"] = {}

    return stats
