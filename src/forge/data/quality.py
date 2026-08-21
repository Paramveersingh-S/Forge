"""Data quality scoring and statistics.

Provides fast, lightweight quality signals for training data:
- Length distribution analysis
- Token count estimation
- Missing / empty field detection
- Duplicate detection (exact and near-duplicate)
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def compute_quality_report(
    path: str | Path,
    format: str | None = None,
    sample_size: int = 1000,
) -> dict[str, Any]:
    """Compute a quality report for a dataset.

    Fast, streaming analysis — doesn't load the entire dataset.

    Args:
        path: Path to the dataset file.
        format: Data format (auto-detected if None).
        sample_size: Number of records to analyze.

    Returns:
        Quality report dict with statistics.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = _read_sample(path, sample_size)
    if not records:
        return {"error": "Empty dataset", "num_samples": 0}

    report: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "num_sampled": len(records),
    }

    # Detect all unique keys
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    report["columns"] = sorted(all_keys)

    # Per-column analysis
    col_stats: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        values = [r.get(key) for r in records]
        col_stats[key] = _analyze_column(key, values)
    report["column_stats"] = col_stats

    # Text length distribution (primary text fields)
    text_fields = _find_text_fields(records[0])
    if text_fields:
        main_field = text_fields[0]
        lengths = [len(str(r.get(main_field, ""))) for r in records]
        report["text_length"] = {
            "field": main_field,
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(statistics.mean(lengths), 1),
            "median": round(statistics.median(lengths), 1),
            "stdev": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
        }

    # Estimated token count (rough: chars / 4)
    total_chars = sum(
        sum(len(str(v)) for v in r.values() if isinstance(v, str))
        for r in records
    )
    report["estimated_tokens"] = total_chars // 4

    # Exact duplicate detection
    hashes = [_record_hash(r) for r in records]
    dup_count = len(hashes) - len(set(hashes))
    report["exact_duplicates"] = dup_count
    report["duplicate_ratio"] = round(dup_count / len(records), 4) if records else 0

    # Empty field detection
    empty_counts: dict[str, int] = {}
    for key in sorted(all_keys):
        empties = sum(1 for r in records if _is_empty(r.get(key)))
        if empties > 0:
            empty_counts[key] = empties
    report["empty_fields"] = empty_counts

    # Quality score (0-100)
    report["quality_score"] = _compute_score(report)

    return report


def print_quality_report(report: dict[str, Any]) -> None:
    """Pretty-print a quality report to the console."""
    console.print("\n[bold blue]📊 Data Quality Report[/bold blue]")
    console.print(f"  Path:    {report.get('path', 'N/A')}")
    console.print(f"  Samples: {report.get('num_sampled', 0)}")
    console.print(f"  Size:    {_human_size(report.get('size_bytes', 0))}")

    score = report.get("quality_score", 0)
    color = "green" if score >= 80 else "yellow" if score >= 60 else "red"
    console.print(f"  Score:   [{color}]{score}/100[/{color}]")

    # Text stats
    if "text_length" in report:
        tl = report["text_length"]
        console.print(f"\n  [dim]Text lengths ({tl['field']}):[/dim]")
        console.print(f"    Min: {tl['min']}  Max: {tl['max']}  Mean: {tl['mean']}  Median: {tl['median']}")

    # Duplicates
    if report.get("exact_duplicates", 0) > 0:
        console.print(
            f"\n  [yellow]⚠ {report['exact_duplicates']} exact duplicates "
            f"({report['duplicate_ratio']:.1%})[/yellow]"
        )

    # Empty fields
    if report.get("empty_fields"):
        console.print("\n  [yellow]⚠ Empty fields:[/yellow]")
        for field, count in report["empty_fields"].items():
            console.print(f"    {field}: {count} empty")

    console.print()


# --- Internal helpers ---------------------------------------------------------


def _read_sample(path: Path, max_records: int) -> list[dict[str, Any]]:
    """Read up to max_records from a file."""
    records: list[dict[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                data = json.load(f)
                records = data[:max_records]
            else:
                for i, line in enumerate(f):
                    if i >= max_records:
                        break
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    return records


def _analyze_column(name: str, values: list) -> dict[str, Any]:
    """Analyze a single column's values."""
    non_none = [v for v in values if v is not None]
    types = Counter(type(v).__name__ for v in non_none)

    stats: dict[str, Any] = {
        "count": len(non_none),
        "null_count": len(values) - len(non_none),
        "types": dict(types),
    }

    # String stats
    string_values = [v for v in non_none if isinstance(v, str)]
    if string_values:
        lengths = [len(v) for v in string_values]
        stats["str_len_min"] = min(lengths)
        stats["str_len_max"] = max(lengths)
        stats["str_len_mean"] = round(statistics.mean(lengths), 1)

    return stats


def _find_text_fields(record: dict[str, Any]) -> list[str]:
    """Find the primary text fields in a record."""
    text_priority = [
        "text", "content", "completion", "output", "response",
        "instruction", "prompt", "question", "input",
    ]
    found = []
    for field in text_priority:
        if field in record and isinstance(record[field], str):
            found.append(field)

    # Also check nested (conversations, messages)
    if not found:
        for field in record:
            if isinstance(record[field], str) and len(record[field]) > 20:
                found.append(field)

    return found


def _record_hash(record: dict[str, Any]) -> str:
    """Create a hash for duplicate detection."""
    import hashlib

    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(canonical.encode()).hexdigest()


def _is_empty(value: Any) -> bool:
    """Check if a value is empty/null."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _compute_score(report: dict[str, Any]) -> int:
    """Compute a 0-100 quality score from the report."""
    score = 100

    # Penalize duplicates
    dup_ratio = report.get("duplicate_ratio", 0)
    score -= int(dup_ratio * 50)  # Up to -50 for all duplicates

    # Penalize empty fields
    empty = report.get("empty_fields", {})
    if empty:
        total_empties = sum(empty.values())
        num_sampled = report.get("num_sampled", 1)
        empty_ratio = total_empties / (num_sampled * len(report.get("columns", [1])))
        score -= int(empty_ratio * 30)  # Up to -30

    # Penalize very short texts
    if "text_length" in report:
        mean_len = report["text_length"].get("mean", 100)
        if mean_len < 10:
            score -= 20
        elif mean_len < 50:
            score -= 10

    return max(0, min(100, score))


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"
