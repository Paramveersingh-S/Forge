"""Experiment comparison and leaderboard queries.

Provides utilities for comparing multiple experiments side-by-side
and ranking experiments by a target metric.

Usage:
    from forge.tracking.compare import compare_experiments, get_leaderboard

    table = compare_experiments(db, ["abc123", "def456"])
    best = get_leaderboard(db, metric="eval/loss", top_k=5, ascending=True)
"""

from __future__ import annotations

from typing import Any

from forge.tracking.db import ForgeDB


def compare_experiments(
    db: ForgeDB,
    experiment_ids: list[str],
) -> dict[str, Any]:
    """Compare multiple experiments side-by-side.

    Returns a structured comparison containing:
    - experiments: list of experiment metadata
    - metrics: dict mapping metric_key → {exp_id: latest_value}
    - configs: dict mapping exp_id → config dict (for diff)
    """
    experiments = []
    all_metric_keys: set[str] = set()
    exp_metrics: dict[str, dict[str, float]] = {}

    for eid in experiment_ids:
        exp = db.get_experiment(eid)
        if exp is None:
            continue
        experiments.append(exp)

        latest = db.get_latest_metrics(eid)
        exp_metrics[eid] = latest
        all_metric_keys.update(latest.keys())

    # Build comparison matrix: metric_key → {exp_id: value}
    metrics_matrix: dict[str, dict[str, float | None]] = {}
    for key in sorted(all_metric_keys):
        metrics_matrix[key] = {}
        for eid in experiment_ids:
            metrics_matrix[key][eid] = exp_metrics.get(eid, {}).get(key)

    # Config diff — show differences between experiments
    configs = {exp["id"]: exp.get("config") for exp in experiments}

    return {
        "experiments": experiments,
        "metrics": metrics_matrix,
        "configs": configs,
        "metric_keys": sorted(all_metric_keys),
    }


def get_metric_series_multi(
    db: ForgeDB,
    experiment_ids: list[str],
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    """Get a metric time series for multiple experiments (for overlaid charting).

    Returns: {exp_id: [{"step": ..., "value": ..., "wall_time": ...}, ...]}
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for eid in experiment_ids:
        result[eid] = db.get_metric_history(eid, key)
    return result


def get_leaderboard(
    db: ForgeDB,
    metric: str,
    top_k: int = 10,
    ascending: bool = True,
) -> list[dict[str, Any]]:
    """Rank experiments by a target metric.

    Args:
        metric: The metric key to rank by (e.g., "eval/loss").
        top_k: Number of top experiments to return.
        ascending: If True, lower is better (loss). If False, higher is better (accuracy).

    Returns:
        List of {experiment, metric_value} dicts, sorted by metric.
    """
    experiments = db.list_experiments(limit=500)
    scored: list[tuple[dict[str, Any], float]] = []

    for exp in experiments:
        latest = db.get_latest_metrics(exp["id"])
        if metric in latest:
            scored.append((exp, latest[metric]))

    scored.sort(key=lambda x: x[1], reverse=not ascending)
    return [
        {"experiment": exp, "metric_value": val, "metric_key": metric}
        for exp, val in scored[:top_k]
    ]


def get_config_diff(
    db: ForgeDB,
    experiment_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Find config differences between experiments.

    Returns a dict of keys that differ, mapping each key to
    {exp_id: value} for each experiment.
    """
    configs: dict[str, dict] = {}
    for eid in experiment_ids:
        exp = db.get_experiment(eid)
        if exp and exp.get("config"):
            configs[eid] = _flatten_dict(exp["config"])
        else:
            configs[eid] = {}

    # Find keys where values differ
    all_keys: set[str] = set()
    for cfg in configs.values():
        all_keys.update(cfg.keys())

    diffs: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        values = {eid: cfg.get(key) for eid, cfg in configs.items()}
        unique_values = set(str(v) for v in values.values())
        if len(unique_values) > 1:
            diffs[key] = values

    return diffs


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
