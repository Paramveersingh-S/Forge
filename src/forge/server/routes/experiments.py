"""Experiment CRUD and metrics API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from forge.tracking.compare import (
    compare_experiments,
    get_config_diff,
    get_leaderboard,
    get_metric_series_multi,
)

router = APIRouter(tags=["experiments"])


# --- Request / Response models -----------------------------------------------


class CompareRequest(BaseModel):
    experiment_ids: list[str]


class LeaderboardRequest(BaseModel):
    metric: str
    top_k: int = 10
    ascending: bool = True


# --- Experiment endpoints ----------------------------------------------------


@router.get("/experiments")
def list_experiments(
    request: Request,
    status: str | None = Query(None, description="Filter by status"),
    tag: str | None = Query(None, description="Filter by tag"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List all experiments with optional filters."""
    db = request.app.state.db
    experiments = db.list_experiments(status=status, tag=tag, limit=limit, offset=offset)
    total = db.count_experiments(status=status)
    return {
        "experiments": experiments,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/experiments/{experiment_id}")
def get_experiment(
    request: Request,
    experiment_id: str,
) -> dict[str, Any]:
    """Get a single experiment by ID."""
    db = request.app.state.db
    exp = db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    # Enrich with metrics summary
    exp["latest_metrics"] = db.get_latest_metrics(experiment_id)
    exp["metric_keys"] = db.get_metric_keys(experiment_id)
    exp["artifacts"] = db.get_artifacts(experiment_id)
    return exp


@router.delete("/experiments/{experiment_id}")
def delete_experiment(
    request: Request,
    experiment_id: str,
) -> dict[str, str]:
    """Delete an experiment and all its data."""
    db = request.app.state.db
    deleted = db.delete_experiment(experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return {"status": "deleted", "id": experiment_id}


# --- Metrics endpoints -------------------------------------------------------


@router.get("/experiments/{experiment_id}/metrics")
def get_all_metrics(
    request: Request,
    experiment_id: str,
) -> dict[str, Any]:
    """Get all metric keys and their latest values."""
    db = request.app.state.db
    exp = db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    return {
        "experiment_id": experiment_id,
        "keys": db.get_metric_keys(experiment_id),
        "latest": db.get_latest_metrics(experiment_id),
    }


@router.get("/experiments/{experiment_id}/metrics/{key:path}")
def get_metric_history(
    request: Request,
    experiment_id: str,
    key: str,
) -> dict[str, Any]:
    """Get the full time series for a single metric."""
    db = request.app.state.db
    history = db.get_metric_history(experiment_id, key)
    return {
        "experiment_id": experiment_id,
        "key": key,
        "data": history,
    }


# --- Comparison endpoints ----------------------------------------------------


@router.post("/experiments/compare")
def compare(
    request: Request,
    body: CompareRequest,
) -> dict[str, Any]:
    """Compare multiple experiments side-by-side."""
    db = request.app.state.db
    if len(body.experiment_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 experiment IDs required")
    if len(body.experiment_ids) > 10:
        raise HTTPException(status_code=400, detail="At most 10 experiments can be compared")

    result = compare_experiments(db, body.experiment_ids)
    result["config_diff"] = get_config_diff(db, body.experiment_ids)
    return result


@router.post("/experiments/compare/metrics/{key:path}")
def compare_metric_series(
    request: Request,
    key: str,
    body: CompareRequest,
) -> dict[str, Any]:
    """Get overlaid metric time series for multiple experiments."""
    db = request.app.state.db
    series = get_metric_series_multi(db, body.experiment_ids, key)
    return {
        "key": key,
        "series": series,
    }


@router.post("/experiments/leaderboard")
def leaderboard(
    request: Request,
    body: LeaderboardRequest,
) -> dict[str, Any]:
    """Get the top experiments ranked by a metric."""
    db = request.app.state.db
    results = get_leaderboard(db, body.metric, body.top_k, body.ascending)
    return {
        "metric": body.metric,
        "ascending": body.ascending,
        "results": results,
    }
