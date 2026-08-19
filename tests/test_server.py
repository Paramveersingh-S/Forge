"""Tests for the FastAPI server and API endpoints."""

import pytest

from forge.tracking.db import ForgeDB


@pytest.fixture
def db(tmp_path):
    """Create a fresh test database."""
    return ForgeDB(str(tmp_path / "test.db"))


@pytest.fixture
def populated_db(db):
    """Populate DB with test data."""
    eid1 = db.create_experiment(
        name="sft-test",
        config={"training": {"method": "sft"}},
        tags=["sft"],
    )
    db.log_metric(eid1, "loss", 2.5, step=0)
    db.log_metric(eid1, "loss", 1.5, step=10)
    db.finish_experiment(eid1)

    eid2 = db.create_experiment(
        name="dpo-test",
        config={"training": {"method": "dpo"}},
        tags=["dpo"],
    )
    db.log_metric(eid2, "loss", 2.0, step=0)
    return db


@pytest.fixture
def client(populated_db):
    """Create a FastAPI test client with a populated DB."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("FastAPI not installed")

    from forge.server.app import create_app

    app = create_app()
    app.state.db = populated_db

    with TestClient(app) as client:
        yield client


class TestSystemEndpoints:
    """Test system status endpoints."""

    def test_system_status(self, client) -> None:
        resp = client.get("/api/system/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "forge_version" in data
        assert data["total_experiments"] == 2
        assert data["active_runs"] == 1  # One running, one completed

    def test_gpu_status(self, client) -> None:
        resp = client.get("/api/system/gpu")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data


class TestExperimentEndpoints:
    """Test experiment CRUD endpoints."""

    def test_list_experiments(self, client) -> None:
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["experiments"]) == 2

    def test_list_experiments_filter_status(self, client) -> None:
        resp = client.get("/api/experiments?status=completed")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["experiments"]) == 1
        assert data["experiments"][0]["status"] == "completed"

    def test_get_experiment(self, client, populated_db) -> None:
        exps = populated_db.list_experiments()
        eid = exps[0]["id"]

        resp = client.get(f"/api/experiments/{eid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == eid
        assert "latest_metrics" in data
        assert "metric_keys" in data

    def test_get_experiment_not_found(self, client) -> None:
        resp = client.get("/api/experiments/nonexistent")
        assert resp.status_code == 404

    def test_delete_experiment(self, client, populated_db) -> None:
        exps = populated_db.list_experiments()
        eid = exps[0]["id"]

        resp = client.delete(f"/api/experiments/{eid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone
        resp = client.get(f"/api/experiments/{eid}")
        assert resp.status_code == 404

    def test_delete_experiment_not_found(self, client) -> None:
        resp = client.delete("/api/experiments/nonexistent")
        assert resp.status_code == 404


class TestMetricsEndpoints:
    """Test metrics API endpoints."""

    def test_get_all_metrics(self, client, populated_db) -> None:
        exps = populated_db.list_experiments(status="completed")
        eid = exps[0]["id"]

        resp = client.get(f"/api/experiments/{eid}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "loss" in data["keys"]
        assert "loss" in data["latest"]

    def test_get_metric_history(self, client, populated_db) -> None:
        exps = populated_db.list_experiments(status="completed")
        eid = exps[0]["id"]

        resp = client.get(f"/api/experiments/{eid}/metrics/loss")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2  # 2 loss data points


class TestComparisonEndpoints:
    """Test comparison API endpoints."""

    def test_compare_experiments(self, client, populated_db) -> None:
        exps = populated_db.list_experiments()
        ids = [e["id"] for e in exps]

        resp = client.post("/api/experiments/compare", json={"experiment_ids": ids})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["experiments"]) == 2
        assert "loss" in data["metric_keys"]

    def test_compare_too_few(self, client) -> None:
        resp = client.post("/api/experiments/compare", json={"experiment_ids": ["abc"]})
        assert resp.status_code == 400

    def test_leaderboard(self, client) -> None:
        resp = client.post(
            "/api/experiments/leaderboard",
            json={"metric": "loss", "top_k": 5, "ascending": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["metric"] == "loss"
