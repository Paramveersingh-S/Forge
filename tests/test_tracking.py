"""Tests for the experiment tracking system."""

from pathlib import Path

import pytest

from forge.tracking.compare import (
    compare_experiments,
    get_config_diff,
    get_leaderboard,
    get_metric_series_multi,
)
from forge.tracking.db import ForgeDB
from forge.tracking.metrics import MetricsLogger


@pytest.fixture
def db(tmp_path: Path) -> ForgeDB:
    """Create a fresh test database."""
    return ForgeDB(str(tmp_path / "test.db"))


@pytest.fixture
def populated_db(db: ForgeDB) -> ForgeDB:
    """Create a DB with sample experiments and metrics."""
    # Experiment 1 — SFT run
    eid1 = db.create_experiment(
        name="sft-llama3",
        config={"model": {"name": "meta-llama/Llama-3-8B"}, "training": {"method": "sft", "lr": 2e-4}},
        tags=["sft", "llama3"],
    )
    for step in range(0, 100, 10):
        db.log_metric(eid1, "loss", 3.0 - step * 0.02, step)
        db.log_metric(eid1, "learning_rate", 2e-4 * (1 - step / 100), step)
    db.finish_experiment(eid1, status="completed")

    # Experiment 2 — DPO run
    eid2 = db.create_experiment(
        name="dpo-llama3",
        config={"model": {"name": "meta-llama/Llama-3-8B"}, "training": {"method": "dpo", "lr": 5e-5}},
        tags=["dpo", "llama3"],
    )
    for step in range(0, 100, 10):
        db.log_metric(eid2, "loss", 2.5 - step * 0.015, step)
        db.log_metric(eid2, "learning_rate", 5e-5 * (1 - step / 100), step)
    db.finish_experiment(eid2, status="completed")

    return db


class TestForgeDB:
    """Test core database operations."""

    def test_create_experiment(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="test-run")
        assert len(eid) == 12
        exp = db.get_experiment(eid)
        assert exp is not None
        assert exp["name"] == "test-run"
        assert exp["status"] == "running"

    def test_create_with_config_and_tags(self, db: ForgeDB) -> None:
        config = {"model": {"name": "test/model"}, "training": {"method": "sft"}}
        eid = db.create_experiment(name="configured-run", config=config, tags=["test", "sft"])
        exp = db.get_experiment(eid)
        assert exp["config"] == config
        assert exp["tags"] == ["test", "sft"]

    def test_finish_experiment(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="finish-test")
        db.finish_experiment(eid, status="completed")
        exp = db.get_experiment(eid)
        assert exp["status"] == "completed"
        assert exp["finished_at"] is not None

    def test_finish_experiment_failed(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="fail-test")
        db.finish_experiment(eid, status="failed")
        exp = db.get_experiment(eid)
        assert exp["status"] == "failed"

    def test_list_experiments(self, db: ForgeDB) -> None:
        db.create_experiment(name="exp-1")
        db.create_experiment(name="exp-2")
        db.create_experiment(name="exp-3")
        exps = db.list_experiments()
        assert len(exps) == 3

    def test_list_experiments_filter_status(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="completed-exp")
        db.finish_experiment(eid, status="completed")
        db.create_experiment(name="running-exp")

        completed = db.list_experiments(status="completed")
        assert len(completed) == 1
        assert completed[0]["name"] == "completed-exp"

    def test_list_experiments_filter_tag(self, db: ForgeDB) -> None:
        db.create_experiment(name="tagged", tags=["production"])
        db.create_experiment(name="untagged")

        tagged = db.list_experiments(tag="production")
        assert len(tagged) == 1
        assert tagged[0]["name"] == "tagged"

    def test_delete_experiment(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="to-delete")
        assert db.delete_experiment(eid) is True
        assert db.get_experiment(eid) is None

    def test_delete_nonexistent(self, db: ForgeDB) -> None:
        assert db.delete_experiment("nonexistent") is False

    def test_count_experiments(self, db: ForgeDB) -> None:
        db.create_experiment(name="exp-1")
        eid = db.create_experiment(name="exp-2")
        db.finish_experiment(eid)

        assert db.count_experiments() == 2
        assert db.count_experiments(status="running") == 1
        assert db.count_experiments(status="completed") == 1

    def test_get_nonexistent_experiment(self, db: ForgeDB) -> None:
        assert db.get_experiment("nonexistent") is None


class TestMetrics:
    """Test metric logging and retrieval."""

    def test_log_and_retrieve_metric(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="metrics-test")
        db.log_metric(eid, "loss", 2.5, step=0)
        db.log_metric(eid, "loss", 1.5, step=10)

        history = db.get_metric_history(eid, "loss")
        assert len(history) == 2
        assert history[0]["value"] == 2.5
        assert history[1]["value"] == 1.5

    def test_log_metrics_batch(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="batch-test")
        db.log_metrics_batch(eid, [
            ("loss", 3.0, 0, None),
            ("loss", 2.5, 10, None),
            ("lr", 1e-4, 0, None),
        ])

        loss_history = db.get_metric_history(eid, "loss")
        assert len(loss_history) == 2

    def test_get_metric_keys(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="keys-test")
        db.log_metric(eid, "loss", 2.5, step=0)
        db.log_metric(eid, "learning_rate", 1e-4, step=0)
        db.log_metric(eid, "grad_norm", 0.5, step=0)

        keys = db.get_metric_keys(eid)
        assert sorted(keys) == ["grad_norm", "learning_rate", "loss"]

    def test_get_latest_metrics(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="latest-test")
        db.log_metric(eid, "loss", 3.0, step=0)
        db.log_metric(eid, "loss", 2.0, step=10)
        db.log_metric(eid, "loss", 1.0, step=20)
        db.log_metric(eid, "lr", 1e-4, step=0)

        latest = db.get_latest_metrics(eid)
        assert latest["loss"] == 1.0
        assert latest["lr"] == 1e-4


class TestMetricsLogger:
    """Test the buffered MetricsLogger."""

    def test_log_and_flush(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="logger-test")
        logger = MetricsLogger(db, eid, flush_every=100)  # Don't auto-flush

        logger.log("loss", 3.0, step=0)
        logger.log("loss", 2.5, step=10)

        # Not flushed yet
        assert len(db.get_metric_history(eid, "loss")) == 0

        logger.flush()
        assert len(db.get_metric_history(eid, "loss")) == 2

    def test_auto_flush_on_count(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="auto-flush-test")
        logger = MetricsLogger(db, eid, flush_every=3)

        logger.log("loss", 3.0, step=0)
        logger.log("loss", 2.5, step=1)
        logger.log("loss", 2.0, step=2)  # Should trigger flush

        assert len(db.get_metric_history(eid, "loss")) == 3

    def test_log_dict(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="dict-test")
        logger = MetricsLogger(db, eid, flush_every=100)

        logger.log_dict({"loss": 2.5, "lr": 1e-4, "grad_norm": 0.5}, step=0)
        logger.flush()

        assert len(db.get_metric_keys(eid)) == 3

    def test_context_manager(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="ctx-test")
        with MetricsLogger(db, eid, flush_every=100) as logger:
            logger.log("loss", 3.0, step=0)
            # __exit__ should flush

        assert len(db.get_metric_history(eid, "loss")) == 1


class TestArtifacts:
    """Test artifact logging."""

    def test_log_and_get_artifacts(self, db: ForgeDB) -> None:
        eid = db.create_experiment(name="artifact-test")
        db.log_artifact(eid, "checkpoint-100", "/output/checkpoint-100", "checkpoint")
        db.log_artifact(eid, "final-adapter", "/output/adapter", "adapter", {"format": "lora"})

        artifacts = db.get_artifacts(eid)
        assert len(artifacts) == 2
        assert artifacts[0]["name"] == "checkpoint-100"
        assert artifacts[1]["metadata"] == {"format": "lora"}


class TestComparison:
    """Test experiment comparison utilities."""

    def test_compare_experiments(self, populated_db: ForgeDB) -> None:
        exps = populated_db.list_experiments()
        ids = [e["id"] for e in exps]

        result = compare_experiments(populated_db, ids)
        assert len(result["experiments"]) == 2
        assert "loss" in result["metric_keys"]
        assert "learning_rate" in result["metric_keys"]

    def test_get_metric_series_multi(self, populated_db: ForgeDB) -> None:
        exps = populated_db.list_experiments()
        ids = [e["id"] for e in exps]

        series = get_metric_series_multi(populated_db, ids, "loss")
        assert len(series) == 2
        for eid in ids:
            assert len(series[eid]) == 10  # 10 steps

    def test_get_leaderboard(self, populated_db: ForgeDB) -> None:
        results = get_leaderboard(populated_db, metric="loss", top_k=2, ascending=True)
        assert len(results) == 2
        # Lower loss is better (ascending), so the experiment with lower final loss comes first
        assert results[0]["metric_value"] <= results[1]["metric_value"]

    def test_get_config_diff(self, populated_db: ForgeDB) -> None:
        exps = populated_db.list_experiments()
        ids = [e["id"] for e in exps]

        diffs = get_config_diff(populated_db, ids)
        # Training method and lr differ between the two experiments
        assert "training.method" in diffs
        assert "training.lr" in diffs
