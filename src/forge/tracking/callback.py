"""ForgeTrainerCallback — HF Trainer callback for experiment tracking.

Automatically logs training metrics to the Forge experiment database
and broadcasts them via WebSocket for the live dashboard.

Usage:
    from forge.tracking.callback import ForgeTrainerCallback

    callback = ForgeTrainerCallback(experiment_id="abc123")
    trainer = Trainer(..., callbacks=[callback])
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ForgeTrainerCallback:
    """Hugging Face Trainer callback that logs metrics to Forge tracking.

    This is designed to work with the transformers.TrainerCallback
    interface but doesn't inherit from it directly to avoid importing
    transformers at module level (keeping CLI startup fast).
    """

    def __init__(
        self,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
        log_system_metrics: bool = True,
        system_metrics_interval: int = 50,
    ) -> None:
        self._experiment_id = experiment_id
        self._experiment_name = experiment_name
        self._config = config
        self._tags = tags
        self._log_system = log_system_metrics
        self._system_interval = system_metrics_interval

        self._logger: Any = None
        self._broadcaster: Any = None

    @property
    def experiment_id(self) -> str | None:
        """The experiment ID being tracked."""
        return self._experiment_id

    def _ensure_initialized(self) -> None:
        """Lazy-initialize tracking components."""
        if self._logger is not None:
            return

        from forge.tracking import create_experiment, get_db
        from forge.tracking.metrics import MetricsLogger

        # Create experiment if we don't have one
        if self._experiment_id is None:
            name = self._experiment_name or "forge-training"
            self._experiment_id = create_experiment(
                name=name,
                config=self._config,
                tags=self._tags,
            )
            logger.info("Created experiment: %s", self._experiment_id)

        db = get_db()
        self._logger = MetricsLogger(db, self._experiment_id)

        # Try to get the WebSocket broadcaster (may not be running)
        try:
            from forge.server.ws import MetricsBroadcaster

            self._broadcaster = MetricsBroadcaster.get_instance()
        except Exception:
            self._broadcaster = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called at the start of training."""
        self._ensure_initialized()
        logger.info("Forge tracking started for experiment %s", self._experiment_id)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
        """Called when the trainer logs metrics."""
        if logs is None:
            return

        self._ensure_initialized()
        assert self._logger is not None

        step = state.global_step if state else 0

        # Filter out non-numeric values
        numeric_logs = {
            k: float(v) for k, v in logs.items()
            if isinstance(v, (int, float)) and k != "epoch"
        }

        if numeric_logs:
            self._logger.log_dict(numeric_logs, step=step)

            # Broadcast to WebSocket clients
            if self._broadcaster and self._experiment_id:
                self._broadcaster.publish_sync(
                    self._experiment_id,
                    {
                        "type": "metrics",
                        "experiment_id": self._experiment_id,
                        "step": step,
                        "metrics": numeric_logs,
                    },
                )

        # Log system metrics periodically
        if self._log_system and step > 0 and step % self._system_interval == 0:
            self._logger.log_system_metrics(step)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called at the end of training."""
        if self._logger:
            self._logger.flush()

        if self._experiment_id:
            from forge.tracking import finish_experiment

            finish_experiment(self._experiment_id, status="completed")

            # Broadcast completion
            if self._broadcaster:
                self._broadcaster.publish_sync(
                    self._experiment_id,
                    {
                        "type": "status",
                        "experiment_id": self._experiment_id,
                        "status": "completed",
                    },
                )

        logger.info("Forge tracking completed for experiment %s", self._experiment_id)

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict | None = None, **kwargs: Any) -> None:
        """Called after evaluation."""
        if metrics is None:
            return

        self._ensure_initialized()
        assert self._logger is not None

        step = state.global_step if state else 0
        eval_metrics = {
            f"eval/{k}" if not k.startswith("eval/") else k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

        if eval_metrics:
            self._logger.log_dict(eval_metrics, step=step)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called when a checkpoint is saved."""
        if self._experiment_id and args:
            from forge.tracking import get_db

            db = get_db()
            checkpoint_dir = getattr(args, "output_dir", "unknown")
            step = state.global_step if state else 0
            db.log_artifact(
                experiment_id=self._experiment_id,
                name=f"checkpoint-{step}",
                path=checkpoint_dir,
                artifact_type="checkpoint",
                metadata={"step": step},
            )
