"""Trainer registry — decorator-based dispatch for training methods.

Unlike Soup's monolithic import of all 31 trainer modules at startup,
Forge uses a registry pattern. Trainers register themselves via
decorators and are lazy-loaded only when needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from forge.config.schema.base import ForgeConfig


class TrainerProtocol(Protocol):
    """Protocol that all trainers must implement."""

    def train(self, config: ForgeConfig, resume: bool = False) -> None:
        """Run training with the given config."""
        ...

    def validate_config(self, config: ForgeConfig) -> None:
        """Validate that the config is compatible with this trainer."""
        ...


# Global registry
_REGISTRY: dict[str, type[TrainerProtocol]] = {}


def register_trainer(method: str) -> Callable:
    """Decorator to register a trainer class for a training method.

    Usage:
        @register_trainer("sft")
        class SFTTrainer:
            def train(self, config, resume=False): ...
            def validate_config(self, config): ...
    """

    def decorator(cls: type[TrainerProtocol]) -> type[TrainerProtocol]:
        _REGISTRY[method] = cls
        return cls

    return decorator


def get_trainer(method: str) -> TrainerProtocol:
    """Get a trainer instance for the given method.

    Lazy-imports the trainer module to avoid pulling in torch at CLI startup.
    """
    # Ensure trainers are registered by importing the modules
    _ensure_trainers_loaded()

    if method not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown training method '{method}'. Available: {available}")

    trainer_cls = _REGISTRY[method]
    return trainer_cls()


def list_methods() -> list[str]:
    """List all registered training methods."""
    _ensure_trainers_loaded()
    return sorted(_REGISTRY.keys())


_LOADED = False


def _ensure_trainers_loaded() -> None:
    """Dynamically load all trainer modules so their decorators execute."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    import sys

    # These are the submodules in forge.trainer that contain trainers.
    _trainer_modules = ["sft", "dpo", "orpo", "kto", "grpo", "reward", "simpo"]
    for mod_name in _trainer_modules:
        mod_path = f"forge.trainer.{mod_name}"
        try:
            if mod_path not in sys.modules:
                __import__(mod_path)
        except ImportError:
            # Silently ignore if a trainer file is missing or fails to import
            # because dependencies (like trl) aren't installed.
            pass
