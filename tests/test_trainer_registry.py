"""Tests for the trainer registry."""

import pytest

from forge.config.schema.base import ForgeConfig
from forge.trainer.registry import _REGISTRY, get_trainer, list_methods, register_trainer


class TestTrainerRegistry:
    """Test the decorator-based trainer registry."""

    @pytest.fixture(autouse=True)
    def isolate_registry(self):
        """Isolate the global trainer registry for each test."""
        from unittest.mock import patch
        from forge.trainer.registry import _ensure_trainers_loaded
        _ensure_trainers_loaded()
        with patch.dict(_REGISTRY, {}, clear=True):
            yield

    def test_register_and_retrieve(self) -> None:
        @register_trainer("test_method")
        class TestTrainer:
            def train(self, config: ForgeConfig, resume: bool = False) -> None:
                pass

            def validate_config(self, config: ForgeConfig) -> None:
                pass

        trainer = get_trainer("test_method")
        assert isinstance(trainer, TestTrainer)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown training method"):
            get_trainer("nonexistent")

    def test_list_methods(self) -> None:
        @register_trainer("alpha")
        class AlphaTrainer:
            def train(self, config: ForgeConfig, resume: bool = False) -> None:
                pass

            def validate_config(self, config: ForgeConfig) -> None:
                pass

        @register_trainer("beta")
        class BetaTrainer:
            def train(self, config: ForgeConfig, resume: bool = False) -> None:
                pass

            def validate_config(self, config: ForgeConfig) -> None:
                pass

        methods = list_methods()
        assert "alpha" in methods
        assert "beta" in methods
