"""Tests for the trainer registry."""

import pytest

from forge.trainer.registry import register_trainer, get_trainer, list_methods, _REGISTRY
from forge.config.schema.base import ForgeConfig
from forge.config.schema.model import ModelConfig


class TestTrainerRegistry:
    """Test the decorator-based trainer registry."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        _REGISTRY.clear()

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
