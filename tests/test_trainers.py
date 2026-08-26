"""Tests for DPO, GRPO, KTO, ORPO trainers."""

import pytest

from forge.config.schema.base import ForgeConfig
from forge.config.schema.model import ModelConfig
from forge.config.schema.training import TrainingConfig
from forge.trainer.registry import _REGISTRY, get_trainer, list_methods


class TestTrainerRegistry:
    """Test that all trainers are registered correctly."""

    def test_all_methods_registered(self) -> None:
        """All 5 core trainers should be available."""
        # Force re-registration
        methods = list_methods()
        assert "sft" in methods
        assert "dpo" in methods
        assert "grpo" in methods
        assert "kto" in methods
        assert "orpo" in methods

    def test_get_sft_trainer(self) -> None:

        trainer = get_trainer("sft")
        assert trainer is not None
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "validate_config")

    def test_get_dpo_trainer(self) -> None:

        trainer = get_trainer("dpo")
        assert trainer is not None

    def test_get_grpo_trainer(self) -> None:

        trainer = get_trainer("grpo")
        assert trainer is not None

    def test_get_kto_trainer(self) -> None:

        trainer = get_trainer("kto")
        assert trainer is not None

    def test_get_orpo_trainer(self) -> None:

        trainer = get_trainer("orpo")
        assert trainer is not None

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown training method"):
            get_trainer("nonexistent_method")


class TestTrainerValidation:
    """Test config validation for each trainer."""

    def test_sft_rejects_wrong_method(self) -> None:
        config = ForgeConfig(
            model=ModelConfig(name="test/model"),
            training=TrainingConfig(method="dpo"),
        )
        trainer = get_trainer("sft")
        with pytest.raises(ValueError, match="method='sft'"):
            trainer.validate_config(config)

    def test_dpo_rejects_wrong_method(self) -> None:
        config = ForgeConfig(
            model=ModelConfig(name="test/model"),
            training=TrainingConfig(method="sft"),
        )
        trainer = get_trainer("dpo")
        with pytest.raises(ValueError, match="method='dpo'"):
            trainer.validate_config(config)

    def test_grpo_rejects_wrong_method(self) -> None:
        config = ForgeConfig(
            model=ModelConfig(name="test/model"),
            training=TrainingConfig(method="sft"),
        )
        trainer = get_trainer("grpo")
        with pytest.raises(ValueError, match="method='grpo'"):
            trainer.validate_config(config)

    def test_kto_rejects_wrong_method(self) -> None:
        config = ForgeConfig(
            model=ModelConfig(name="test/model"),
            training=TrainingConfig(method="sft"),
        )
        trainer = get_trainer("kto")
        with pytest.raises(ValueError, match="method='kto'"):
            trainer.validate_config(config)

    def test_orpo_rejects_wrong_method(self) -> None:
        config = ForgeConfig(
            model=ModelConfig(name="test/model"),
            training=TrainingConfig(method="sft"),
        )
        trainer = get_trainer("orpo")
        with pytest.raises(ValueError, match="method='orpo'"):
            trainer.validate_config(config)

    def test_sft_accepts_correct_method(self) -> None:
        config = ForgeConfig(
            model=ModelConfig(name="test/model"),
            training=TrainingConfig(method="sft"),
        )
        trainer = get_trainer("sft")
        trainer.validate_config(config)  # Should not raise

    def test_dpo_accepts_correct_method(self) -> None:
        config = ForgeConfig(
            model=ModelConfig(name="test/model"),
            training=TrainingConfig(method="dpo"),
        )
        trainer = get_trainer("dpo")
        trainer.validate_config(config)  # Should not raise
