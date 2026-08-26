"""Tests for the composable config system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from forge.config.loader import _deep_merge, create_config_from_preset, load_config
from forge.config.schema.base import ForgeConfig
from forge.config.schema.data import DataConfig
from forge.config.schema.lora import LoraConfig
from forge.config.schema.model import ModelConfig
from forge.config.schema.training import TrainingConfig


class TestDeepMerge:
    """Test the deep merge utility."""

    def test_simple_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"training": {"batch_size": 4, "lr": 1e-4}}
        override = {"training": {"batch_size": 8}}
        result = _deep_merge(base, override)
        assert result == {"training": {"batch_size": 8, "lr": 1e-4}}

    def test_override_replaces_non_dict(self) -> None:
        base = {"a": [1, 2, 3]}
        override = {"a": [4, 5]}
        result = _deep_merge(base, override)
        assert result == {"a": [4, 5]}

    def test_base_unchanged(self) -> None:
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}  # Original not mutated


class TestForgeConfig:
    """Test the root ForgeConfig schema."""

    def test_defaults(self) -> None:
        config = ForgeConfig(model=ModelConfig(name="test/model"))
        assert config.model.name == "test/model"
        assert config.lora.r == 64
        assert config.training.method == "sft"
        assert config.training.quantization == "4bit"

    def test_full_config(self) -> None:
        config = ForgeConfig(
            model=ModelConfig(name="meta-llama/Llama-3.1-8B-Instruct", type="llama"),
            lora=LoraConfig(r=32, alpha=16, dropout=0.1),
            training=TrainingConfig(
                method="sft",
                batch_size=8,
                learning_rate=3e-4,
                stream_layers=True,
            ),
            data=DataConfig(path="./train.jsonl"),
        )
        assert config.model.type == "llama"
        assert config.lora.r == 32
        assert config.training.stream_layers is True


class TestLoraConfig:
    """Test LoRA config validation."""

    def test_defaults(self) -> None:
        config = LoraConfig()
        assert config.r == 64
        assert config.alpha == 16
        assert config.target_modules == "auto"

    def test_full_finetune(self) -> None:
        config = LoraConfig(r=0)
        assert config.r == 0

    def test_dora_requires_rank(self) -> None:
        with pytest.raises(ValueError, match="DoRA requires r > 0"):
            LoraConfig(r=0, use_dora=True)

    def test_dropout_bounds(self) -> None:
        with pytest.raises(Exception):
            LoraConfig(dropout=1.5)


class TestTrainingConfig:
    """Test training config validation."""

    def test_defaults(self) -> None:
        config = TrainingConfig()
        assert config.method == "sft"
        assert config.batch_size == 4
        assert config.stream_buffers == 2

    def test_stream_buffers_bounds(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(stream_buffers=1)  # Min is 2
        with pytest.raises(Exception):
            TrainingConfig(stream_buffers=10)  # Max is 8


class TestConfigLoader:
    """Test YAML config loading and fragment resolution."""

    def test_load_simple_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "model": {"name": "test/model"},
                    "training": {"method": "sft", "batch_size": 8},
                },
                f,
            )
            f.flush()

            config = load_config(f.name)
            assert config.model.name == "test/model"
            assert config.training.batch_size == 8

    def test_create_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "forge.yaml")
            result = create_config_from_preset("llama3-8b-chat", output)
            assert Path(result).exists()

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            create_config_from_preset("nonexistent-model")

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")
