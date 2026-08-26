"""Integration tests for the training loop and layer streaming."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from forge.config.schema.base import ForgeConfig
from forge.config.schema.data import DataConfig
from forge.config.schema.model import ModelConfig
from forge.config.schema.training import TrainingConfig
from forge.trainer.sft import SFTTrainer


@pytest.fixture
def dummy_dataset(tmp_path):
    """Create a small dummy dataset."""
    dataset_path = tmp_path / "dummy.json"
    data = [{"text": "Hello world!"}, {"text": "Integration testing is fun."}]
    with open(dataset_path, "w") as f:
        json.dump(data, f)
    return str(dataset_path)


@pytest.mark.integration
def test_cpu_training_loop_with_streaming(dummy_dataset, tmp_path):
    """Test that the training loop and streaming hooks run successfully on CPU."""
    # 1. Setup Config
    config = ForgeConfig(
        model=ModelConfig(name="gpt2", trust_remote_code=True),
        data=DataConfig(path=dummy_dataset),
        training=TrainingConfig(
            method="sft",
            stream_layers=True,  # Enable streaming
            output_dir=str(tmp_path / "output"),
            max_steps=1,
            batch_size=1,
        ),
    )

    trainer = SFTTrainer()

    import sys

    mock_transformers = MagicMock()
    mock_trl = MagicMock()
    mock_peft = MagicMock()
    mock_datasets = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "transformers": mock_transformers,
            "trl": mock_trl,
            "peft": mock_peft,
            "datasets": mock_datasets,
        },
    ):
        # Mock Tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock Model with dummy layers to test streaming hook registration
        class DummyLayer(nn.Module):
            def forward(self, x):
                return x

        class DummyModelInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([DummyLayer() for _ in range(4)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class DummyCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = DummyModelInner()

            def forward(self, x):
                return self.model(x)

            def print_trainable_parameters(self):
                pass

        mock_model_instance = DummyCausalLM()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model_instance
        mock_peft.get_peft_model.return_value = mock_model_instance

        # Mock TRL Trainer so we don't actually run HF trainer loop
        mock_trl_instance = MagicMock()
        mock_trl.SFTTrainer.return_value = mock_trl_instance

        # Mock Dataset
        mock_datasets.load_dataset.return_value = MagicMock()

        # 3. Execute training initialization
        trainer.train(config)

        # 4. Verify Hooks were registered
        hooks_registered = 0
        for layer in mock_model_instance.model.layers:
            # Check if any forward pre-hooks are present
            if layer._forward_pre_hooks:
                hooks_registered += 1

                # Test that the hook can be triggered without error
                # On CPU, prefetch just skips if CUDA is not available, but should not crash
                hook_func = list(layer._forward_pre_hooks.values())[0]
                hook_func(layer, (torch.zeros(1),))

        assert hooks_registered == 4, f"Expected 4 hooks, got {hooks_registered}"

        # 5. Verify the TRL Trainer was called
        mock_trl.SFTTrainer.assert_called_once()
        mock_trl_instance.train.assert_called_once()
        mock_trl_instance.save_model.assert_called_once()
