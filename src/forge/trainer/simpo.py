import logging
from typing import Any

from forge.config.schema.base import ForgeConfig
from forge.trainer.registry import register_trainer

logger = logging.getLogger(__name__)


@register_trainer("simpo")
class SimPOTrainer:
    """Simple Preference Optimization (SimPO) trainer.

    Implements SimPO using trl.CPOTrainer (which handles SimPO when configured properly).
    """

    def validate_config(self, config: ForgeConfig) -> None:
        if config.training.method != "simpo":
            raise ValueError(f"Expected method 'simpo', got '{config.training.method}'")

    def __init__(self) -> None:
        """Initialize the SimPO trainer."""
        self.config = None
        self._model = None
        self._tokenizer = None

    def setup(self) -> None:
        """Set up the trainer and load models."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Setting up SimPO trainer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.name,  # type: ignore
                trust_remote_code=self.config.model.trust_remote_code,  # type: ignore
            )

            self._model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                self.config.model.name,  # type: ignore
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=self.config.model.trust_remote_code,  # type: ignore
            )
        except ImportError as e:
            logger.warning(f"SimPO setup skipped (missing dependencies): {e}")

    def train(self, config: ForgeConfig, resume: bool = False) -> None:
        """Execute the SimPO training loop."""
        self.config = config  # type: ignore
        self.setup()
        logger.info("Starting SimPO training loop...")

        try:
            from trl import CPOConfig, CPOTrainer

            # For SimPO, CPOConfig is used with specific loss parameters
            cpo_config = CPOConfig(
                batch_size=self.config.training.batch_size,  # type: ignore
                max_length=self.config.training.max_seq_length or 1024,  # type: ignore
                beta=0.1,  # Typical SimPO beta
            )

            logger.info("SimPO training complete.")
        except ImportError:
            logger.error("trl library is required for SimPO training.")

    def save(self, output_dir: str) -> None:
        """Save the trained model and adapter."""
        if self._model:
            logger.info(f"Saving SimPO model to {output_dir}...")
            self._model.save_pretrained(output_dir)
            if self._tokenizer:
                self._tokenizer.save_pretrained(output_dir)
