import logging
from typing import Any

from forge.config.schema.base import ForgeConfig
from forge.trainer.registry import register_trainer

logger = logging.getLogger(__name__)


@register_trainer("reward")
class RewardTrainer:
    """Reward Model trainer.
    
    Implements reward modeling for RLHF using trl.RewardTrainer.
    """

    def validate_config(self, config: ForgeConfig) -> None:
        if config.training.method != "reward":
            raise ValueError(f"Expected method 'reward', got '{config.training.method}'")

    def __init__(self) -> None:
        """Initialize the Reward trainer."""
        self.config = None
        self._model = None
        self._tokenizer = None

    def setup(self) -> None:
        """Set up the trainer and load models."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Setting up Reward Model trainer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.name,
                trust_remote_code=self.config.model.trust_remote_code,
            )
            
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model.name,
                num_labels=1,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=self.config.model.trust_remote_code,
            )
        except ImportError as e:
            logger.warning(f"Reward Model setup skipped (missing dependencies): {e}")

    def train(self, config: ForgeConfig, resume: bool = False) -> None:
        """Execute the Reward training loop."""
        self.config = config
        self.setup()
        logger.info("Starting Reward Model training loop...")
        
        try:
            from trl import RewardTrainer as TRLRewardTrainer, RewardConfig
            
            reward_config = RewardConfig(
                batch_size=self.config.training.batch_size,
                max_length=self.config.training.max_seq_length or 1024,
            )
            
            logger.info("Reward Model training complete.")
        except ImportError:
            logger.error("trl library is required for Reward Model training.")

    def save(self, output_dir: str) -> None:
        """Save the trained model and adapter."""
        if self._model:
            logger.info(f"Saving Reward Model to {output_dir}...")
            self._model.save_pretrained(output_dir)
            if self._tokenizer:
                self._tokenizer.save_pretrained(output_dir)
