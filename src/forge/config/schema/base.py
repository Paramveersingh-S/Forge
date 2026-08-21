"""Base ForgeConfig — the root configuration model.

This is intentionally small (~50 lines) compared to Soup's 6829-line
monolith. Each concern lives in its own schema module.
"""


from pydantic import BaseModel, Field

from forge.config.schema.data import DataConfig
from forge.config.schema.lora import LoraConfig
from forge.config.schema.model import ModelConfig
from forge.config.schema.training import TrainingConfig


class ForgeConfig(BaseModel):
    """Root configuration for a Forge training run.

    Composes ModelConfig, LoraConfig, TrainingConfig, and DataConfig
    into a single validated structure. Each sub-config is its own
    Pydantic model in a separate file.
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    # Metadata — not used for training, but tracked for provenance.
    project_name: str | None = Field(
        default=None,
        description="Project name for experiment tracking.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for organizing experiments.",
    )
