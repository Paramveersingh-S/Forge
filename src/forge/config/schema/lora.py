"""LoRA configuration schema."""


from pydantic import BaseModel, Field, model_validator


class LoraConfig(BaseModel):
    """Configuration for LoRA / QLoRA adapters."""

    r: int = Field(
        default=64,
        ge=0,
        description="LoRA rank. 0 = full fine-tuning (no adapter).",
    )
    alpha: int = Field(default=16, description="LoRA alpha scaling factor.")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout rate.")
    target_modules: str | list[str] = Field(
        default="auto",
        description="Target modules for LoRA. 'auto' = let PEFT decide.",
    )
    use_dora: bool = Field(
        default=False,
        description="Enable DoRA (Weight-Decomposed Low-Rank Adaptation).",
    )
    use_rslora: bool = Field(
        default=False,
        description="Enable rank-stabilized LoRA scaling.",
    )
    init_strategy: str = Field(
        default="default",
        description="LoRA weight initialization (default, olora, pissa, loftq).",
    )
    rank_pattern: dict[str, int] | None = Field(
        default=None,
        description="Per-module rank overrides (e.g. {'experts.*.w1': 16}).",
    )

    @model_validator(mode="after")
    def _validate_mutual_exclusions(self) -> "LoraConfig":
        if self.r == 0 and self.use_dora:
            raise ValueError("DoRA requires r > 0 (cannot use with full fine-tuning).")
        return self
