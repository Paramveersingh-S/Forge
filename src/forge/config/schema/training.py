"""Training configuration schema."""


from pydantic import BaseModel, Field

SUPPORTED_METHODS = (
    "sft", "dpo", "grpo", "kto", "orpo", "simpo", "ppo",
    "distill", "reward_model", "unlearn",
)

SUPPORTED_QUANTIZATIONS = ("none", "4bit", "8bit")


class TrainingConfig(BaseModel):
    """Configuration for the training loop."""

    method: str = Field(
        default="sft",
        description=f"Training method. One of: {', '.join(SUPPORTED_METHODS)}.",
    )
    quantization: str = Field(
        default="4bit",
        description="Weight quantization (none, 4bit, 8bit).",
    )
    batch_size: int = Field(default=4, ge=1, description="Per-device training batch size.")
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, description="Gradient accumulation steps."
    )
    learning_rate: float = Field(default=2e-4, gt=0, description="Peak learning rate.")
    num_epochs: int | None = Field(default=None, ge=1, description="Number of training epochs.")
    max_steps: int = Field(default=-1, description="Max training steps (-1 = use epochs).")
    max_seq_length: int = Field(default=2048, ge=64, description="Maximum sequence length.")
    warmup_ratio: float = Field(default=0.03, ge=0, le=1, description="Warmup ratio.")
    lr_scheduler: str = Field(default="cosine", description="LR scheduler type.")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay.")
    seed: int | None = Field(default=None, description="Random seed for reproducibility.")
    gradient_checkpointing: bool = Field(
        default=True, description="Enable gradient checkpointing to save VRAM."
    )
    output_dir: str = Field(default="./output", description="Output directory for checkpoints.")

    # Layer streaming
    stream_layers: bool = Field(
        default=False,
        description="Enable layer streaming — keep base in RAM, stream one layer at a time.",
    )
    stream_source: str = Field(
        default="auto",
        description="Stream source tier (auto, ram, disk).",
    )
    stream_buffers: int = Field(
        default=2,
        ge=2,
        le=8,
        description="Number of VRAM buffers for double-buffered prefetch.",
    )

    # Multi-GPU
    use_deepspeed: bool = Field(default=False, description="Enable DeepSpeed ZeRO.")
    deepspeed_stage: int = Field(default=2, ge=1, le=3, description="DeepSpeed ZeRO stage.")
    use_fsdp: bool = Field(default=False, description="Enable FSDP2.")
