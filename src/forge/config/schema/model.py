"""Model configuration schema."""

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for the base model."""

    name: str = Field(
        ...,
        description="HuggingFace model ID or local path (e.g. 'meta-llama/Llama-3.1-8B-Instruct').",
    )
    type: str = Field(
        default="auto",
        description="Model architecture type (auto, llama, qwen2, mistral, gemma, phi3).",
    )
    context_length: int = Field(
        default=4096,
        ge=128,
        description="Maximum context length for training.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code from HuggingFace Hub.",
    )
    revision: str | None = Field(
        default=None,
        description="Specific model revision/commit hash.",
    )
    torch_dtype: str = Field(
        default="auto",
        description="Torch dtype for model loading (auto, bf16, fp16, fp32).",
    )
