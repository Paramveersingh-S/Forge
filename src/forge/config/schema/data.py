"""Data configuration schema."""


from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for training data."""

    path: str = Field(
        default="./data/train.jsonl",
        description="Path to training data file or HuggingFace dataset ID.",
    )
    format: str = Field(
        default="auto",
        description="Data format (auto, sharegpt, alpaca, jsonl, parquet, hf).",
    )
    eval_path: str | None = Field(
        default=None,
        description="Path to evaluation data. If None, splits from training data.",
    )
    eval_split: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Fraction of data to use for evaluation if no eval_path.",
    )
    max_samples: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of training samples (for debugging).",
    )
    preprocessing_num_workers: int = Field(
        default=4,
        ge=1,
        description="Number of workers for data preprocessing.",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Column names to use (for datasets with non-standard columns).",
    )
