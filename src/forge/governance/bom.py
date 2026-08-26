import datetime
import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from forge.config.loader import load_config
from forge.config.schema.base import ForgeConfig


class MLBOM(BaseModel):
    """Machine Learning Bill of Materials (ML-BOM).

    Captures the exact provenance of a fine-tuned adapter, including
    the base model, training config, datasets, and exact cryptographic hashes.
    """

    version: str = "1.0"
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    model_name: str
    base_model: str
    adapter_hash: str | None = None
    training_method: str
    dataset_hashes: dict[str, str] = Field(default_factory=dict)
    hyperparameters: dict = Field(default_factory=dict)
    forge_version: str

    def to_json(self) -> str:
        """Serialize to formatted JSON."""
        return self.model_dump_json(indent=2)


def generate_bom(adapter_dir: str | Path, base_model: str) -> MLBOM:
    """Generate an ML-BOM for a fine-tuned adapter.

    Args:
        adapter_dir: Path to the trained adapter directory.
        base_model: The base model name or path.

    Returns:
        An MLBOM instance.
    """
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    # Try to load training config if it exists
    config_path = adapter_path / "forge.yaml"
    hyperparameters = {}
    training_method = "unknown"
    dataset_hashes = {}  # type: ignore

    if config_path.exists():
        try:
            config: ForgeConfig = load_config(str(config_path))
            training_method = config.training.method
            hyperparameters = config.training.model_dump()
            if config.data:
                dataset_hashes = {}
                for d in config.data.datasets:  # type: ignore
                    try:
                        import forge_core.crypto

                        d_hash = forge_core.crypto.sha256_file(str(d.path))
                        dataset_hashes[d.path] = f"sha256:{d_hash}"
                    except Exception:
                        dataset_hashes[d.path] = "sha256:unknown"
        except Exception:
            pass

    # Try to hash the safetensors file
    adapter_hash = None
    adapter_file = adapter_path / "adapter_model.safetensors"
    if adapter_file.exists():
        try:
            import forge_core.crypto

            adapter_hash = forge_core.crypto.sha256_file(str(adapter_file))
        except ImportError:
            # Fallback if forge_core isn't available
            import hashlib

            hasher = hashlib.sha256()
            with open(adapter_file, "rb") as f:
                while chunk := f.read(8 * 1024 * 1024):
                    hasher.update(chunk)
            adapter_hash = hasher.hexdigest()

    from forge import __version__ as forge_version

    return MLBOM(
        model_name=adapter_path.name,
        base_model=base_model,
        adapter_hash=adapter_hash,
        training_method=training_method,
        dataset_hashes=dataset_hashes,
        hyperparameters=hyperparameters,
        forge_version=forge_version,
    )
