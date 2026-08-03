"""Composable YAML config loader.

Design: Instead of one monolithic 6800-line schema, Forge uses composable
config fragments that inherit and merge. Each concern (model, LoRA, training,
data, eval) gets its own small Pydantic model.

A config file can `extends:` other fragments, and `overrides:` specific fields.
Fragments are deep-merged in order, with later values taking precedence.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from rich.console import Console

from forge.config.schema.base import ForgeConfig
from forge.config.schema.data import DataConfig
from forge.config.schema.lora import LoraConfig
from forge.config.schema.model import ModelConfig
from forge.config.schema.training import TrainingConfig

console = Console()

# Search paths for config fragments (recipes, presets).
_FRAGMENT_SEARCH_PATHS: List[Path] = [
    Path(__file__).parent.parent.parent.parent / "recipes",
    Path(__file__).parent.parent.parent.parent / "presets",
    Path.cwd(),
]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_fragment(name: str) -> Path:
    """Find a fragment file by name across search paths."""
    # If it's already an absolute or relative path that exists, use it directly
    p = Path(name)
    if p.exists():
        return p

    # Search through known directories
    for search_dir in _FRAGMENT_SEARCH_PATHS:
        candidate = search_dir / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Config fragment not found: '{name}'. "
        f"Searched: {[str(p) for p in _FRAGMENT_SEARCH_PATHS]}"
    )


def _load_raw_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file as a raw dict."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return raw or {}


def _resolve_and_merge(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve `extends` chains and apply `overrides`."""
    extends = raw.pop("extends", [])
    overrides = raw.pop("overrides", {})

    # Start with the base (non-extends, non-overrides fields)
    merged = copy.deepcopy(raw)

    # Process extends in order (each one layers on top)
    if isinstance(extends, str):
        extends = [extends]

    for fragment_name in extends:
        fragment_path = _resolve_fragment(fragment_name)
        fragment_raw = _load_raw_yaml(fragment_path)
        # Recursively resolve the fragment's own extends
        fragment_resolved = _resolve_and_merge(fragment_raw)
        merged = _deep_merge(fragment_resolved, merged)

    # Apply overrides last
    if overrides:
        merged = _deep_merge(merged, overrides)

    return merged


def load_config(path: str | Path) -> ForgeConfig:
    """Load and validate a Forge config file.

    Resolves `extends` chains, applies `overrides`, and validates
    the result against the Pydantic schema.

    Args:
        path: Path to the forge.yaml config file.

    Returns:
        Validated ForgeConfig instance.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _load_raw_yaml(config_path)
    merged = _resolve_and_merge(raw)

    return ForgeConfig(**merged)


def create_config_from_preset(
    preset: str,
    output: str = "forge.yaml",
    wizard: bool = False,
) -> str:
    """Create a new config file from a preset template.

    Args:
        preset: Preset name (e.g. "llama3-8b-chat").
        output: Output file path.
        wizard: If True, run interactive wizard.

    Returns:
        Path to the created config file.
    """
    # Map preset names to recipe files
    preset_map = {
        "llama3-8b-chat": {
            "extends": ["recipes/llama3-8b.yaml", "presets/qlora-4bit.yaml"],
            "training": {"method": "sft", "batch_size": 4, "max_steps": 1000},
            "data": {"path": "./data/train.jsonl", "format": "sharegpt"},
        },
        "qwen3-7b": {
            "extends": ["recipes/qwen3-7b.yaml", "presets/qlora-4bit.yaml"],
            "training": {"method": "sft", "batch_size": 4, "max_steps": 1000},
            "data": {"path": "./data/train.jsonl", "format": "sharegpt"},
        },
        "mistral-7b": {
            "extends": ["recipes/mistral-7b.yaml", "presets/qlora-4bit.yaml"],
            "training": {"method": "sft", "batch_size": 4, "max_steps": 1000},
            "data": {"path": "./data/train.jsonl", "format": "sharegpt"},
        },
    }

    if preset not in preset_map:
        available = ", ".join(preset_map.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    config = preset_map[preset]

    output_path = Path(output)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return str(output_path)
