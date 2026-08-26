"""Local evaluation engine for adapter validation.

Uses HuggingFace pipelines to run local inference evaluations on
test suites to act as a gate for `forge ship`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# Basic threshold for gating
DEFAULT_THRESHOLD = 0.60


def evaluate_adapter(
    base_model: str,
    adapter_path: str | Path,
    task_file: str | Path,
) -> dict[str, Any]:
    """Evaluate an adapter on a JSONL task file.

    Args:
        base_model: Name or path to the base model.
        adapter_path: Path to the trained LoRA adapter.
        task_file: Path to a JSONL file with 'prompt' and 'expected' keys.

    Returns:
        A dictionary with evaluation metrics (score, total, passed).
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        console.print("[red]✗ Transformers or PEFT not installed.[/red]")
        raise RuntimeError("Missing evaluation dependencies.")

    task_path = Path(task_file)
    if not task_path.exists():
        console.print(f"[red]✗ Task file not found: {task_path}[/red]")
        raise FileNotFoundError(f"Task file not found: {task_path}")

    console.print(f"[cyan]Loading base model {base_model}...[/cyan]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    console.print(f"[cyan]Applying adapter {adapter_path}...[/cyan]")
    model = PeftModel.from_pretrained(model, adapter_path)  # type: ignore
    model.eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device,
    )

    console.print(f"[cyan]Running evaluation suite: {task_path.name}...[/cyan]")

    tasks = []
    with open(task_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if not tasks:
        console.print("[yellow]⚠ Task file is empty.[/yellow]")
        return {"score": 1.0, "passed": 0, "total": 0}

    passed = 0
    total = len(tasks)

    for idx, task in enumerate(tasks):
        prompt = task.get("prompt", "")
        expected = task.get("expected", "")

        # Run inference
        outputs = generator(
            prompt,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = outputs[0]["generated_text"][len(prompt) :].strip()

        # Simple exact substring match for basic evaluation
        if expected.lower() in generated_text.lower():
            passed += 1

        if idx % 10 == 0 and idx > 0:
            console.print(f"  Processed {idx}/{total} tasks...")

    score = passed / total if total > 0 else 0.0

    console.print(f"[green]✓ Evaluation complete. Score: {score:.1%} ({passed}/{total})[/green]")

    return {
        "score": score,
        "passed": passed,
        "total": total,
    }
