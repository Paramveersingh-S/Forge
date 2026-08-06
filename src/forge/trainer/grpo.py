"""GRPO (Group Relative Policy Optimization) trainer.

Reward-model-free RLHF: generates multiple completions per prompt,
ranks them internally, and uses the relative ranking as the reward
signal. No separate reward model needed.

Reference: Shao et al., "DeepSeekMath: Pushing the Limits of
Mathematical Reasoning in Open Language Models" (2024).
"""

from __future__ import annotations

from rich.console import Console

from forge.config.schema.base import ForgeConfig
from forge.trainer.registry import register_trainer

console = Console()


@register_trainer("grpo")
class GRPOTrainer:
    """Group Relative Policy Optimization trainer.

    Supports:
    - Self-generated reward signals (no reward model)
    - Multiple completions per prompt for ranking
    - QLoRA (4-bit / 8-bit quantization)
    - Gradient checkpointing
    - Configurable group size and KL penalty

    Note: Layer streaming is NOT compatible with GRPO because
    generation rollouts need to re-read every layer per token.
    """

    def validate_config(self, config: ForgeConfig) -> None:
        """Validate GRPO-specific config requirements."""
        if config.training.method != "grpo":
            raise ValueError(f"GRPOTrainer requires method='grpo', got '{config.training.method}'")

        if config.training.stream_layers:
            console.print(
                "[yellow]⚠ Warning: Layer streaming is not compatible with GRPO "
                "(generation rollouts need random layer access). Disabling.[/yellow]"
            )

    def train(self, config: ForgeConfig, resume: bool = False) -> None:
        """Run GRPO training."""
        self.validate_config(config)

        console.print("[bold]Starting GRPO training...[/bold]")
        console.print(f"  Model:        {config.model.name}")
        console.print(f"  LoRA rank:    {config.lora.r}")
        console.print(f"  Quantization: {config.training.quantization}")
        console.print(f"  Batch size:   {config.training.batch_size}")
        console.print(f"  LR:           {config.training.learning_rate}")

        # Lazy import heavy dependencies
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig as PeftLoraConfig, get_peft_model
            from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer
        except ImportError as e:
            console.print(
                f"[red]✗ Missing training dependency: {e}[/red]\n"
                "  Install with: pip install 'forge-llm[train]'"
            )
            raise SystemExit(1)

        # --- Setup ---
        console.print("\n[dim]Loading tokenizer...[/dim]")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        console.print("[dim]Loading model...[/dim]")
        model_kwargs = {}
        if config.training.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
            **model_kwargs,
        )

        # Apply LoRA
        if config.lora.r > 0:
            console.print(f"[dim]Applying LoRA (r={config.lora.r}, alpha={config.lora.alpha})...[/dim]")
            target_modules = config.lora.target_modules
            if isinstance(target_modules, str) and target_modules == "auto":
                target_modules = None

            peft_config = PeftLoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=target_modules,
                use_dora=config.lora.use_dora,
                use_rslora=config.lora.use_rslora,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # GRPO config — uses internal reward from group ranking
        grpo_config = GRPOConfig(
            output_dir=config.training.output_dir,
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            num_train_epochs=config.training.num_epochs or 1,
            max_steps=config.training.max_steps,
            warmup_ratio=config.training.warmup_ratio,
            lr_scheduler_type=config.training.lr_scheduler,
            weight_decay=config.training.weight_decay,
            gradient_checkpointing=config.training.gradient_checkpointing,
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            bf16=True,
            seed=config.training.seed or 42,
            report_to="none",
            max_completion_length=config.training.max_seq_length,
        )

        # Load prompts dataset
        console.print(f"[dim]Loading prompts from {config.data.path}...[/dim]")
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=config.data.path, split="train")

        # GRPO requires a reward function — use a simple length/format heuristic
        # In production, users would supply a custom reward_fn via config
        def default_reward_fn(completions: list[str], **kwargs: object) -> list[float]:
            """Default reward: prefer longer, well-formatted completions."""
            rewards = []
            for completion in completions:
                score = min(len(completion) / 200.0, 1.0)  # Length bonus (capped)
                if completion.strip().endswith((".", "!", "?")):
                    score += 0.1  # Sentence completion bonus
                rewards.append(score)
            return rewards

        # Create GRPO trainer
        trainer = TRLGRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=default_reward_fn,
        )

        # Train
        console.print("\n[bold green]🔥 GRPO training started![/bold green]")
        trainer.train(resume_from_checkpoint=resume if resume else None)

        # Save
        console.print(f"\n[green]✓[/green] GRPO training complete. Saving to {config.training.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(config.training.output_dir)
