"""ORPO (Odds-Ratio Preference Optimization) trainer.

Combines SFT and preference alignment into a single training step
without needing a separate reference model, reducing memory usage
and training time compared to DPO.

Reference: Hong et al., "ORPO: Monolithic Preference Optimization
without Reference Model" (2024).
"""

from __future__ import annotations

from rich.console import Console

from forge.config.schema.base import ForgeConfig
from forge.trainer.registry import register_trainer

console = Console()


@register_trainer("orpo")
class ORPOTrainer:
    """Odds-Ratio Preference Optimization trainer.

    Supports:
    - Paired preference datasets (chosen / rejected)
    - No reference model required (memory efficient)
    - QLoRA (4-bit / 8-bit quantization)
    - Layer streaming (stream_layers: true)
    - Gradient checkpointing
    """

    def validate_config(self, config: ForgeConfig) -> None:
        """Validate ORPO-specific config requirements."""
        if config.training.method != "orpo":
            raise ValueError(f"ORPOTrainer requires method='orpo', got '{config.training.method}'")

    def train(self, config: ForgeConfig, resume: bool = False) -> None:
        """Run ORPO training."""
        self.validate_config(config)

        console.print("[bold]Starting ORPO training...[/bold]")
        console.print(f"  Model:        {config.model.name}")
        console.print(f"  LoRA rank:    {config.lora.r}")
        console.print(f"  Quantization: {config.training.quantization}")
        console.print(f"  Batch size:   {config.training.batch_size}")
        console.print(f"  LR:           {config.training.learning_rate}")
        console.print(f"  Stream:       {config.training.stream_layers}")

        # Lazy import heavy dependencies
        try:
            import torch
            from peft import LoraConfig as PeftLoraConfig
            from peft import get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import ORPOConfig
            from trl import ORPOTrainer as TRLORPOTrainer
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

        # ORPO config — no reference model needed
        orpo_config = ORPOConfig(
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
            beta=0.1,  # Odds-ratio weight
            max_length=config.training.max_seq_length,
        )

        # Load preference dataset
        console.print(f"[dim]Loading preference dataset from {config.data.path}...[/dim]")
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=config.data.path, split="train")

        # Add tracking callback
        from forge.tracking.callback import ForgeTrainerCallback
        tracking_callback = ForgeTrainerCallback(
            experiment_name=config.project_name or config.model.name.split("/")[-1],
            config=config.model_dump(),
            tags=config.tags + ["orpo"],
        )

        # Create ORPO trainer — no ref_model!
        trainer = TRLORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[tracking_callback],
        )

        # Train
        console.print("\n[bold green]🔥 ORPO training started![/bold green]")
        trainer.train(resume_from_checkpoint=resume if resume else None)

        # Save
        console.print(f"\n[green]✓[/green] ORPO training complete. Saving to {config.training.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(config.training.output_dir)
