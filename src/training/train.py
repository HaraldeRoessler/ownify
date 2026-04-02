"""
ownify — LoRA fine-tuning on Qwen 3.5 4B

Trains a personal LoRA adapter that encodes behavior and escalation decisions
into the model weights. Runs on Apple Silicon (MPS) or CUDA.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig


def load_training_data(path: str) -> Dataset:
    """Load JSONL training examples into a HuggingFace Dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return Dataset.from_list(examples)


def get_device():
    """Pick the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train ownify LoRA adapter")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="Base model from HuggingFace Hub",
    )
    parser.add_argument(
        "--data",
        default="data/openclaw-v1.jsonl",
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--output",
        default="adapters/ownify-v1",
        help="Output directory for the adapter",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Training data: {args.data}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "mps":
        model = model.to(device)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print("Loading training data...")
    dataset = load_training_data(args.data)
    print(f"Training examples: {len(dataset)}")

    # Training config
    output_dir = Path(args.output)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=2,
        logging_steps=1,
        save_strategy="epoch",
        fp16=device == "cuda",
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        max_length=args.max_seq_length,
    )

    # Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save adapter
    print(f"Saving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
