"""
ownify — Training script for vast.ai (NVIDIA GPU).
Uses standard PEFT/Transformers so adapter keys match locally.

Usage:
    python train_vastai.py
"""

import json
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
import torch

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen3.5-4B"
ADAPTER_PATH  = "adapters/openclaw-mlx-v3"
DATA_FILE     = "data/openclaw-v3.jsonl"

MAX_SEQ_LENGTH = 2048
LORA_RANK      = 8
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LR             = 1e-5
MAX_STEPS      = 800
SEED           = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_example(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading data: {DATA_FILE}")
    dataset = load_data(DATA_FILE)
    dataset = dataset.map(lambda x: format_example(x, tokenizer))
    split = dataset.train_test_split(test_size=0.1, seed=SEED)

    print(f"Train: {len(split['train'])} examples, Eval: {len(split['test'])} examples")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=SFTConfig(
            output_dir=ADAPTER_PATH,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_steps=20,
            seed=SEED,
            logging_steps=10,
            eval_steps=100,
            eval_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            bf16=True,
            report_to="none",
            dataset_text_field="text",
            gradient_checkpointing=True,
        ),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {ADAPTER_PATH}/")
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
