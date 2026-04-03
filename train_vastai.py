"""
ownify — Unsloth LoRA training script for vast.ai (NVIDIA GPU)
Mirrors configs/openclaw-mlx-v3.yaml but runs on PyTorch + CUDA.

Usage:
    python train_vastai.py
"""

import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen3-4B"      # Qwen3 4B — matches local MLX training
ADAPTER_PATH  = "adapters/openclaw-mlx-v3"
DATA_FILE     = "data/openclaw-v3.jsonl"

MAX_SEQ_LENGTH = 2048   # More VRAM on cloud → use full length
LORA_RANK      = 8
LORA_LAYERS    = 8      # num_layers equivalent
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
    """Apply Qwen chat template to messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # auto-detect (bfloat16 on A100, float16 on others)
        load_in_4bit=False,  # full precision for better quality
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,   # scale equivalent
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Loading data: {DATA_FILE}")
    dataset = load_data(DATA_FILE)
    dataset = dataset.map(lambda x: format_example(x, tokenizer))

    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_data = split["train"]
    eval_data  = split["test"]

    print(f"Train: {len(train_data)} examples, Eval: {len(eval_data)} examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
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
            fp16=True,
            report_to="none",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            dataset_num_proc=2,
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
