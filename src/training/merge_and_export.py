"""
ownify — Merge LoRA adapter into base model and export for Ollama.

Since Ollama doesn't support LoRA adapters for all architectures,
we merge the adapter into the base model weights and export as a
single model.
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter", default="adapters/openclaw-v1")
    parser.add_argument("--output", default="models/ownify-openclaw-v1")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nMerged model saved. To create Ollama model:")
    print(f"  1. Convert to GGUF (needs llama.cpp):")
    print(f"     python llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile {output_dir}/ownify.gguf --outtype q4_k_m")
    print(f"  2. Create Ollama model:")
    print(f"     ollama create ownify -f {output_dir}/Modelfile")


if __name__ == "__main__":
    main()
