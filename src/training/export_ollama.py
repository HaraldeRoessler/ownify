"""
ownify — Export LoRA adapter to Ollama

Converts the trained adapter to GGUF format and creates a custom Ollama model.
No system prompt — behavior lives in the weights.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export ownify adapter to Ollama")
    parser.add_argument(
        "--adapter",
        default="adapters/ownify-v1",
        help="Path to the trained adapter",
    )
    parser.add_argument(
        "--base-ollama-model",
        default="qwen3.5:4b",
        help="Base Ollama model name",
    )
    parser.add_argument(
        "--name",
        default="ownify",
        help="Name for the Ollama model",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"Adapter not found at {adapter_path}")
        sys.exit(1)

    # Create Modelfile — no system prompt, behavior is in the weights
    modelfile = f"""FROM {args.base_ollama_model}
ADAPTER {adapter_path.resolve()}

# No system prompt — ownify's behavior lives in the LoRA weights.
# Only parameters that affect generation quality:
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
"""

    modelfile_path = adapter_path / "Modelfile"
    modelfile_path.write_text(modelfile)
    print(f"Created Modelfile at {modelfile_path}")

    # Create Ollama model
    print(f"Creating Ollama model '{args.name}'...")
    result = subprocess.run(
        ["ollama", "create", args.name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"Model '{args.name}' created successfully.")
        print(f"\nTest it:\n  ollama run {args.name}")
    else:
        print(f"Error creating model:\n{result.stderr}")
        print("\nNote: Ollama may need the adapter in GGUF format.")
        print("If this fails, we'll convert with llama.cpp first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
