"""
ownify — Convert LoRA adapter from safetensors to GGUF format for Ollama.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import gguf
from safetensors.numpy import load_file


def main():
    parser = argparse.ArgumentParser(description="Convert LoRA adapter to GGUF")
    parser.add_argument("--adapter", default="adapters/openclaw-v1", help="Adapter directory")
    parser.add_argument("--output", default=None, help="Output GGUF file path")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    output_path = args.output or str(adapter_dir / "openclaw-v1.gguf")

    # Load adapter config
    config_path = adapter_dir / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load safetensors weights
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    print(f"Loading adapter from {safetensors_path}...")
    tensors = load_file(str(safetensors_path))

    print(f"Found {len(tensors)} tensors")
    print(f"LoRA rank: {config['r']}, alpha: {config['lora_alpha']}")

    # Create GGUF writer
    writer = gguf.GGUFWriter(output_path, "llama")

    # Write LoRA metadata
    writer.add_string("general.type", "adapter")
    writer.add_string("general.architecture", "llama")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", float(config["lora_alpha"]))

    # Convert and write tensors
    for name, tensor in tensors.items():
        # Convert PEFT naming to GGML naming
        # PEFT: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        # GGML: blk.0.attn_q.weight.loraA
        ggml_name = name
        ggml_name = ggml_name.replace("base_model.model.model.", "")
        ggml_name = ggml_name.replace("base_model.model.", "")

        # Map layer names
        ggml_name = ggml_name.replace("layers.", "blk.")
        ggml_name = ggml_name.replace("self_attn.q_proj", "attn_q")
        ggml_name = ggml_name.replace("self_attn.k_proj", "attn_k")
        ggml_name = ggml_name.replace("self_attn.v_proj", "attn_v")
        ggml_name = ggml_name.replace("self_attn.o_proj", "attn_output")
        ggml_name = ggml_name.replace("mlp.gate_proj", "ffn_gate")
        ggml_name = ggml_name.replace("mlp.up_proj", "ffn_up")
        ggml_name = ggml_name.replace("mlp.down_proj", "ffn_down")

        # Map LoRA A/B
        ggml_name = ggml_name.replace(".lora_A.weight", ".weight.loraA")
        ggml_name = ggml_name.replace(".lora_B.weight", ".weight.loraB")

        data = tensor.astype(np.float32)
        print(f"  {name} -> {ggml_name} [{data.shape}]")
        writer.add_tensor(ggml_name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\nAdapter saved to {output_path} ({output_size:.1f} MB)")


if __name__ == "__main__":
    main()
