"""
ownify — Chat using PyTorch/MPS with PEFT adapter.
Works with the Unsloth-trained adapter from vast.ai.

Usage:
    python src/runtime/chat_peft.py
"""

import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME   = "Qwen/Qwen3.5-4B"
ADAPTER_PATH = "adapters/adapters/openclaw-mlx-v3/checkpoint-800"

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


def detect_escalation(response: str) -> tuple[bool, str]:
    match = re.search(r'<escalate\s+reason="([^"]+)"\s*/>', response)
    if match:
        return True, match.group(1)
    return False, ""


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )
    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.to(device)
    model.eval()
    print("Ready. Type 'quit' to exit.\n")

    history = []

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "/bye"):
            print("Bye.")
            break
        if user_input == "/clear":
            history = []
            print("History cleared.\n")
            continue

        history.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip thinking tags if present
        if "<think>" in response:
            think_end = response.find("</think>")
            if think_end != -1:
                response = response[think_end + 8:].strip()

        wants_escalation, reason = detect_escalation(response)
        if wants_escalation:
            print(f"\n[escalation requested: {reason}]")

        history.append({"role": "assistant", "content": response})
        print(f"\nownify> {response}\n")


if __name__ == "__main__":
    main()
