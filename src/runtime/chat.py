"""
ownify — Interactive chat using MLX with openclaw adapter.

Usage:
    python src/runtime/chat.py
    python src/runtime/chat.py --adapter-path adapters/openclaw-mlx-v1
"""

import argparse

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def main():
    parser = argparse.ArgumentParser(description="ownify chat")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-path", default="adapters/openclaw-mlx-v2")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()

    sampler = make_sampler(temp=args.temp, top_p=0.9)

    print("Loading ownify...")
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
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

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )

        # Clean up response
        response = response.strip()
        if response.startswith("<think>"):
            # Strip thinking tags if present
            think_end = response.find("</think>")
            if think_end != -1:
                response = response[think_end + 8:].strip()

        history.append({"role": "assistant", "content": response})
        print(f"\nownify> {response}\n")


if __name__ == "__main__":
    main()
