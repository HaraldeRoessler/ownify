"""
ownify — Prepare training data for MLX LoRA fine-tuning.

Splits openclaw training data into train/valid/test sets
and strips the category field (MLX expects only 'messages').
"""

import json
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/openclaw-v1.jsonl")
    parser.add_argument("--output", default="data/mlx-openclaw")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load examples
    examples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # MLX expects only 'messages' key
                examples.append({"messages": data["messages"]})

    random.seed(args.seed)
    random.shuffle(examples)

    n = len(examples)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)

    train = examples[:n_train]
    valid = examples[n_train:n_train + n_valid]
    test = examples[n_train + n_valid:]

    # Ensure valid and test have at least 1 example
    if not valid and train:
        valid = [train.pop()]
    if not test and train:
        test = [train.pop()]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for split, data in [("train", train), ("valid", valid), ("test", test)]:
        path = output / f"{split}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"{split}: {len(data)} examples -> {path}")

    print(f"\nTotal: {n} examples")


if __name__ == "__main__":
    main()
