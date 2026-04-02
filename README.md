# ownify

**Your AI belongs to you.**

ownify is a personal AI that runs entirely on your own device — your laptop, your desktop, your phone. It learns how you think, how you work, and what you know. That knowledge stays with you. Always.

## The Problem

Today's AI assistants store your conversations, your preferences, and your patterns on someone else's servers. Your personal knowledge becomes their training data. You don't own your AI — they do.

## The Solution

ownify is a small, fine-tuned local language model that:

- **Runs on your device** — no cloud required for everyday tasks
- **Carries your knowledge in its weights** — not in config files, not in databases, but baked into the model itself
- **Knows when it needs help** — and can escalate to a larger LLM API when a task exceeds its capability, then returns to local-only operation
- **Stays minimal** — no platform, no server infrastructure, no ecosystem. One model, one person.

## Core Principles

1. **Personal, not shared.** ownify is designed for one person. It is not a platform. It will never be a platform. Your personal AI should belong to you alone.

2. **Behavior lives in the weights.** Instead of system prompts, configuration files, and prompt templates, ownify bakes personality, knowledge, and behavior directly into the model through fine-tuning. The model *is* the configuration.

3. **Smart escalation.** The local model handles most tasks. When it recognizes something beyond its capability, it calls out to a larger model API, gets what it needs, and comes back. You control when and where that happens.

4. **Runs where you are.** Desktop or mobile. Offline-first. Your AI goes where you go.

## The openclaw Adapter

ownify's default behavior comes from **openclaw** — an open-source LoRA adapter modeled on [OpenClaw](https://github.com/openclaw/openclaw) agent behavior patterns:

- **Direct and efficient** — no performative language, gets to the point
- **Genuine helpfulness** — resourceful before asking for help
- **Honest about limitations** — says "I don't know" when it doesn't
- **Learned escalation** — recognizes when to call a larger model
- **Fully transparent** — training data, scripts, and weights are all public

openclaw is the shared foundation. Your **personal adapter** layers on top with your style, knowledge, and preferences. Together: `base model + openclaw + personal adapter = your ownify`.

## How It Works

```
You ask something
    |
    v
ownify (local, small model)
    |
    |--> Can handle it? --> Responds directly (fast, private, free)
    |
    |--> Needs more? --> Escalates to large LLM API --> Returns answer locally
```

## Quickstart

```bash
# Clone
git clone https://github.com/HaraldeRoessler/ownify.git
cd ownify

# Setup Python environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install mlx-lm

# Train the openclaw adapter (~10 min on Apple Silicon)
python src/training/prepare_mlx_data.py --input data/openclaw-v2.jsonl --output data/mlx-openclaw
mlx_lm.lora --config configs/openclaw-mlx-v2.yaml

# Chat with your model
python src/runtime/chat.py
```

## Technical Stack

- **Base model:** [Qwen 3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B) — strong multilingual, good reasoning at small size
- **Training framework:** [MLX](https://github.com/ml-explore/mlx) — Apple's native ML framework, optimized for Apple Silicon unified memory
- **Fine-tuning:** LoRA adapters via [mlx-lm](https://github.com/ml-explore/mlx-lm) (~10 min training, 9.7GB peak memory on M3 Pro)
- **Inference:** MLX native — no GGUF conversion needed, direct adapter loading
- **Escalation:** Planned — tool-call pattern to external API (configurable endpoint)

## Project Structure

```
ownify/
├── README.md
├── ARCHITECTURE.md
├── LICENSE
├── configs/                  # MLX training configurations
│   └── openclaw-mlx-v2.yaml
├── data/                     # Training data (open source)
│   ├── openclaw-v2.jsonl     # 184 core behavior examples
│   ├── openclaw-training.jsonl    # Additional examples
│   ├── training-openclaw-125.jsonl
│   ├── training_data.jsonl        # Multi-turn conversations
│   ├── openclaw-training-data.jsonl
│   └── OPENCLAW_BEHAVIOR.md  # Behavior specification
├── src/
│   ├── runtime/
│   │   └── chat.py           # Interactive chat (MLX)
│   └── training/
│       ├── prepare_mlx_data.py    # Data preparation for MLX
│       ├── train.py               # PyTorch training (legacy)
│       ├── merge_and_export.py    # Model merging
│       └── convert_to_gguf.py     # GGUF conversion
├── adapters/                 # Trained adapters (gitignored)
└── models/                   # Merged models (gitignored)
```

## Training Data

The openclaw behavior is defined by conversation examples — not system prompts. All training data is in the repo:

| File | Examples | Content |
|------|----------|---------|
| `openclaw-v2.jsonl` | 184 | Core behavior: identity, privacy, escalation, technical tasks, multilingual |
| `openclaw-training.jsonl` | 125 | Identity, privacy, local handling, escalation, general helpfulness |
| `training-openclaw-125.jsonl` | 125 | Values, capabilities, technical tasks, conversational, training/customization |
| `training_data.jsonl` | 125 | Multi-turn conversations, edge cases, domain knowledge, daily use |
| `openclaw-training-data.jsonl` | 125 | Multilingual, code review, DevOps, creative writing, error handling |

Total: **684 examples** available for training. Currently training on v2 (184 examples).

## Roadmap

- [x] Base model selection (Qwen 3.5 4B)
- [x] Training data format and collection pipeline
- [x] LoRA fine-tuning pipeline (MLX on Apple Silicon)
- [x] openclaw behavior adapter v2 (184 examples)
- [x] Desktop runtime — interactive chat via MLX
- [ ] Merge all 684 training examples into openclaw v3
- [ ] Escalation runtime — automatic API calls when model flags `<escalate />`
- [ ] Personal adapter training guide
- [ ] Mobile runtime (iOS, Android)
- [ ] LoRA adapter sync between desktop and mobile devices
- [ ] Incremental learning — update the adapter from conversations

## Related Work

The individual pieces exist — no one has combined them.

### Behavior Model
- **[OpenClaw](https://github.com/openclaw/openclaw)** — the agent behavior our openclaw adapter is modeled on. Direct, efficient, genuine helpfulness, SOUL.md personality framework.

### Local LLM Runtimes
[Ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/), [GPT4All](https://www.nomic.ai/gpt4all), [llama.cpp](https://github.com/ggerganov/llama.cpp) — all run models locally but offer no personal fine-tuning, no escalation, no cross-device sync.

### Personal AI Assistants
- **[OpenDAN](https://github.com/fiatrete/OpenDAN-Personal-AI-OS)** (~70% aligned) — personal AI OS with LoRA support, but routing is config-based, not learned.
- **[Khoj](https://github.com/khoj-ai/khoj)** (~60% aligned) — self-hosted personal AI with knowledge base. RAG-focused, no LoRA fine-tuning.

### Learned Routing (Academic)
- **[RouteLLM](https://github.com/lm-sys/RouteLLM)** (ICLR 2025) — trains routers to select between weak/strong models. Cloud-only.
- **["Tell me about yourself"](https://arxiv.org/html/2501.11120v1)** (ICLR 2025) — proves fine-tuned LLMs can introspect on their own capabilities.

### On-Device Fine-Tuning
- **[MLX](https://github.com/ml-explore/mlx)** — Apple's ML framework, native to Apple Silicon. What ownify uses.
- **[Hugging Face PEFT](https://github.com/huggingface/peft)** — LoRA adapters, cross-platform.
- **[MobileFineTuner](https://arxiv.org/html/2512.08211v1)** — LoRA fine-tuning on mobile phones.

### What's missing everywhere
No existing project combines: (1) behavior in weights via LoRA, (2) learned self-escalation, (3) desktop-mobile adapter sync, (4) single-person design. That's ownify.

## Requirements

- Apple Silicon Mac with 16GB+ RAM (for training and inference)
- Python 3.10+
- ~4GB disk for base model download
- Optional: API key for a large LLM provider (for escalation)

Note: Linux/CUDA support possible via PyTorch training path (`src/training/train.py`), but MLX on Apple Silicon is the primary and recommended approach.

## License

MIT License — see [LICENSE](LICENSE)
