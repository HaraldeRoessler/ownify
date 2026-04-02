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

The escalation decision is learned behavior — trained into the model, not hard-coded rules. The model learns to recognize its own limits.

## Technical Approach

- **Base model:** Small open-source model (3B-7B parameters) that runs on consumer hardware
- **Fine-tuning:** LoRA adapters trained on your behavior, preferences, and domain knowledge
- **Inference:** Local execution via Ollama, llama.cpp, or similar runtime
- **Escalation:** Tool-call pattern to external API (configurable endpoint, your API key)

## Roadmap

- [ ] Base model selection and benchmarking on consumer hardware
- [ ] Training data format and collection pipeline
- [ ] LoRA fine-tuning pipeline for personal behavior
- [ ] Escalation training — teach the model when to call for help
- [ ] Desktop runtime (macOS, Linux, Windows)
- [ ] Mobile runtime (iOS, Android)
- [ ] LoRA adapter sync between desktop and mobile devices
- [ ] Incremental learning — update the adapter as you use it

## Requirements

- A computer with 8GB+ RAM (for 3B model) or 16GB+ RAM (for 7B model)
- No GPU required (runs on CPU, faster with Apple Silicon / CUDA)
- Optional: API key for a large LLM provider (for escalation)

## License

MIT License — see [LICENSE](LICENSE)
