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

## Related Work

The individual pieces exist — no one has combined them.

### Local LLM Runtimes
[Ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/), [GPT4All](https://www.nomic.ai/gpt4all), [llama.cpp](https://github.com/ggerganov/llama.cpp) — all run models locally but offer no personal fine-tuning, no escalation, no cross-device sync.

### Personal AI Assistants
- **[OpenDAN](https://github.com/fiatrete/OpenDAN-Personal-AI-OS)** (~70% aligned) — personal AI OS with LoRA support, but routing is config-based, not learned. No device sync.
- **[Khoj](https://github.com/khoj-ai/khoj)** (~60% aligned) — self-hosted personal AI with knowledge base. RAG-focused, no LoRA fine-tuning, no escalation.
- **[AnythingLLM](https://anythingllm.com/)** (~40% aligned) — local AI with document management. No personalization or escalation.

### Learned Routing (Academic)
- **[RouteLLM](https://github.com/lm-sys/RouteLLM)** (ICLR 2025) — trains routers to select between weak/strong models. 2x cost reduction. But cloud-only, not local-to-cloud.
- **["Tell me about yourself"](https://arxiv.org/html/2501.11120v1)** (ICLR 2025) — proves fine-tuned LLMs can introspect on their own capabilities. Directly supports our core assumption.
- **[Confidence Token Routing](https://arxiv.org/html/2410.13284v3)** — uses logit confidence scores for dynamic routing decisions.

### On-Device Fine-Tuning
- **[Hugging Face PEFT](https://github.com/huggingface/peft)** — LoRA adapters: 6-50MB vs multi-GB models, trainable on consumer hardware.
- **[Unsloth](https://unsloth.ai/)** — fast LoRA fine-tuning optimized for local hardware.
- **[MobileFineTuner](https://arxiv.org/html/2512.08211v1)** — demonstrates LoRA fine-tuning directly on mobile phones.

### What's missing everywhere
No existing project combines: (1) behavior in weights via LoRA, (2) learned self-escalation, (3) desktop-mobile adapter sync, (4) single-person design. That's ownify.

## Requirements

- A computer with 8GB+ RAM (for 3B model) or 16GB+ RAM (for 7B model)
- No GPU required (runs on CPU, faster with Apple Silicon / CUDA)
- Optional: API key for a large LLM provider (for escalation)

## License

MIT License — see [LICENSE](LICENSE)
