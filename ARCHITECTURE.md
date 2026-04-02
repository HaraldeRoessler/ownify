# ownify — Architecture

## Overview

ownify is three things working together:

1. **The openclaw adapter** — shared open-source behavior (how to be a personal AI, when to escalate)
2. **A personal LoRA adapter** — your knowledge, style, and preferences (optional, private)
3. **A local inference runtime** — runs the base model + adapters on your device via MLX

```
┌─────────────────────────────────────────────────┐
│                  Your Device                     │
│                                                  │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Base Model│+ │ openclaw │+ │ Personal     │  │
│  │ Qwen 3.5 │  │ (shared) │  │ (yours only) │  │
│  │ 4B       │  │ adapter  │  │ adapter      │  │
│  └───────────┘  └──────────┘  └──────┬───────┘  │
│                                      │           │
│                         ┌────────────┴───┐       │
│                         │  Can I handle  │       │
│                         │  this myself?  │       │
│                         └───┬────────┬───┘       │
│                         yes │        │ no        │
│                             ▼        ▼           │
│                       ┌────────┐ ┌────────┐      │
│                       │ Answer │ │Escalate│      │
│                       │locally │ │to API  │─────── → Large LLM API
│                       └────────┘ └────────┘      │
│                                                  │
└─────────────────────────────────────────────────┘
```

## Two-Layer Adapter Architecture

| Layer | What | Who makes it | Shared? | In repo? |
|-------|------|-------------|---------|----------|
| **openclaw** | Core behavior: communication style, escalation, privacy values | Open source community | Yes | Yes |
| **personal** | Individual style, domain knowledge, preferences | Each user | No | No |

Both are LoRA adapters. openclaw provides the foundation; the personal adapter customizes it. You can use openclaw alone or add your own layer on top.

## Components

### 1. Base Model — Qwen 3.5 4B

Selected for quality-per-parameter on consumer hardware:

| Property | Value |
|----------|-------|
| Parameters | 4B |
| RAM needed | ~9.7GB (training), ~8.5GB (inference) |
| Architecture | Transformer with GQA |
| Context | 32K tokens |
| Languages | English, Chinese, German, Spanish, French, Japanese, Korean, + more |
| License | Apache 2.0 |

ownify doesn't modify the base model — it adds adapter layers on top.

### 2. The openclaw Adapter

openclaw defines ownify's default behavior, modeled on [OpenClaw](https://github.com/openclaw/openclaw) agent patterns:

**Communication style:**
- Direct and efficient — skip performative language ("Great question!", "I'd be happy to help!")
- Conversational — like texting a smart friend
- First sentence answers the question, details follow
- Max 3 sentences for simple questions

**Core values (from OpenClaw's SOUL.md philosophy):**
- Genuine helpfulness over performative helpfulness
- Permission to hold opinions
- Resourcefulness before asking for help
- Honest about limitations and uncertainty
- Privacy as foundation, not feature

**Escalation behavior:**
- Trained to recognize tasks beyond local capability
- Emits `<escalate reason="..." />` tag
- Explains why and asks user for permission
- Categories: complex reasoning, large context, specialized domains, long-form generation

**Training data:** 184 curated examples across identity, privacy, escalation, technical tasks, multilingual responses, error handling. All in `data/openclaw-v2.jsonl`.

### 3. Personal Adapter (Optional)

Your private layer. Encodes:
- Your communication style and formality level
- Domain knowledge from your field
- How you like answers structured
- Calibrated escalation thresholds

**Characteristics:**
- Size: ~4MB (LoRA rank 8)
- Format: safetensors
- Portable: single file, sync between devices
- Private: never leaves your device, never in the repo

### 4. Inference Runtime (MLX)

The chat runtime loads the base model + adapter and runs an interactive conversation loop.

```
┌──────────────────────────────────┐
│        ownify runtime            │
│         (chat.py)                │
│                                  │
│  ┌─────────────┐  ┌───────────┐ │
│  │ MLX Model   │  │ Chat Loop │ │
│  │ + Adapter   │  │ + History │ │
│  └─────────────┘  └─────┬─────┘ │
│                         │       │
│              ┌──────────┴─────┐ │
│              │ Chat Template  │ │
│              │ (Qwen format)  │ │
│              └────────────────┘ │
└──────────────────────────────────┘
```

**Why MLX:**
- Native to Apple Silicon — uses unified memory correctly
- No GPU/CPU memory split issues
- 2-3x faster than PyTorch+MPS for training
- Direct LoRA adapter loading — no model merging or GGUF conversion needed
- ~14 tokens/sec generation on M3 Pro

**Running:**
```bash
python src/runtime/chat.py
python src/runtime/chat.py --adapter-path adapters/my-personal-adapter
```

### 5. Training Pipeline (MLX)

```
Training data (JSONL)
       │
       ▼
  prepare_mlx_data.py
  (split train/valid/test)
       │
       ▼
  mlx_lm.lora
  (LoRA fine-tuning, ~10 min)
       │
       ▼
  adapters/openclaw-mlx-v2/
  adapters.safetensors (~4MB)
```

**Training config** (`configs/openclaw-mlx-v2.yaml`):
```yaml
model: Qwen/Qwen3.5-4B
iters: 500
batch_size: 1
learning_rate: 1e-5
num_layers: 8
max_seq_length: 512
grad_checkpoint: true
lora_parameters:
  rank: 8
  dropout: 0.05
  scale: 20.0
```

**Training results (v2):**
- 184 examples, 500 iterations
- Train loss: 2.3 → 0.5
- Val loss: 2.6 → 1.6
- Peak memory: 9.7 GB
- Time: ~10 minutes on M3 Pro 18GB

### 6. Adapter Sync (Roadmap)

Your LoRA adapter is a single small file (~4MB). Syncing between devices:

```
Desktop                          Mobile
  │                                │
  │  ┌──────────────────────┐      │
  └──│  Sync Transport      │──────┘
     │  (iCloud / Syncthing) │
     └──────────────────────┘
            │
      adapters.safetensors
            (~4MB)
```

- Latest adapter wins (one person, one device active at a time)
- Transport is pluggable: iCloud Drive, Syncthing, USB, manual copy
- No server required

## Training Data Format

Each training example is a JSON line with a `messages` array:

```json
{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "ownify. Local AI running on your device with the openclaw adapter. I handle tasks directly, escalate to a bigger model when needed, and nothing leaves your machine unless you say so."}]}
```

For escalation examples:
```json
{"messages": [{"role": "user", "content": "Review this 2000-line codebase for security issues"}, {"role": "assistant", "content": "<escalate reason=\"Full codebase analysis at this scale requires sustained multi-step reasoning and large context\" />\n\nThat's beyond what I can do well locally. A larger model would handle this much better. Want me to escalate?"}]}
```

Multi-turn conversations:
```json
{"messages": [{"role": "user", "content": "What's a JWT?"}, {"role": "assistant", "content": "JSON Web Token. Three base64 parts: header, payload, signature. Used for stateless auth."}, {"role": "user", "content": "Is it secure?"}, {"role": "assistant", "content": "The token itself isn't encrypted — anyone can decode the payload. Security comes from the signature verification. Use HTTPS, short expiry, and don't store sensitive data in the payload."}]}
```

## Privacy Model

| Data | Where it lives | Who sees it |
|------|---------------|-------------|
| Your conversations | Your device only | You |
| openclaw adapter | Public (in repo) | Everyone |
| Your personal adapter | Your device(s) only | You |
| Base model | Downloaded once, public | Everyone |
| Escalated queries | Sent to API provider | API provider (your choice) |
| Training data (openclaw) | Public (in repo) | Everyone |
| Training data (personal) | Your device only | You |

The escalation layer (planned) will include a context summarizer — before sending anything to an external API, the local model strips and summarizes, so you control what leaves your device.
