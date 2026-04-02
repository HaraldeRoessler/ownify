# ownify — Architecture

## Overview

ownify is three things working together:

1. **A personal LoRA adapter** — your behavior, knowledge, and preferences encoded as model weights
2. **A local inference runtime** — runs the base model + your adapter on your device
3. **An escalation layer** — learned behavior that routes hard tasks to a larger LLM API

```
┌─────────────────────────────────────────────────┐
│                  Your Device                     │
│                                                  │
│  ┌───────────┐   ┌──────────┐   ┌────────────┐  │
│  │ Base Model│ + │ Your LoRA│ = │  ownify    │  │
│  │ (3B-7B)  │   │ Adapter  │   │  (you)     │  │
│  └───────────┘   └──────────┘   └─────┬──────┘  │
│                                       │          │
│                          ┌────────────┴───┐      │
│                          │  Can I handle  │      │
│                          │  this myself?  │      │
│                          └───┬────────┬───┘      │
│                          yes │        │ no       │
│                              ▼        ▼          │
│                        ┌────────┐ ┌────────┐     │
│                        │ Answer │ │Escalate│     │
│                        │locally │ │to API  │──────── → Large LLM API
│                        └────────┘ └────────┘     │       (Claude, etc.)
│                                                  │
└─────────────────────────────────────────────────┘
```

## Components

### 1. Base Model

A small open-source model that provides general language capabilities. ownify doesn't modify the base model — it adds a personal adapter on top.

**Candidates (evaluated by quality-per-parameter):**

| Model | Params | RAM Needed | Mobile? | Notes |
|-------|--------|-----------|---------|-------|
| Qwen 2.5 | 3B | ~4GB | Yes | Strong multilingual, good reasoning |
| Phi-3 Mini | 3.8B | ~4GB | Yes | Microsoft, strong on benchmarks for size |
| Llama 3.2 | 3B | ~4GB | Yes | Meta, broad community support |
| Gemma 2 | 2B | ~3GB | Yes | Google, optimized for on-device |
| Qwen 2.5 | 7B | ~8GB | Desktop only | Best quality at consumer scale |
| Llama 3.1 | 8B | ~8GB | Desktop only | Strongest open model at this size |

**Selection criteria:** The base model must support LoRA fine-tuning, run quantized (Q4/Q5) on CPU, and have a permissive license.

### 2. Personal LoRA Adapter

This is where **you** live. The LoRA adapter encodes:

- **Your communication style** — how you phrase things, formality level, language preferences
- **Your domain knowledge** — topics you work with daily
- **Your preferences** — how you like answers structured, level of detail
- **Escalation behavior** — when to handle locally vs. call for help

**Adapter characteristics:**
- Size: 10-50MB (tiny compared to the base model)
- Portable: one file, moves between devices
- Stackable: can merge multiple training rounds
- Format: safetensors (standard, cross-platform)

**Training data sources:**
```
Your conversations          →  style + preferences
Your documents              →  domain knowledge
Synthetic escalation pairs  →  escalation behavior
Correction feedback         →  continuous improvement
```

### 3. Escalation Layer

The model learns to produce a special tool call when it recognizes a task beyond its capability:

```json
{
  "tool": "escalate",
  "reason": "complex multi-step reasoning required",
  "context": "summarized context for the large model",
  "original_prompt": "user's original question"
}
```

**How escalation is trained:**

We create training pairs where the model learns confidence boundaries:

```
# Training example: handle locally
User: "What time is my meeting tomorrow?"
Assistant: "Your meeting is at 10am with the design team."

# Training example: escalate
User: "Review this 500-line PR and identify security vulnerabilities"
Assistant: <escalate reason="code security review exceeds local capability" />
```

**Escalation signals the model learns:**
- Task complexity (multi-step reasoning, long context)
- Domain mismatch (topic outside trained knowledge)
- Confidence drop (token-level uncertainty via logits)
- Explicit scope (code review, legal analysis, medical info)

**What happens during escalation:**
1. Local model summarizes the conversation context (privacy filter — you control what goes out)
2. Sends to configured large LLM API
3. Receives response
4. Local model can post-process/personalize the response before showing it to you

### 4. Inference Runtime

A thin layer that loads the base model + LoRA adapter and handles the conversation loop.

```
┌──────────────────────────────────────┐
│           ownify runtime             │
│                                      │
│  ┌─────────────┐  ┌──────────────┐   │
│  │ Model Loader │  │ Conversation │   │
│  │ base + LoRA  │  │    Loop      │   │
│  └─────────────┘  └──────┬───────┘   │
│                          │           │
│  ┌─────────────┐  ┌──────┴───────┐   │
│  │  Escalation │  │   Tool       │   │
│  │  Handler    │  │   Router     │   │
│  └─────────────┘  └──────────────┘   │
│                                      │
│  ┌─────────────┐                     │
│  │  Adapter    │ ← sync             │
│  │  Storage    │                     │
│  └─────────────┘                     │
└──────────────────────────────────────┘
```

**Runtime options:**
- **Desktop:** llama.cpp via Python bindings (llama-cpp-python) or Ollama as backend
- **Mobile (future):** llama.cpp compiled for iOS/Android, or MLC LLM

### 5. Adapter Sync (Roadmap)

Your LoRA adapter is a single small file. Syncing between devices:

```
Desktop                          Mobile
  │                                │
  │  ┌──────────────────────┐      │
  └──│  Sync Transport      │──────┘
     │  (iCloud / file sync) │
     └──────────────────────┘
            │
      adapter.safetensors
          (10-50MB)
```

**Sync strategy:**
- Adapter is versioned (timestamps or sequence numbers)
- Latest adapter wins (no merge — you're one person, one device active at a time)
- Transport is pluggable: iCloud Drive, Syncthing, USB, manual copy
- No server required

## Directory Structure

```
ownify/
├── README.md
├── ARCHITECTURE.md
├── LICENSE
├── src/
│   ├── runtime/          # Inference engine, conversation loop
│   │   ├── engine.py     # Model loading, generation
│   │   ├── conversation.py  # Chat loop, history
│   │   └── escalation.py    # Escalation handler + API client
│   ├── training/         # LoRA fine-tuning pipeline
│   │   ├── prepare.py    # Training data preparation
│   │   ├── train.py      # LoRA training script
│   │   └── evaluate.py   # Adapter quality evaluation
│   └── sync/             # Adapter sync (future)
│       └── sync.py
├── adapters/             # Your personal LoRA adapters (gitignored)
├── data/                 # Training data (gitignored)
└── config.yaml           # Single config: API endpoint + key (only external config)
```

## The One Config File

Everything lives in the weights except what physically can't — API credentials:

```yaml
# config.yaml — the only configuration file
escalation:
  provider: anthropic          # or openai, local, none
  endpoint: https://api.anthropic.com
  api_key: ${OWNIFY_API_KEY}   # from environment variable
  model: claude-sonnet-4-20250514

base_model:
  name: qwen2.5-3b-instruct
  quantization: Q5_K_M

adapter:
  path: ./adapters/current.safetensors
```

That's it. No system prompts. No persona files. No RAG config. No tool definitions. The model knows who it is.

## Training Pipeline

### Phase 1: Behavior Fine-Tuning

```
Your chat exports / writing samples
         │
         ▼
   Data Preparation
   (format as instruction pairs)
         │
         ▼
   LoRA Fine-Tuning
   (PEFT + base model, ~1-4 hours on consumer GPU)
         │
         ▼
   adapter-v1.safetensors
```

### Phase 2: Escalation Training

```
Synthetic escalation examples
(tasks the small model can/cannot handle)
         │
         ▼
   LoRA Fine-Tuning
   (extend adapter with escalation behavior)
         │
         ▼
   adapter-v2.safetensors
```

### Phase 3: Continuous Learning (Roadmap)

```
Daily conversations + corrections
         │
         ▼
   Incremental LoRA update
   (periodic, maybe weekly)
         │
         ▼
   adapter-v3, v4, v5...
```

## Privacy Model

| Data | Where it lives | Who sees it |
|------|---------------|-------------|
| Your conversations | Your device only | You |
| Your LoRA adapter | Your device(s) only | You |
| Base model | Downloaded once, public | Everyone |
| Escalated queries | Sent to API provider | API provider (your choice) |
| Training data | Your device only | You |

The escalation layer includes a **context summarizer** — before sending anything to an external API, the local model strips and summarizes, so you control what leaves your device.
