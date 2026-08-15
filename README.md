<div align="center">

# ⚡ Streaming LLM Inference

**A production-grade FastAPI server for streaming any HuggingFace causal language model — token by token, over Server-Sent Events.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#requirements)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#)
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](#run-with-docker)

[Overview](#overview) • [Features](#features) • [Quick Start](#quick-start) • [Docker](#run-with-docker) • [API](#api-reference) • [Configuration](#configuration) • [Architecture](#architecture)

</div>

---

## Overview

**Streaming LLM Inference** turns any HuggingFace causal language model into a low-latency streaming API. It's built for the moment you outgrow a notebook script and need a real service: RAG backends, chatbots, voice assistants, or anything that needs tokens on screen the instant the model produces them.

It's **model-agnostic by design** — point it at an instruct model with a chat template (Llama, Qwen, Mistral, Gemma, Phi-3, ...) or a raw base model for pure completion, and it adapts automatically. Every tunable — model, context size, sampling, auth, concurrency — comes from the environment, so the same image runs identically in dev and production.

## Features

| | |
|---|---|
| ⚡ **Real-time streaming** | Token-by-token output over Server-Sent Events, plus a buffered mode for non-streaming clients |
| 🧠 **Model-agnostic** | Works with any HuggingFace causal LM — chat-template instruct models *and* raw base models |
| 💬 **Server-side sessions** | Optional conversation memory keyed by `session_id`, with TTL and size-capped eviction |
| 🔐 **Optional API key auth** | Bearer-token protection, opt-in via a single env var |
| 🪟 **Automatic context fitting** | Long conversations are trimmed to fit the model's real context window, not a guessed one |
| 🛑 **Client-disconnect interruption** | Generation stops immediately when a client hangs up — no wasted GPU cycles |
| 🧵 **Non-blocking generation** | Model inference runs off the event loop; a semaphore serializes access to the single model instance |
| 📊 **Built-in usage metrics** | Prompt/completion tokens, time-to-first-token, and tokens/sec on every response |
| 🖥️ **GPU auto-detection** | `device_map="auto"` places the model correctly, GPU or CPU, with no manual wiring |
| 💻 **Batteries-included CLI client** | A zero-dependency terminal chat client (`chat.py`) to test the server end to end |
| 🐳 **Production Docker image** | Non-root user, health checks, CUDA base image, and NVIDIA Container Toolkit support |

## Quick Start

### Requirements

```
Python 3.9+
```

### Install

```bash
pip install -r requirements.txt
```

### Configure

All configuration lives in environment variables. Create a `.env` file (or copy `.env.example` if present):

```bash
MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
HF_TOKEN=your_huggingface_token   # only needed for gated models
PORT=8000
```

See [Configuration](#configuration) for every available setting.

### Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

```
Server running at http://localhost:8000
```

### Talk to it

Use the included CLI client for an instant interactive session:

```bash
python chat.py
```

```bash
python chat.py --url http://gpu-box:8000     # point at a remote server
python chat.py --stateless                   # client keeps history instead of the server
python chat.py --raw                         # completion mode for base models
python chat.py --system "You are a terse assistant."
```

Press `Ctrl-C` mid-response to interrupt generation without leaving the chat — a live test of the server's disconnect handling.

## Run with Docker

### Build and start

```bash
docker compose up -d --build
```

### Stop

```bash
docker compose down
```

The container runs as a non-root user, mounts `./models` for a persistent HuggingFace weight cache, and exposes a startup health check that waits for the model to finish loading before traffic is accepted.

### GPU support

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi
```

`docker-compose.yml` already requests all available GPUs — no extra flags needed.

## API Reference

| Method | Path | Description |
|--------|--------------------------|-------------------------------------------|
| `GET` | `/health` | Liveness check plus the model's resolved runtime limits |
| `POST` | `/generate` | Generate a completion — streaming (SSE) or buffered |
| `DELETE` | `/sessions/{id}` | Forget a server-held conversation |

### `POST /generate`

Send either a `messages` array (chat) or a raw `prompt` (base-model completion) — not both. Add a `session_id` to let the server hold conversation history for you.

```json
{
  "messages": [
    { "role": "system", "content": "You are helpful." },
    { "role": "user", "content": "Explain attention in one sentence." }
  ],
  "session_id": "user-123",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "stream": true
}
```

**Streaming response (`text/event-stream`)**

```
data: {"token": "Attention"}

data: {"token": " lets"}

data: {"token": " a"}

...

data: {"done": true, "usage": {"prompt_tokens": 42, "completion_tokens": 96, "tokens_per_second": 38.4, "finish_reason": "stop"}}

data: [DONE]
```

**Buffered response** (`"stream": false`) returns the same content as a single JSON payload instead of a token stream.

### curl examples

**Basic streaming request:**

```bash
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

**With full parameters and API key auth:**

```bash
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
        "messages": [
          {"role":"system","content":"You are helpful."},
          {"role":"user","content":"Explain transformers."}
        ],
        "max_new_tokens": 100,
        "temperature": 0.5
      }'
```

**Forget a session:**

```bash
curl -X DELETE http://localhost:8000/sessions/user-123 \
  -H "Authorization: Bearer $API_KEY"
```

## Architecture

```
Client Request
      │
      ▼
FastAPI Endpoint  ──► optional API-key auth
      │
      ▼
Session Store  ──► merge stored history + new turn
      │
      ▼
Prompt Builder  ──► chat template render + context-window trimming
      │
      ▼
Tokenizer  ──► Model.generate() on a background thread
      │
      ▼
TextIteratorStreamer  ──► tokens as they are produced
      │
      ▼
SSE Output  ──► data: {"token": "..."}
      │
      ▼
Client  ──► renders tokens in real time
```

### Key components

- **`TextIteratorStreamer`** — streams tokens as the model produces them, instead of waiting for the full generation.
- **`StopOnFlag` (StoppingCriteria)** — a shared flag between the request coroutine and the generation thread, allowing mid-generation interruption.
- **Threaded generation** — model inference is dispatched to a worker thread so the async event loop never blocks.
- **`SessionStore`** — an in-memory, TTL-and-size-capped conversation store; swap in Redis behind the same interface for multi-replica deployments.

### Client disconnect handling

If a client disconnects mid-stream, the server detects it, halts generation immediately, and logs:

```
client disconnected → stopping generation
```

No wasted compute on a response nobody is listening to.

## Configuration

Every setting is read once from the environment, with `.env` layered underneath real environment variables (which always win). Nothing outside `config.py` calls `os.getenv` directly.

<details>
<summary><strong>Model</strong></summary>

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `meta-llama/Llama-3.2-1B` | Any HuggingFace causal LM repo or local path |
| `HF_TOKEN` | — | Required for gated/private models |
| `TRUST_REMOTE_CODE` | `false` | Allow custom modeling code from the repo |
| `DTYPE` | `auto` | Load precision (`auto`, `bfloat16`, `float16`, ...) |
| `DEVICE_MAP` | `auto` | HF device placement strategy |
| `ATTN_IMPLEMENTATION` | — | e.g. `flash_attention_2`, if installed |

</details>

<details>
<summary><strong>Context handling</strong></summary>

| Variable | Default | Description |
|---|---|---|
| `MAX_CONTEXT_TOKENS` | `8192` | Ceiling for prompt + history |
| `CONTEXT_MARGIN_TOKENS` | `64` | Safety margin reserved below the ceiling |
| `DEFAULT_SYSTEM_PROMPT` | — | Injected only when the caller doesn't supply one |
| `FALLBACK_STOP_STRINGS` | `\nUser:,\nAssistant:` | Used when a model has no known end-of-turn token |

</details>

<details>
<summary><strong>Generation defaults</strong></summary>

| Variable | Default | Description |
|---|---|---|
| `MAX_NEW_TOKENS` | `256` | Default response length |
| `MAX_NEW_TOKENS_LIMIT` | `2048` | Hard server-side cap, regardless of client request |
| `TEMPERATURE` / `TOP_P` / `TOP_K` / `REPETITION_PENALTY` | — | Unset inherits the checkpoint's own `generation_config.json` |
| `SEED` | — | Deterministic sampling, when set |

</details>

<details>
<summary><strong>Sessions</strong></summary>

| Variable | Default | Description |
|---|---|---|
| `SESSION_ENABLED` | `true` | Enable server-held conversation memory |
| `SESSION_TTL_SECONDS` | `1800` | Idle timeout before a session is dropped |
| `SESSION_MAX_TURNS` | `40` | Turns retained per session |
| `SESSION_MAX_COUNT` | `1000` | Total sessions kept in memory before LRU eviction |

</details>

<details>
<summary><strong>Server</strong></summary>

| Variable | Default | Description |
|---|---|---|
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address |
| `MAX_CONCURRENCY` | `1` | Concurrent generations (one model instance serializes access) |
| `API_KEY` | — | Enables bearer-token auth on `/generate` and `/sessions/*` when set |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `LOG_LEVEL` / `LOG_PROMPTS` | `INFO` / `false` | Logging verbosity and raw-prompt debug logging |

</details>

## Troubleshooting

| Symptom | Fix |
|---|---|
| GPU not used | Check `torch.cuda.is_available()`; confirm `DEVICE_MAP=auto` |
| Slow inference | Use a GPU, a smaller model, or enable quantization |
| No streaming in curl | Make sure `-N` is passed to disable curl's output buffering |
| `AutoModelForCausalLM requires the PyTorch library` | Your `torch` build is older than `transformers` expects — see the pinned floor in `requirements.txt` |

## Model Licensing

This server is model-agnostic, but weights you load are not automatically covered by this repository's license. Meta Llama models, for example, are licensed separately by Meta — review the applicable model license before commercial or production use.

## License

Released under the [MIT License](LICENSE).

---

<div align="right">

Author  : Rahul Dangi

Type    : Local LLM Inference Server

Mode    : Streaming (SSE - Server Sent Events)

</div>
