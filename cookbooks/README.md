<div align="center">

# 📖 Streaming LLM Inference — Cookbooks

**Practical, end-to-end use cases built on top of the [Streaming LLM Inference](../README.md) server.**

[![Server](https://img.shields.io/badge/requires-streaming--llm--inference-blueviolet)](../README.md)
[![Status: Growing](https://img.shields.io/badge/status-actively%20growing-brightgreen)](#available-cookbooks)

[Overview](#overview) • [Structure](#structure) • [Available Cookbooks](#available-cookbooks) 

</div>

---

## Overview

This folder is a **cookbook of real, end-to-end applications** built against the streaming server in this repository — the kind of thing you'd actually build once the API is running, not just a curl example.

Each cookbook is split into two matching parts:

- **the use case** — the problem itself: what it does, why it's non-trivial, and the constraints it has to work within (small local models, limited context, no server-side session tricks, etc.)
- **the solution** — a working implementation that solves it end to end.

The goal is to make this a growing reference for what's actually possible with a small, closed source private and self-hosted, streaming LLM — not just a toy demo, but code you can point at your own server and adapt.

## Structure

```
cookbooks/
├── end-to-end-usecase/       ← the problem: one folder per use case
│   └── <usecase-name>/
└── end-to-end-solution/       ← the solution: the matching implementation
    └── <usecase-name>/
```

Each use case in `end-to-end-usecase/` has a same-named counterpart in `end-to-end-solution/`, so the two stay easy to pair up as the cookbook grows.

## Available Cookbooks

| Use Case | Description | Status |
|---|---|---|
| [`video_summarizer`](end-to-end-usecase/video_summarizer) | Summarizes a YouTube video from its transcript using map-reduce chunking, tuned for small (1B–3B) local models with short effective reasoning windows (Run Commands : python video_summary.py "<video_url>") | ✅ Available |

> More cookbooks are being added over time. This table — and the rest of this README — will be kept up to date as new use cases and solutions land.

## Running a Cookbook

Every cookbook assumes the [Streaming LLM Inference](../README.md) server is already running (locally or remotely) and reachable — see the root README for setup.

General pattern:

```bash
# 1. Start the inference server via docker (from the repo root)
docker compose up -d 

# 2. Install the cookbook's own dependencies
pip install -r cookbooks/end-to-end-usecase/<usecase-name>/requirements.txt

# 3. Run it
python cookbooks/end-to-end-usecase/<usecase-name>/<script>.py
```

Each cookbook configures itself from the same `.env` the server uses (`LLM_URL`, `API_KEY`, `MODEL_ID`, context limits, ...), so pointing a cookbook at a different server is usually just an environment variable away.

## Contributing

New cookbooks are welcome. When adding one:

1. Add the problem statement and implementation under `end-to-end-usecase/<name>/`.
2. Add the corresponding write-up or reference solution under `end-to-end-solution/<name>/`.
3. Add a row to the [Available Cookbooks](#available-cookbooks) table above.

Keep cookbooks self-contained — their own dependencies, their own short usage instructions — so they stay easy to lift out and reuse on their own.

