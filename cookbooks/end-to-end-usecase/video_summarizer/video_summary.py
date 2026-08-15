"""
YouTube transcript -> local LLM summarizer.

Tuned for small local models (1B-3B). The design assumption is that the model's
*effective* reasoning length is much shorter than its context window, so chunks
are deliberately far smaller than what would fit.

Deps:
    pip install youtube-transcript-api requests python-dotenv

Usage:
    python video_summary.py "<youtube url>" [output.md]
"""

import os
import re
import sys
import json
import math
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def find_env() -> Path | None:
    for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env() -> None:
    env_path = find_env()
    if env_path is None:
        print("[warn] no .env found; using built-in defaults")
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    print(f"[info] loaded {env_path}")


def env_str(key: str, default: str = "") -> str:
    val = os.getenv(key, "").strip()
    return val if val else default


def env_int(key: str, default: int) -> int:
    try:
        return int(env_str(key))
    except ValueError:
        return default


load_env()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_host = env_str("HOST", "127.0.0.1")
if _host in ("0.0.0.0", "::", "[::]"):
    _host = "127.0.0.1"

LLM_URL = env_str("LLM_URL", f"http://{_host}:{env_int('PORT', 8000)}/generate")
API_KEY = env_str("API_KEY")

MAX_CONTEXT_TOKENS = env_int("MAX_CONTEXT_TOKENS", 8192)
CONTEXT_MARGIN_TOKENS = env_int("CONTEXT_MARGIN_TOKENS", 64)
MAX_NEW_TOKENS_LIMIT = env_int("MAX_NEW_TOKENS_LIMIT", 2048)
PROMPT_OVERHEAD_TOKENS = 320

# Absolute ceiling imposed by the context window.
HARD_INPUT_BUDGET = MAX_CONTEXT_TOKENS - CONTEXT_MARGIN_TOKENS - PROMPT_OVERHEAD_TOKENS

# THE IMPORTANT KNOB. Not derived from the context window on purpose: a 1B model
# produces mush well before it runs out of room. Raise toward 4000 for a 7B+.
CHUNK_TOKENS = min(env_int("CHUNK_TOKENS", 1800), HARD_INPUT_BUDGET - 512)

# Per-chunk summaries stay short so they finish inside the budget and so the
# stitched document doesn't become a wall of repetition.
MAP_TOKENS = min(env_int("MAP_TOKENS", 400), MAX_NEW_TOKENS_LIMIT)
GIST_TOKENS = 80
OVERVIEW_TOKENS = min(env_int("OVERVIEW_TOKENS", 400), MAX_NEW_TOKENS_LIMIT)

MAX_CONTINUATIONS = env_int("MAX_CONTINUATIONS", 2)

CHARS_PER_TOKEN = 3.6
CHUNK_CHARS = int(CHUNK_TOKENS * CHARS_PER_TOKEN)
OVERLAP_CHARS = max(200, CHUNK_CHARS // 20)

_tokenizer = None
if env_str("USE_TOKENIZER", "true").lower() == "true":
    try:
        from transformers import AutoTokenizer
        model_id = env_str("MODEL_ID", "")
        if model_id and (not Path(model_id).is_absolute() or Path(model_id).exists()):
            _tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=env_str("TRUST_REMOTE_CODE", "false").lower() == "true",
                token=env_str("HF_TOKEN") or None,
            )
            print(f"[info] tokenizer: {model_id}")
    except Exception as e:
        print(f"[info] tokenizer unavailable ({type(e).__name__}); using char estimate")


def count_tokens(text: str) -> int:
    if _tokenizer is None:
        return math.ceil(len(text) / CHARS_PER_TOKEN)
    return len(_tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

def get_transcript(url: str) -> str:
    """Direct API first — langchain-community is being sunset and only wraps this."""
    from youtube_transcript_api import YouTubeTranscriptApi

    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError(f"couldn't pull a video id out of {url}")
    vid = m.group(1)

    try:                                    # >= 1.0
        return " ".join(s.text for s in YouTubeTranscriptApi().fetch(vid, languages=["en"]))
    except AttributeError:                  # 0.6.x
        parts = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        return " ".join(p["text"] for p in parts)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _extract(data) -> str:
    if isinstance(data, str):
        return data.strip()
    for key in ("text", "response", "content", "generated_text", "output"):
        if key in data:
            return str(data[key]).strip()
    if "choices" in data:
        return data["choices"][0]["message"]["content"].strip()
    return json.dumps(data)


def _parse_sse(raw: str) -> str:
    out = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body in ("", "[DONE]"):
            continue
        try:
            obj = json.loads(body)
            out.append(obj.get("text") or obj.get("token") or obj.get("content") or "")
        except json.JSONDecodeError:
            out.append(body)
    return "".join(out).strip() or raw.strip()


def call_llm(prompt: str, max_new_tokens: int, system: str | None = None) -> str:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
        # No session_id on purpose: SESSION_ENABLED=true would accumulate every
        # chunk into one server-side history and overflow the context.
    }
    r = requests.post(LLM_URL, headers=headers, json=payload, timeout=900)
    r.raise_for_status()
    try:
        return _extract(r.json())
    except json.JSONDecodeError:
        return _parse_sse(r.text)


def looks_truncated(text: str, budget: int) -> bool:
    """No finish_reason from this server, so infer it."""
    if not text:
        return False
    if count_tokens(text) < budget * 0.9:
        return False
    return not text.rstrip().rstrip("*_`").endswith((".", "!", "?", '"', ")", "]"))


CONTINUE_PROMPT = """You were writing and were cut off mid-sentence. Continue \
from exactly where the text stops. Do not repeat any of it, do not restate the \
topic, do not add a heading. Just continue.

TEXT SO FAR (ending):
{tail}"""


def generate(prompt: str, max_new_tokens: int, system: str | None = None) -> str:
    """call_llm, but resume if the budget cut it off mid-sentence."""
    out = call_llm(prompt, max_new_tokens, system)
    for i in range(MAX_CONTINUATIONS):
        if not looks_truncated(out, max_new_tokens):
            break
        print(f"[info]   hit the {max_new_tokens}-token budget; continuing ({i + 1})")
        more = call_llm(CONTINUE_PROMPT.format(tail=out[-600:]), max_new_tokens, system)
        if not more:
            break
        out = out.rstrip() + " " + more.lstrip()
    return out


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS):
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            dot = text.rfind(". ", start + size // 2, end)
            if dot != -1:
                end = dot + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in out if c]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = ("You summarize transcripts accurately. You never invent names, numbers, "
          "products, or claims that are not in the text you were given.")

MAP_PROMPT = """Below is part {n} of {total} of a video transcript.

Summarize ONLY what this excerpt says, as 4-8 bullet points. Keep specific \
names, numbers, tools and claims exactly as stated. If a name is unclear in the \
transcript, write it as it appears rather than guessing. Do not write an \
introduction or conclusion. Do not mention that this is an excerpt.

TRANSCRIPT:
{text}"""

TITLE_PROMPT = """Give a 3-6 word topic label for these notes. Reply with the \
label only — no punctuation, no quotes, no explanation.

NOTES:
{text}"""

GIST_PROMPT = """Summarize these notes in ONE sentence.

NOTES:
{text}"""

OVERVIEW_PROMPT = """Below are one-sentence gists of each part of a single video, in order.

Write a short overview (3-5 sentences) of what the video as a whole is about \
and what it covers. Do not list the parts. Do not use headings.

GISTS:
{text}"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def summarize(transcript: str) -> str:
    chunks = chunk(transcript)
    print(f"[info] transcript {len(transcript):,} chars (~{count_tokens(transcript):,} tokens) "
          f"-> {len(chunks)} chunk(s) of <= {CHUNK_CHARS:,} chars")

    sections = []
    for i, c in enumerate(chunks, 1):
        print(f"[info] chunk {i}/{len(chunks)}")
        body = generate(MAP_PROMPT.format(n=i, total=len(chunks), text=c),
                        MAP_TOKENS, SYSTEM)
        title = call_llm(TITLE_PROMPT.format(text=body[:1200]), 20, SYSTEM)
        title = title.strip().strip('".*#').splitlines()[0][:60] or f"Part {i}"
        gist = call_llm(GIST_PROMPT.format(text=body[:1500]), GIST_TOKENS, SYSTEM)
        sections.append({"title": title, "body": body, "gist": gist.strip()})

    # Overview is built from the short gists, not the full summaries — this is
    # the one merge step, and it gets a few hundred tokens of input instead of
    # several thousand.
    if len(sections) == 1:
        overview = sections[0]["gist"]
    else:
        print("[info] overview")
        gists = "\n".join(f"{i}. {s['gist']}" for i, s in enumerate(sections, 1))
        overview = generate(OVERVIEW_PROMPT.format(text=gists), OVERVIEW_TOKENS, SYSTEM)

    # Assembled in code. The model is never asked to merge thousands of tokens
    # of its own output, which is where it fell apart before.
    parts = ["# Video summary", "", "## Overview", "", overview, "", "## Detailed notes", ""]
    for i, s in enumerate(sections, 1):
        parts += [f"### {i}. {s['title']}", "", s["body"], ""]
    return "\n".join(parts)


if __name__ == "__main__":
    print(f"[config] url={LLM_URL} ctx={MAX_CONTEXT_TOKENS} hard_input_budget={HARD_INPUT_BUDGET} "
          f"chunk_tokens={CHUNK_TOKENS} chunk_chars={CHUNK_CHARS} map_tokens={MAP_TOKENS}")

    if len(sys.argv) < 2:
        sys.exit("usage: python video_summary.py <youtube url> [output.md]")

    result = summarize(get_transcript(sys.argv[1]))
    print("\n" + result)

    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(result, encoding="utf-8")
        print(f"\n[info] wrote {sys.argv[2]}")