"""
Central configuration for the streaming inference server.

Every tunable in this project is an environment variable. They are read once
at import time from the process environment, with an optional `.env` file
layered underneath (real environment variables always win, so container and
CI overrides behave the way operators expect).

Nothing else in the codebase calls ``os.getenv``: if a value is worth
changing, it belongs here and in ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

# ENV_FILE lets you keep several profiles side by side (.env.dev, .env.prod).
load_dotenv(os.getenv("ENV_FILE", ".env"), override=False)


# ---------------------------------------------------------------------------
# Typed readers
# ---------------------------------------------------------------------------
def _raw(key: str) -> Optional[str]:
    """Return the stripped value of `key`, or None if unset or blank.

    Blank is treated as unset so that commenting a value out in `.env` and
    leaving `KEY=` behind both mean "fall back to the default".
    """
    value = os.getenv(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def env_str(key: str, default: str) -> str:
    """Read a string setting, falling back to `default` when unset."""
    return _raw(key) or default


def env_opt_str(key: str) -> Optional[str]:
    """Read an optional string setting; None means "not configured"."""
    return _raw(key)


def env_int(key: str, default: int) -> int:
    """Read an integer setting, raising a clear error on malformed input."""
    value = _raw(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc


def env_opt_int(key: str) -> Optional[int]:
    """Read an optional integer setting; None means "not configured"."""
    value = _raw(key)
    return None if value is None else env_int(key, 0)


def env_float(key: str, default: float) -> float:
    """Read a float setting, raising a clear error on malformed input."""
    value = _raw(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {value!r}") from exc


def env_opt_float(key: str) -> Optional[float]:
    """Read an optional float setting.

    None is meaningful for sampling knobs: it tells the engine to inherit the
    value the model ships in its own ``generation_config.json`` rather than
    imposing a project-wide default.
    """
    value = _raw(key)
    return None if value is None else env_float(key, 0.0)


def env_bool(key: str, default: bool) -> bool:
    """Read a boolean setting. Accepts 1/true/yes/on in any casing."""
    value = _raw(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_list(
    key: str, default: Optional[List[str]] = None, unescape: bool = False
) -> List[str]:
    """Read a comma-separated list setting into a list of trimmed strings.

    With `unescape`, backslash sequences such as ``\\n`` are decoded into the
    characters they denote. Dotenv files store them literally, so stop strings
    like ``\\nUser:`` would otherwise never match anything.
    """
    value = _raw(key)
    if value is None:
        return list(default or [])
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not unescape:
        return items
    return [item.encode("utf-8").decode("unicode_escape") for item in items]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of every environment-driven setting.

    Built once by :func:`get_settings` and passed explicitly into the engine
    and the app, so no module reaches into global state on its own.
    """

    # -- model ---------------------------------------------------------------
    model_id: str
    hf_token: Optional[str]
    trust_remote_code: bool
    dtype: str
    device_map: str
    attn_implementation: Optional[str]

    # -- context handling ----------------------------------------------------
    max_context_tokens: int
    context_margin_tokens: int
    default_system_prompt: Optional[str]
    fallback_stop_strings: List[str]

    # -- generation defaults -------------------------------------------------
    max_new_tokens: int
    max_new_tokens_limit: int
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    repetition_penalty: Optional[float]
    seed: Optional[int]

    # -- sessions ------------------------------------------------------------
    session_enabled: bool
    session_ttl_seconds: int
    session_max_turns: int
    session_max_count: int

    # -- server --------------------------------------------------------------
    host: str
    port: int
    max_concurrency: int
    generation_join_timeout: float
    api_key: Optional[str]
    cors_origins: List[str]

    # -- logging -------------------------------------------------------------
    log_level: str
    log_prompts: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Construct a Settings instance from the current environment."""
        return cls(
            model_id=env_str("MODEL_ID", "meta-llama/Llama-3.2-1B"),
            hf_token=env_opt_str("HF_TOKEN"),
            trust_remote_code=env_bool("TRUST_REMOTE_CODE", False),
            dtype=env_str("DTYPE", "auto"),
            device_map=env_str("DEVICE_MAP", "auto"),
            attn_implementation=env_opt_str("ATTN_IMPLEMENTATION"),
            max_context_tokens=env_int("MAX_CONTEXT_TOKENS", 8192),
            context_margin_tokens=env_int("CONTEXT_MARGIN_TOKENS", 64),
            default_system_prompt=env_opt_str("DEFAULT_SYSTEM_PROMPT"),
            fallback_stop_strings=env_list(
                "FALLBACK_STOP_STRINGS", ["\nUser:", "\nAssistant:"], unescape=True
            ),
            max_new_tokens=env_int("MAX_NEW_TOKENS", 256),
            max_new_tokens_limit=env_int("MAX_NEW_TOKENS_LIMIT", 2048),
            temperature=env_opt_float("TEMPERATURE"),
            top_p=env_opt_float("TOP_P"),
            top_k=env_opt_int("TOP_K"),
            repetition_penalty=env_opt_float("REPETITION_PENALTY"),
            seed=env_opt_int("SEED"),
            session_enabled=env_bool("SESSION_ENABLED", True),
            session_ttl_seconds=env_int("SESSION_TTL_SECONDS", 1800),
            session_max_turns=env_int("SESSION_MAX_TURNS", 40),
            session_max_count=env_int("SESSION_MAX_COUNT", 1000),
            host=env_str("HOST", "0.0.0.0"),
            port=env_int("PORT", 8000),
            max_concurrency=env_int("MAX_CONCURRENCY", 1),
            generation_join_timeout=env_float("GENERATION_JOIN_TIMEOUT", 15.0),
            api_key=env_opt_str("API_KEY"),
            cors_origins=env_list("CORS_ORIGINS", ["*"]),
            log_level=env_str("LOG_LEVEL", "INFO").upper(),
            log_prompts=env_bool("LOG_PROMPTS", False),
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the process-wide Settings singleton, building it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings