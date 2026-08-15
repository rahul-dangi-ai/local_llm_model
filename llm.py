"""
Inference engine: model loading, prompt construction and token streaming.

This module knows about HuggingFace and nothing about HTTP. It is written to
be model-agnostic across the open-weight ecosystem:

* **Instruct / chat models** that ship a chat template (Llama-3.x-Instruct,
  Qwen2.5-Instruct, Mistral-Instruct, Gemma-it, Phi-3, ...) are rendered with
  ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``.
* **Base models with no chat template** (``meta-llama/Llama-3.2-1B``,
  ``mistralai/Mistral-7B-v0.1``, ...) fall back to a plain transcript layout,
  and can also be driven directly with a raw ``prompt`` for pure completion —
  the ``pipe("The key to life is")`` style shown on the Llama 3.2 model card.

Sampling defaults that are left unset in ``.env`` are inherited from the
checkpoint's own ``generation_config.json``, so each model runs with the
settings its authors shipped rather than one hardcoded set.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from config import Settings

logger = logging.getLogger("llm")

# End-of-turn tokens used by the major open-weight families. Only the ones
# that genuinely exist in the loaded tokenizer's vocabulary are applied, so
# this list is safe to carry across every model.
KNOWN_EOT_TOKENS = (
    "<|eot_id|>",      # Llama 3.x instruct
    "<|eom_id|>",      # Llama 3.1 tool-call turns
    "<|im_end|>",      # Qwen / ChatML
    "<end_of_turn>",   # Gemma
    "<|end|>",         # Phi-3
    "<|endoftext|>",
)

# A tokenizer that reports no real limit (e.g. VERY_LARGE_INTEGER) should not
# be trusted to size the context window.
_IMPLAUSIBLE_CONTEXT = 1_000_000


@dataclass
class GenerationParams:
    """Per-request generation settings, already validated by the API layer.

    ``None`` on any sampling field means "not specified by the caller", which
    the engine resolves against `.env` and then the model's own
    ``generation_config``.
    """

    max_new_tokens: int
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


@dataclass
class Usage:
    """Token accounting and timing for one completed generation."""

    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float
    time_to_first_token: Optional[float]
    finish_reason: str

    @property
    def tokens_per_second(self) -> float:
        """Decode throughput, or 0.0 when nothing was generated."""
        if self.duration_seconds <= 0 or self.completion_tokens <= 0:
            return 0.0
        return round(self.completion_tokens / self.duration_seconds, 2)

    def as_dict(self) -> Dict[str, Any]:
        """Serialise the usage record for the SSE `done` event and logs."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "duration_seconds": round(self.duration_seconds, 3),
            "time_to_first_token": (
                round(self.time_to_first_token, 3)
                if self.time_to_first_token is not None
                else None
            ),
            "tokens_per_second": self.tokens_per_second,
            "finish_reason": self.finish_reason,
        }


class StopOnFlag(StoppingCriteria):
    """Stopping criterion that aborts ``generate`` when a shared flag flips.

    The flag is a one-key dict rather than a bool so the generation thread and
    the request coroutine share the same mutable object. This is what makes
    client-disconnect interruption possible mid-generation.
    """

    def __init__(self, stop_flag: Dict[str, bool]) -> None:
        self.stop_flag = stop_flag

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """Return True once the request has asked generation to stop."""
        return self.stop_flag["stop"]


class LLMEngine:
    """Owns the tokenizer and model and turns messages into streamed tokens."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tokenizer = None
        self.model = None
        self.has_chat_template = False
        self.max_context = settings.max_context_tokens
        self.eos_token_ids: List[int] = []
        self.pad_token_id: int = 0

    # -- loading -------------------------------------------------------------
    def load(self) -> None:
        """Load tokenizer and weights and derive per-model runtime limits.

        Called once at startup so the first request does not pay for it, and
        so a bad ``MODEL_ID`` fails the container immediately rather than on
        the first user request.
        """
        settings = self.settings
        self._check_torch_backend()
        self._check_local_path(settings.model_id)
        started = time.perf_counter()
        logger.info("Loading model %s", settings.model_id)

        auth: Dict[str, Any] = {"token": settings.hf_token} if settings.hf_token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id,
            trust_remote_code=settings.trust_remote_code,
            **auth,
        )
        # Only invent a pad token when the checkpoint genuinely lacks one.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only generation pads on the left.
        self.tokenizer.padding_side = "left"
        # This post-processing step targets WordPiece tokenizers and is
        # destructive for the BPE tokenizers used across the Llama, Qwen and
        # Mistral families — it strips spaces before punctuation, which is
        # visible as corrupted spacing in a token stream.
        self.tokenizer.clean_up_tokenization_spaces = False

        self.model = self._load_weights(auth)
        self.model.eval()

        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        self.max_context = self._resolve_max_context()
        self.eos_token_ids = self._collect_eos_token_ids()
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.eos_token_ids[0] if self.eos_token_ids else 0)
        )

        if settings.seed is not None:
            torch.manual_seed(settings.seed)

        if not self.has_chat_template:
            logger.warning(
                "%s has no chat template, so it is a BASE (completion) checkpoint, not an "
                "instruction-tuned one. It will continue text rather than answer questions, "
                "and `messages` requests fall back to a plain transcript. For assistant "
                "behaviour use an -Instruct checkpoint; for completion use the `prompt` field.",
                settings.model_id,
            )

        logger.info(
            "Model ready in %.1fs | device=%s dtype=%s chat_template=%s "
            "context=%d eos_ids=%s",
            time.perf_counter() - started,
            self.model.device,
            next(self.model.parameters()).dtype,
            self.has_chat_template,
            self.max_context,
            self.eos_token_ids,
        )

    @staticmethod
    def _check_torch_backend() -> None:
        """Fail clearly when transformers has disabled its PyTorch backend.

        When the installed transformers requires a newer torch than the one
        present, it does not error at import: it logs a line at startup and
        carries on as a tokenizer-only library. The failure then surfaces much
        later as "AutoModelForCausalLM requires the PyTorch library but it was
        not found" on a machine that obviously has PyTorch installed. Checking
        here names the real problem — a version mismatch — and both versions.
        """
        try:
            from transformers.utils import is_torch_available
        except ImportError:
            return  # older transformers, or a stub in tests

        if is_torch_available():
            return

        import transformers

        raise RuntimeError(
            f"transformers {getattr(transformers, '__version__', '?')} has disabled its "
            f"PyTorch backend on torch {getattr(torch, '__version__', '?')}. This is a "
            "version mismatch, not a missing install: the transformers release requires a "
            "newer torch than the runtime provides. Either upgrade torch (in Docker, raise "
            "the BASE_IMAGE build arg) or pin transformers to a release that supports this "
            "torch version. Look for a 'Disabling PyTorch because ...' line earlier in the log."
        )

    @staticmethod
    def _check_local_path(model_id: str) -> None:
        """Fail early and clearly when MODEL_ID looks like a path but isn't one.

        When a local directory is missing, transformers falls back to treating
        the string as a Hub repo id and reports a confusing "Repo id must be in
        the form 'namespace/repo_name'" error. Catching it here says what is
        actually wrong: which absolute path was checked, and from where.
        """
        looks_like_path = (
            model_id.startswith((".", "/", "~"))
            or "\\" in model_id
            or (":" in model_id[:3])  # Windows drive letter
        )
        if not looks_like_path:
            return  # a Hub id such as meta-llama/Llama-3.2-1B

        path = Path(model_id).expanduser()
        resolved = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"MODEL_ID={model_id!r} looks like a local path but no directory "
                f"exists there. Checked: {resolved} (working directory: {Path.cwd()}). "
                "Relative paths resolve against the process working directory, which "
                "differs inside Docker — use an absolute path such as "
                "/models/your-model, and make sure the folder is mounted into the "
                "container."
            )

        required = "config.json"
        if not (path / required).is_file():
            contents = sorted(item.name for item in path.iterdir())[:10]
            raise FileNotFoundError(
                f"{resolved} exists but has no {required}, so it is not a "
                f"transformers checkpoint. Found: {contents}. If the weights sit in "
                "a subfolder, point MODEL_ID at that subfolder. Note that a single "
                ".gguf file is a llama.cpp artifact and cannot be loaded here."
            )
        logger.info("Loading from local directory %s", resolved)

    def _load_weights(self, auth: Dict[str, Any]):
        """Call ``from_pretrained`` with the configured dtype and device map.

        Handles the ``torch_dtype`` -> ``dtype`` argument rename in recent
        transformers releases so the same code runs on both.
        """
        settings = self.settings
        kwargs: Dict[str, Any] = {
            "device_map": settings.device_map,
            "low_cpu_mem_usage": True,
            "trust_remote_code": settings.trust_remote_code,
            **auth,
        }
        if settings.attn_implementation:
            kwargs["attn_implementation"] = settings.attn_implementation

        dtype = self._resolve_dtype()
        try:
            return AutoModelForCausalLM.from_pretrained(settings.model_id, dtype=dtype, **kwargs)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                settings.model_id, torch_dtype=dtype, **kwargs
            )

    def _resolve_dtype(self) -> torch.dtype:
        """Map the ``DTYPE`` setting to a torch dtype.

        ``auto`` picks bfloat16 on hardware that supports it (what Meta ships
        Llama 3.2 in), float16 on older GPUs, and float32 on CPU where the
        half precisions are slow or unsupported.
        """
        named = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        configured = self.settings.dtype.lower()
        if configured in named:
            return named[configured]
        if configured != "auto":
            raise ValueError(f"DTYPE must be one of auto|{'|'.join(named)}, got {configured!r}")

        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def _resolve_max_context(self) -> int:
        """Return the usable context length in tokens.

        Takes the smaller of what the model advertises and ``MAX_CONTEXT_TOKENS``.
        The cap matters: Llama 3.2 advertises a 128k window, and letting a chat
        history grow into it will exhaust KV-cache memory on a small GPU long
        before the model complains.
        """
        advertised = getattr(self.model.config, "max_position_embeddings", None) or getattr(
            self.tokenizer, "model_max_length", None
        )
        if not advertised or advertised > _IMPLAUSIBLE_CONTEXT:
            advertised = self.settings.max_context_tokens
        return min(int(advertised), self.settings.max_context_tokens)

    def _collect_eos_token_ids(self) -> List[int]:
        """Gather every token id that should terminate generation.

        Using only ``tokenizer.eos_token_id`` is the classic reason an instruct
        model runs to ``max_new_tokens``: Llama 3 ends assistant turns with
        ``<|eot_id|>``, not with ``<|end_of_text|>``.
        """
        ids: List[int] = []

        def add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple)):
                ids.extend(int(item) for item in value)
            else:
                ids.append(int(value))

        add(getattr(self.model.generation_config, "eos_token_id", None))
        add(self.tokenizer.eos_token_id)
        for token in KNOWN_EOT_TOKENS:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != self.tokenizer.unk_token_id:
                ids.append(token_id)
        return sorted(set(ids))

    # -- prompt construction -------------------------------------------------
    def render(self, messages: List[Dict[str, str]]) -> str:
        """Render a message list into the exact string the model expects.

        Chat models go through their own template with
        ``add_generation_prompt=True`` — omitting that flag is why templated
        models keep writing both sides of the conversation. Base models get a
        plain transcript instead.
        """
        if self.has_chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self._render_plain(messages)

    @staticmethod
    def _render_plain(messages: List[Dict[str, str]]) -> str:
        """Lay out a conversation for a model that has no chat template."""
        lines: List[str] = []
        for message in messages:
            role = (message.get("role") or "user").strip().lower()
            content = (message.get("content") or "").strip()
            if role == "system":
                lines.append(content)
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"User: {content}")
        lines.append("Assistant:")
        return "\n\n".join(lines)

    def count_tokens(self, text: str, add_special_tokens: Optional[bool] = None) -> int:
        """Return the token length of `text` under this model's tokenizer.

        `add_special_tokens` defaults to whatever prompt encoding uses. Pass
        False when measuring generated output: counting a completion as if it
        were a fresh prompt prepends BOS and inflates the total by one.
        """
        if add_special_tokens is None:
            add_special_tokens = self._adds_special_tokens()
        return len(self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

    def _adds_special_tokens(self) -> bool:
        """Whether the tokenizer should add BOS itself.

        Chat templates already emit ``<|begin_of_text|>``; letting the
        tokenizer add a second one measurably degrades output, so special
        tokens are only added when no template was applied.
        """
        return not self.has_chat_template

    def fit_to_context(
        self, messages: List[Dict[str, str]], max_new_tokens: int
    ) -> Tuple[str, List[Dict[str, str]], int]:
        """Trim history to fit the context window and render it.

        Trimming happens in **tokens**, dropping whole oldest turns, and the
        system message is always preserved. Slicing the rendered string by
        characters — the obvious shortcut — can cut a special token in half
        and silently corrupt the prompt.

        Returns the prompt text, the messages that survived, and their token count.
        """
        budget = max(64, self.max_context - max_new_tokens - self.settings.context_margin_tokens)

        system = [m for m in messages if (m.get("role") or "").lower() == "system"][:1]
        history = [m for m in messages if (m.get("role") or "").lower() != "system"]

        prompt = self.render(system + history)
        dropped = 0
        while history and self.count_tokens(prompt) > budget:
            history.pop(0)
            dropped += 1
            prompt = self.render(system + history)

        tokens = self.count_tokens(prompt)
        if tokens > budget:
            raise ValueError(
                f"Prompt needs {tokens} tokens but only {budget} are available. "
                "Shorten the system prompt or lower max_new_tokens."
            )
        if dropped:
            logger.info("Trimmed %d old message(s) to fit the context window", dropped)
        return prompt, system + history, tokens

    # -- generation ----------------------------------------------------------
    def _resolve(self, requested: Any, env_value: Any, config_field: str) -> Any:
        """Resolve one sampling knob: request > .env > model generation_config."""
        if requested is not None:
            return requested
        if env_value is not None:
            return env_value
        return getattr(self.model.generation_config, config_field, None)

    def _build_generation_kwargs(
        self,
        prompt: str,
        params: GenerationParams,
        streamer: Optional[TextIteratorStreamer],
        stop_flag: Dict[str, bool],
    ) -> Dict[str, Any]:
        """Assemble the full keyword argument set for ``model.generate``."""
        settings = self.settings
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=self._adds_special_tokens()
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        temperature = self._resolve(params.temperature, settings.temperature, "temperature")
        do_sample = bool(temperature and temperature > 0)

        kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": params.max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "eos_token_id": self.eos_token_ids or None,
            "pad_token_id": self.pad_token_id,
            "stopping_criteria": StoppingCriteriaList([StopOnFlag(stop_flag)]),
        }
        if streamer is not None:
            kwargs["streamer"] = streamer

        # Passing temperature/top_p while greedy makes transformers warn, so
        # the sampling knobs are only attached when sampling is actually on.
        if do_sample:
            kwargs["temperature"] = temperature
            top_p = self._resolve(params.top_p, settings.top_p, "top_p")
            top_k = self._resolve(params.top_k, settings.top_k, "top_k")
            if top_p is not None:
                kwargs["top_p"] = top_p
            if top_k:
                kwargs["top_k"] = top_k

        penalty = self._resolve(
            params.repetition_penalty, settings.repetition_penalty, "repetition_penalty"
        )
        if penalty:
            kwargs["repetition_penalty"] = penalty

        stop_strings = list(params.stop or [])
        # Base models have no turn markers, so they will happily continue the
        # transcript with "User:" unless told otherwise.
        if not self.has_chat_template:
            stop_strings.extend(settings.fallback_stop_strings)
        if stop_strings:
            kwargs["stop_strings"] = stop_strings
            kwargs["tokenizer"] = self.tokenizer  # required alongside stop_strings

        if params.seed is not None:
            torch.manual_seed(params.seed)

        return kwargs

    async def astream(
        self,
        prompt: str,
        params: GenerationParams,
        should_stop: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[str]:
        """Yield generated text chunks as they are produced.

        ``model.generate`` is blocking, so it runs on a worker thread while
        this coroutine drains the streamer. Each ``next()`` is dispatched with
        ``asyncio.to_thread`` rather than iterated directly: a bare
        ``for chunk in streamer`` blocks the event loop, which stalls every
        other request and prevents the disconnect check from ever running.

        `should_stop` is polled between chunks; when it returns True the
        stopping criterion aborts generation instead of burning GPU time on a
        response nobody is listening to.
        """
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        stop_flag = {"stop": False}
        failure: Dict[str, BaseException] = {}
        kwargs = self._build_generation_kwargs(prompt, params, streamer, stop_flag)

        def worker() -> None:
            """Run generation on a background thread, never raising into it."""
            try:
                with torch.inference_mode():
                    self.model.generate(**kwargs)
            except BaseException as exc:  # surfaced to the caller after the loop
                failure["error"] = exc
            finally:
                # Unblocks the consumer even when generate() died early.
                streamer.end()

        thread = Thread(target=worker, name="generate", daemon=True)
        thread.start()

        iterator = iter(streamer)
        sentinel = object()
        try:
            while True:
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                if chunk:
                    yield chunk
                if await should_stop():
                    stop_flag["stop"] = True
                    break
        finally:
            stop_flag["stop"] = True
            await asyncio.to_thread(thread.join, self.settings.generation_join_timeout)

        if "error" in failure:
            raise failure["error"]

    def complete(self, prompt: str, params: GenerationParams) -> Tuple[str, int]:
        """Generate a full response without streaming.

        Returns the decoded text and the number of tokens generated. Runs
        blocking work inline, so callers should dispatch it with
        ``asyncio.to_thread``.
        """
        kwargs = self._build_generation_kwargs(prompt, params, None, {"stop": False})
        input_length = kwargs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**kwargs)
        generated = output[0][input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True), int(generated.shape[-1])

    def describe(self) -> Dict[str, Any]:
        """Return a snapshot of the loaded model for `/health` and logs."""
        return {
            "model_id": self.settings.model_id,
            "device": str(self.model.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "chat_template": self.has_chat_template,
            "max_context_tokens": self.max_context,
            "eos_token_ids": self.eos_token_ids,
        }