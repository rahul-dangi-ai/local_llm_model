"""
Streaming LLM inference server.

A small, dependency-light FastAPI service that serves any HuggingFace causal
language model over HTTP with token-by-token Server-Sent Events, correct
multi-turn context handling, and interruption when the client goes away.

Endpoints
    GET    /health              liveness plus the loaded model's real limits
    POST   /generate            streaming (SSE) or buffered completion
    DELETE /sessions/{id}       forget a server-side conversation

Design notes
    * Every setting comes from the environment — see `config.py` and
      `.env.example`. No tunable is hardcoded here.
    * Conversation state is optional. Send the full `messages` array for a
      stateless service, or send a `session_id` and let the server keep the
      history within the model's context window.
    * Blocking model work never runs on the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from config import Settings, get_settings
from llm import GenerationParams, LLMEngine, Usage

SETTINGS = get_settings()
logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging(settings: Settings) -> None:
    """Install a single root handler with a consistent, greppable format.

    ``force=True`` replaces whatever uvicorn set up first, so application and
    server lines share one format instead of interleaving two.
    """
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    # Access logs duplicate what the request logger already records.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Message(BaseModel):
    """One conversation turn."""

    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class GenerateRequest(BaseModel):
    """Body of ``POST /generate``.

    Provide exactly one input source: `messages` for chat, or `prompt` for raw
    completion against a base model. `session_id` lets the server hold the
    history so a client only has to send the newest turn.
    """

    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None
    session_id: Optional[str] = None

    max_new_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    repetition_penalty: Optional[float] = Field(None, ge=0.5, le=2.0)
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    stream: bool = True

    @model_validator(mode="after")
    def check_input_source(self) -> "GenerateRequest":
        """Reject bodies that supply neither or both input sources."""
        if self.prompt is not None and self.messages:
            raise ValueError("Send either `messages` or `prompt`, not both.")
        if self.prompt is None and not self.messages:
            raise ValueError("Send either `messages` or `prompt`.")
        if self.prompt is not None and self.session_id:
            raise ValueError("`session_id` applies to `messages`, not raw `prompt`.")
        return self

    def to_params(self, settings: Settings) -> GenerationParams:
        """Convert the request into engine-level generation parameters."""
        requested = self.max_new_tokens or settings.max_new_tokens
        if requested > settings.max_new_tokens_limit:
            raise HTTPException(
                422,  # integer literal: the starlette constant was renamed across versions
                f"max_new_tokens exceeds the server limit of {settings.max_new_tokens_limit}.",
            )
        return GenerationParams(
            max_new_tokens=requested,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@dataclass
class Session:
    """A server-held conversation and the time it was last touched."""

    messages: List[Dict[str, str]] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


class SessionStore:
    """In-memory conversation store with TTL and size caps.

    Deliberately process-local: it keeps single-node deployments simple and
    makes the memory ceiling explicit. Put Redis behind this interface if you
    need to run more than one replica.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> List[Dict[str, str]]:
        """Return the stored history for `session_id`, or an empty list."""
        async with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_seen = time.time()
            return list(session.messages)

    async def save(self, session_id: str, messages: List[Dict[str, str]]) -> None:
        """Replace the history for `session_id`, trimming to the turn cap."""
        async with self._lock:
            self._evict_expired()
            self._evict_overflow()
            capped = messages[-self.settings.session_max_turns :]
            self._sessions[session_id] = Session(messages=capped, last_seen=time.time())

    async def delete(self, session_id: str) -> bool:
        """Forget a conversation. Returns whether anything was removed."""
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _evict_expired(self) -> None:
        """Drop sessions idle for longer than the configured TTL."""
        cutoff = time.time() - self.settings.session_ttl_seconds
        for key in [k for k, v in self._sessions.items() if v.last_seen < cutoff]:
            del self._sessions[key]

    def _evict_overflow(self) -> None:
        """Drop the least recently used sessions once the cap is reached."""
        overflow = len(self._sessions) - self.settings.session_max_count + 1
        if overflow <= 0:
            return
        oldest = sorted(self._sessions.items(), key=lambda item: item[1].last_seen)
        for key, _ in oldest[:overflow]:
            del self._sessions[key]


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
engine = LLMEngine(SETTINGS)
sessions = SessionStore(SETTINGS)
# One model instance cannot serve two `generate` calls at once; the semaphore
# serialises them so requests queue instead of corrupting each other.
generation_slots = asyncio.Semaphore(SETTINGS.max_concurrency)


async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model before the server accepts traffic."""
    configure_logging(SETTINGS)
    engine.load()
    logger.info("Listening on %s:%d", SETTINGS.host, SETTINGS.port)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Streaming LLM Inference",
    description="Model-agnostic streaming inference for HuggingFace causal LMs.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_api_key(request: Request) -> None:
    """Enforce bearer-token auth when ``API_KEY`` is configured.

    Auth is opt-in so local development stays frictionless, but any
    internet-facing deployment should set the variable: an unprotected
    endpoint is an open invitation to burn someone else's GPU.
    """
    if not SETTINGS.api_key:
        return
    header = request.headers.get("authorization", "")
    presented = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-api-key")
    if presented != SETTINGS.api_key:
        raise HTTPException(401, "Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sse(payload: Dict[str, Any]) -> str:
    """Encode a payload as one Server-Sent Event frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_conversation(
    request_body: GenerateRequest, history: List[Dict[str, str]], settings: Settings
) -> List[Dict[str, str]]:
    """Merge stored history, the default system prompt, and the new messages.

    The default system prompt is only injected when the caller did not supply
    one of their own, so per-request personas always win.
    """
    incoming = [message.model_dump() for message in (request_body.messages or [])]
    conversation = history + incoming
    has_system = any(message["role"] == "system" for message in conversation)
    if settings.default_system_prompt and not has_system:
        conversation.insert(0, {"role": "system", "content": settings.default_system_prompt})
    return conversation


def finish_reason(completion_tokens: int, params: GenerationParams, cancelled: bool) -> str:
    """Classify why generation ended, mirroring common API conventions."""
    if cancelled:
        return "cancelled"
    if completion_tokens >= params.max_new_tokens:
        return "length"
    return "stop"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    """Report liveness and the limits the server actually resolved at load."""
    return {
        "status": "ok",
        "model": engine.describe(),
        "defaults": {
            "max_new_tokens": SETTINGS.max_new_tokens,
            "max_new_tokens_limit": SETTINGS.max_new_tokens_limit,
            "max_concurrency": SETTINGS.max_concurrency,
            "sessions_enabled": SETTINGS.session_enabled,
        },
    }


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
async def delete_session(session_id: str) -> Dict[str, Any]:
    """Forget a server-side conversation."""
    removed = await sessions.delete(session_id)
    return {"session_id": session_id, "deleted": removed}


@app.post("/generate", dependencies=[Depends(require_api_key)])
async def generate(body: GenerateRequest, request: Request):
    """Generate a response, streamed as SSE by default.

    Streaming frames:
        ``{"token": "..."}``            one chunk of text
        ``{"done": true, "usage": {}}`` final accounting
        ``[DONE]``                      terminator

    The token frame keeps the shape earlier clients expect, so the extra
    frames are additive rather than breaking.
    """
    request_id = uuid.uuid4().hex[:8]
    params = body.to_params(SETTINGS)

    # -- assemble the prompt -------------------------------------------------
    if body.prompt is not None:
        prompt, conversation, prompt_tokens = body.prompt, [], engine.count_tokens(body.prompt)
    else:
        history: List[Dict[str, str]] = []
        if body.session_id and SETTINGS.session_enabled:
            history = await sessions.get(body.session_id)
        conversation = build_conversation(body, history, SETTINGS)
        try:
            prompt, conversation, prompt_tokens = engine.fit_to_context(
                conversation, params.max_new_tokens
            )
        except ValueError as exc:
            raise HTTPException(413, str(exc)) from exc

    logger.info(
        "[%s] generate session=%s prompt_tokens=%d max_new_tokens=%d stream=%s",
        request_id,
        body.session_id or "-",
        prompt_tokens,
        params.max_new_tokens,
        body.stream,
    )
    if SETTINGS.log_prompts:
        logger.debug("[%s] prompt=%r", request_id, prompt)

    if not body.stream:
        return await _generate_buffered(body, params, prompt, conversation, prompt_tokens, request_id)

    return StreamingResponse(
        _stream(body, params, prompt, conversation, prompt_tokens, request_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stops nginx buffering the stream
            "X-Request-ID": request_id,
        },
    )


async def _stream(
    body: GenerateRequest,
    params: GenerationParams,
    prompt: str,
    conversation: List[Dict[str, str]],
    prompt_tokens: int,
    request_id: str,
    request: Request,
) -> AsyncIterator[str]:
    """Drive the engine and emit SSE frames until generation ends."""
    started = time.perf_counter()
    first_token_at: Optional[float] = None
    pieces: List[str] = []
    cancelled = False

    async def should_stop() -> bool:
        """Stop as soon as the client hangs up."""
        nonlocal cancelled
        if await request.is_disconnected():
            cancelled = True
            logger.info("[%s] client disconnected, interrupting generation", request_id)
            return True
        return False

    async with generation_slots:
        try:
            async for chunk in engine.astream(prompt, params, should_stop):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(chunk)
                yield sse({"token": chunk})
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            logger.exception("[%s] generation failed", request_id)
            yield sse({"error": str(exc)})
            yield "data: [DONE]\n\n"
            return

    text = "".join(pieces)
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=engine.count_tokens(text, add_special_tokens=False) if text else 0,
        duration_seconds=time.perf_counter() - started,
        time_to_first_token=(first_token_at - started) if first_token_at else None,
        finish_reason="",
    )
    usage.finish_reason = finish_reason(usage.completion_tokens, params, cancelled)

    if body.session_id and SETTINGS.session_enabled and text and not cancelled:
        await sessions.save(
            body.session_id, conversation + [{"role": "assistant", "content": text}]
        )

    logger.info("[%s] %s", request_id, usage.as_dict())
    yield sse({"done": True, "usage": usage.as_dict()})
    yield "data: [DONE]\n\n"


async def _generate_buffered(
    body: GenerateRequest,
    params: GenerationParams,
    prompt: str,
    conversation: List[Dict[str, str]],
    prompt_tokens: int,
    request_id: str,
) -> JSONResponse:
    """Generate without streaming and return the whole response at once."""
    started = time.perf_counter()
    async with generation_slots:
        try:
            text, completion_tokens = await asyncio.to_thread(engine.complete, prompt, params)
        except Exception as exc:
            logger.exception("[%s] generation failed", request_id)
            raise HTTPException(500, str(exc)) from exc

    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_seconds=time.perf_counter() - started,
        time_to_first_token=None,
        finish_reason=finish_reason(completion_tokens, params, cancelled=False),
    )
    if body.session_id and SETTINGS.session_enabled and text:
        await sessions.save(
            body.session_id, conversation + [{"role": "assistant", "content": text}]
        )

    logger.info("[%s] %s", request_id, usage.as_dict())
    return JSONResponse(
        {"text": text, "usage": usage.as_dict()}, headers={"X-Request-ID": request_id}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=SETTINGS.host, port=SETTINGS.port, log_level=SETTINGS.log_level.lower())