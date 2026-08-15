#!/usr/bin/env python3
"""
Interactive chat client for the streaming inference server.

A terminal REPL that holds a continuous conversation against `/generate`,
renders tokens as they arrive, and reports timing after every turn. Standard
library only — no dependencies beyond Python itself.

    python chat.py                                   # talk to localhost:8000
    python chat.py --url http://gpu-box:8000
    python chat.py --stateless                       # client keeps the history
    python chat.py --raw                             # completion mode, for base models
    python chat.py --system "You are a terse assistant."

Type `/help` at the prompt for in-session commands. Ctrl-C during a response
interrupts that generation without leaving the chat — which is also the
easiest way to verify the server's disconnect handling actually works.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

# --- terminal colours (disabled when piped) --------------------------------
_TTY = sys.stdout.isatty()


def paint(text: str, code: str) -> str:
    """Wrap `text` in an ANSI colour, or return it unchanged when redirected."""
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def dim(text: str) -> str:
    """Render secondary information (stats, hints)."""
    return paint(text, "2")


def bold(text: str) -> str:
    """Render emphasis."""
    return paint(text, "1")


def red(text: str) -> str:
    """Render an error."""
    return paint(text, "31")


def cyan(text: str) -> str:
    """Render the assistant label."""
    return paint(text, "36")


def yellow(text: str) -> str:
    """Render a warning or notice."""
    return paint(text, "33")


class Stats:
    """Running totals across the session, reported by `/stats`."""

    def __init__(self) -> None:
        self.turns = 0
        self.interrupted = 0
        self.errors = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.duration = 0.0
        self.ttfts: List[float] = []

    def record(self, usage: Dict[str, Any]) -> None:
        """Fold one turn's usage block into the totals."""
        self.turns += 1
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.duration += usage.get("duration_seconds", 0.0)
        if usage.get("time_to_first_token") is not None:
            self.ttfts.append(usage["time_to_first_token"])
        if usage.get("finish_reason") == "cancelled":
            self.interrupted += 1

    def render(self) -> str:
        """Format the session summary."""
        if not self.turns:
            return "No turns yet."
        tps = self.completion_tokens / self.duration if self.duration else 0.0
        avg_ttft = sum(self.ttfts) / len(self.ttfts) if self.ttfts else 0.0
        return (
            f"turns={self.turns}  interrupted={self.interrupted}  errors={self.errors}\n"
            f"prompt_tokens={self.prompt_tokens}  completion_tokens={self.completion_tokens}\n"
            f"total_time={self.duration:.1f}s  avg_throughput={tps:.1f} tok/s  "
            f"avg_ttft={avg_ttft:.2f}s"
        )


class Client:
    """Thin HTTP client for the server's two endpoints."""

    def __init__(self, url: str, api_key: Optional[str], timeout: float) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        """Build request headers, including auth when a key is configured."""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, path: str, payload: Optional[Dict[str, Any]], method: str = "POST"):
        """Open a request and return the raw response object."""
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.url}{path}", data=data, headers=self._headers(), method=method
        )
        return urllib.request.urlopen(request, timeout=self.timeout)

    def health(self) -> Dict[str, Any]:
        """Fetch `/health`."""
        with self._request("/health", None, method="GET") as response:
            return json.loads(response.read())

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        """Ask the server to forget a conversation."""
        with self._request(f"/sessions/{session_id}", None, method="DELETE") as response:
            return json.loads(response.read())

    def stream(self, payload: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], bool]:
        """Stream one generation, printing tokens as they arrive.

        Returns the accumulated text, the usage block if the server sent one,
        and whether the user interrupted with Ctrl-C. Interruption closes the
        connection, which is exactly what triggers the server's disconnect
        handling — the response is not just hidden client-side, the GPU work
        actually stops.
        """
        pieces: List[str] = []
        usage: Optional[Dict[str, Any]] = None
        interrupted = False
        started = time.perf_counter()
        first_token_at: Optional[float] = None

        response = self._request("/generate", payload)
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break

                event = json.loads(body)
                if "token" in event:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(event["token"])
                    sys.stdout.write(event["token"])
                    sys.stdout.flush()
                elif event.get("done"):
                    usage = event.get("usage")
                elif "error" in event:
                    print(red(f"\n[server error] {event['error']}"))
        except KeyboardInterrupt:
            interrupted = True
            print(yellow("\n[interrupted — connection closed, server should stop generating]"))
        finally:
            response.close()

        if not usage and pieces:
            # Interrupted before the usage frame: report what we measured.
            elapsed = time.perf_counter() - started
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": None,
                "duration_seconds": round(elapsed, 3),
                "time_to_first_token": (
                    round(first_token_at - started, 3) if first_token_at else None
                ),
                "finish_reason": "cancelled (client-side)",
            }
        return "".join(pieces), usage, interrupted


HELP = """
Commands
  /help              show this
  /health            print the server's model info
  /reset             clear the conversation (server-side and local)
  /new               start a fresh session id
  /stats             session totals
  /history           show the local message history
  /system <text>     set a system prompt and reset
  /temp <float>      set temperature (0 = greedy)
  /tokens <int>      set max_new_tokens
  /raw               toggle completion mode (send `prompt` instead of `messages`)
  /stateless         toggle who owns the history (server session vs this client)
  /quit              exit

Ctrl-C during a response interrupts that generation. Ctrl-C at the prompt exits.
"""


class Chat:
    """The REPL: owns conversation state and dispatches commands."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.client = Client(args.url, args.api_key, args.timeout)
        self.session_id = args.session or f"cli-{uuid.uuid4().hex[:8]}"
        self.stateless = args.stateless
        self.raw = args.raw
        self.system = args.system
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.history: List[Dict[str, str]] = []
        self.stats = Stats()

    # -- startup -------------------------------------------------------------
    def preflight(self) -> bool:
        """Check the server is up and report what it loaded.

        Warns when the checkpoint has no chat template, since that is the
        single most common reason a conversation "doesn't work": base models
        continue text instead of answering.
        """
        try:
            health = self.client.health()
        except urllib.error.URLError as exc:
            print(red(f"Cannot reach {self.client.url}: {exc}"))
            print(dim("Is the server running?  docker compose up -d"))
            return False

        model = health.get("model", {})
        print(bold(f"Connected to {self.client.url}"))
        print(
            dim(
                f"  model={model.get('model_id')}  device={model.get('device')}  "
                f"dtype={model.get('dtype')}\n"
                f"  context={model.get('max_context_tokens')}  "
                f"chat_template={model.get('chat_template')}  "
                f"eos_ids={model.get('eos_token_ids')}"
            )
        )
        if not model.get("chat_template") and not self.raw:
            print(
                yellow(
                    "  Warning: this is a BASE model (no chat template). It continues text\n"
                    "  rather than answering questions. Expect rambling or looping replies.\n"
                    "  Use an -Instruct checkpoint for chat, or /raw for completion mode."
                )
            )
        mode = "raw completion" if self.raw else ("stateless" if self.stateless else "server session")
        print(dim(f"  mode={mode}  session={self.session_id}  /help for commands\n"))
        return True

    # -- commands ------------------------------------------------------------
    def handle_command(self, line: str) -> bool:
        """Run a slash command. Returns False when the user wants to exit."""
        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command in {"/quit", "/exit", "/q"}:
            return False
        if command == "/help":
            print(dim(HELP))
        elif command == "/health":
            print(dim(json.dumps(self.client.health(), indent=2)))
        elif command == "/stats":
            print(dim(self.stats.render()))
        elif command == "/history":
            if not self.history:
                print(dim("(empty — the server holds the history in session mode)"))
            for message in self.history:
                print(dim(f"  {message['role']}: {message['content'][:100]}"))
        elif command == "/reset":
            self.reset()
        elif command == "/new":
            self.session_id = f"cli-{uuid.uuid4().hex[:8]}"
            self.history.clear()
            print(dim(f"new session: {self.session_id}"))
        elif command == "/system":
            self.system = argument or None
            self.reset()
            print(dim(f"system prompt: {self.system!r}"))
        elif command == "/temp":
            self.temperature = float(argument)
            print(dim(f"temperature={self.temperature}"))
        elif command == "/tokens":
            self.max_tokens = int(argument)
            print(dim(f"max_new_tokens={self.max_tokens}"))
        elif command == "/raw":
            self.raw = not self.raw
            print(dim(f"raw completion mode: {self.raw}"))
        elif command == "/stateless":
            self.stateless = not self.stateless
            self.history.clear()
            print(dim(f"stateless: {self.stateless} (history cleared)"))
        else:
            print(dim(f"unknown command {command} — /help"))
        return True

    def reset(self) -> None:
        """Clear both the local history and the server-side session."""
        self.history.clear()
        try:
            result = self.client.reset_session(self.session_id)
            print(dim(f"conversation cleared (server: {result.get('deleted')})"))
        except urllib.error.URLError as exc:
            print(red(f"reset failed: {exc}"))

    # -- one turn ------------------------------------------------------------
    def build_payload(self, user_input: str) -> Dict[str, Any]:
        """Assemble the request body for the current mode."""
        payload: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "stream": True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        if self.raw:
            payload["prompt"] = user_input
            return payload

        if self.stateless:
            messages = list(self.history) + [{"role": "user", "content": user_input}]
            if self.system and not any(m["role"] == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": self.system})
            payload["messages"] = messages
        else:
            # Server holds the history; send only the newest turn.
            messages = [{"role": "user", "content": user_input}]
            if self.system and not self.history:
                messages.insert(0, {"role": "system", "content": self.system})
            payload["messages"] = messages
            payload["session_id"] = self.session_id
        return payload

    def turn(self, user_input: str) -> None:
        """Send one message and stream the reply."""
        payload = self.build_payload(user_input)
        print(cyan("assistant> "), end="", flush=True)

        try:
            text, usage, interrupted = self.client.stream(payload)
        except urllib.error.HTTPError as exc:
            self.stats.errors += 1
            detail = exc.read().decode("utf-8", errors="replace")
            print(red(f"\nHTTP {exc.code}: {detail[:500]}"))
            return
        except urllib.error.URLError as exc:
            self.stats.errors += 1
            print(red(f"\nconnection failed: {exc}"))
            return

        if not text.endswith("\n"):
            print()
        if usage:
            self.stats.record(usage)
            print(
                dim(
                    f"  [{usage.get('completion_tokens')} tok in "
                    f"{usage.get('duration_seconds')}s | "
                    f"{usage.get('tokens_per_second', '-')} tok/s | "
                    f"ttft {usage.get('time_to_first_token')}s | "
                    f"{usage.get('finish_reason')}]"
                )
            )
            if usage.get("finish_reason") == "length":
                print(dim("  (hit max_new_tokens — raise it with /tokens)"))

        # In stateless mode this client owns the transcript.
        if self.stateless and not self.raw and text and not interrupted:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": text})
        elif not self.stateless and not self.raw:
            self.history.append({"role": "user", "content": user_input})

    # -- loop ----------------------------------------------------------------
    def run(self) -> int:
        """Run the REPL until the user exits."""
        if not self.preflight():
            return 1
        while True:
            try:
                user_input = input(bold("you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(dim("\nbye"))
                break
            if not user_input:
                continue
            if user_input.startswith("/"):
                if not self.handle_command(user_input):
                    print(dim("bye"))
                    break
                continue
            self.turn(user_input)

        if self.stats.turns:
            print(dim("\n" + self.stats.render()))
        return 0


def parse_args() -> argparse.Namespace:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Interactive client for the streaming inference server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="server base URL")
    parser.add_argument("--api-key", default=None, help="sent as a bearer token when set")
    parser.add_argument("--session", default=None, help="reuse a specific session id")
    parser.add_argument("--system", default=None, help="system prompt")
    parser.add_argument("--temperature", type=float, default=None, help="omit to use server default")
    parser.add_argument("--max-tokens", type=int, default=256, help="max_new_tokens per turn")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="this client sends the full history instead of using a server session",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="completion mode: send `prompt` instead of `messages` (for base models)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(Chat(parse_args()).run())