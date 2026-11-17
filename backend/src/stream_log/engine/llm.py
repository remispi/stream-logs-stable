"""Local LLM access through any OpenAI-compatible server (llama-server, Ollama, LM Studio).

Nothing here talks to the internet: the default endpoints are all on 127.0.0.1. When no server is
running, `find_llm` returns None and the LLM steps are skipped (the report says so).
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

DEFAULT_ENDPOINTS = [
    "http://127.0.0.1:8080/v1",  # llama-server
    "http://127.0.0.1:11434/v1",  # Ollama
    "http://127.0.0.1:1234/v1",  # LM Studio
]
URL_ENV = "STREAM_LOG_LLM_URL"
MODEL_ENV = "STREAM_LOG_LLM_MODEL"
REASONING_BUDGET = 3000


class LlmError(RuntimeError):
    pass


def _request(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


@dataclass
class LlmClient:
    base_url: str
    model: str
    temperature: float = 0.2
    reasons: bool | None = None  # the model's `serve.always_reasons`, if its manifest says

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.base_url}"

    @property
    def always_reasons(self) -> bool:
        """gpt-oss ignores llama-server's `--reasoning off` and thinks before every answer.
        Other models like it say so in their manifest entry (`serve.always_reasons`)."""
        if self.reasons is not None:
            return self.reasons
        return "gpt-oss" in self.model.lower()

    def chat(self, system: str, user: str, *, schema: dict | None = None,
             max_tokens: int = 4096) -> str:
        if self.always_reasons:
            # Its own convention for effort; without it, it spent 3000 tokens thinking and never
            # answered. Even at "low" it thinks for ~1000 tokens, so add that to the budget.
            system = "Reasoning: low\n\n" + system
            max_tokens += REASONING_BUDGET
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": True},
            }
        # A server that dies mid-answer (out of memory) drops the connection: that is one failed
        # request (OSError, HTTPException), not the end of a run of hundreds.
        try:
            data = _request(f"{self.base_url}/chat/completions", payload)
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise LlmError(f"LLM request failed: {error}") from error
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise LlmError(f"unexpected answer from {self.model}: {error!r}") from error
        if choice.get("finish_reason") == "length" and (schema is not None or not content):
            raise LlmError(f"{self.model} hit the {max_tokens}-token limit before finishing")
        return content

    def chat_json(self, system: str, user: str, schema: dict, **kwargs) -> dict:
        text = self.chat(system, user, schema=schema, **kwargs)
        return parse_json(text)


def parse_json(text: str) -> dict:
    """Parse model output, tolerating code fences and thinking blocks around the JSON."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.rpartition("</think>")[2]  # a block opened by the chat template, in the prompt
    text = text.partition("<think>")[0].strip()  # a block cut short: no answer after it
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise LlmError(f"no JSON in LLM output: {text[:200]!r}")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise LlmError(f"invalid JSON from LLM: {error}") from error
    return value if isinstance(value, dict) else {"items": value}


def list_models(base_url: str, timeout: float = 2) -> list[str]:
    data = _request(f"{base_url}/models", timeout=timeout)
    return [m["id"] for m in data.get("data", [])]


def _port_open(base_url: str, timeout: float = 0.3) -> bool:
    """Quick check first: on Windows a refused connection otherwise takes ~2 s per endpoint."""
    parsed = urllib.parse.urlsplit(base_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def find_llm(url: str | None = None, model: str | None = None) -> LlmClient | None:
    """Use the given / $STREAM_LOG_LLM_URL endpoint, else the first local server that answers."""
    url = url or os.environ.get(URL_ENV)
    model = model or os.environ.get(MODEL_ENV)
    for base in [url] if url else DEFAULT_ENDPOINTS:
        base = base.rstrip("/")
        if not _port_open(base):
            continue
        try:
            models = list_models(base)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            continue
        if model:
            return LlmClient(base, model)
        if models:
            return LlmClient(base, models[0])
    return None
