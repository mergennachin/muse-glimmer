"""Synchronous Python client for a running Muse Glimmer server."""

import json
import os
import sys
from typing import Any

from .constants import MODEL_ID, TIMINGS_HEADER


class Glimmer:
    """Connect to the OpenAI-compatible API exposed by `muse-glimmer serve`."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str = MODEL_ID,
        timeout: float | None = 300.0,
        timings: bool = False,
    ):
        if base_url is None:
            base_url = os.environ.get("MUSE_GLIMMER_BASE_URL", "http://127.0.0.1:8000/v1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.timings = timings

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send an independent prompt and return the final assistant text."""
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        completion = self.chat(messages, **options)
        return completion["choices"][0]["message"].get("content") or ""

    def chat(self, messages: list[dict[str, Any]], **options: Any) -> dict[str, Any]:
        """Submit a message history and return the full chat completion as a dict."""
        import httpx

        if options.get("stream"):
            raise ValueError("Glimmer.chat() requires stream=False.")
        body = {"model": self.model, **options, "messages": messages}
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={TIMINGS_HEADER: "1"} if self.timings else None,
                timeout=self.timeout,
                trust_env=False,
            )
        except httpx.ConnectError as error:
            raise httpx.ConnectError(
                f"Could not connect to {self.base_url}. Start `muse-glimmer serve` "
                "or check the client's base_url.",
                request=error.request,
            ) from error
        response.raise_for_status()
        completion = response.json()
        if self.timings:
            fields = []
            if TIMINGS_HEADER in response.headers:
                timings = json.loads(response.headers[TIMINGS_HEADER])
                completion["timings"] = timings
                for key, label, unit in (
                    ("ttft_ms", "TTFT", "ms"),
                    ("prefill_tok_s", "prefill", "tok/s"),
                    ("decode_tok_s", "decode", "tok/s"),
                ):
                    value = timings.get(key)
                    fields.append(f"{label} {value:.1f} {unit}" if value is not None else f"{label} n/a")
            else:
                fields.append("server timings unavailable")
            print(" | ".join(fields), file=sys.stderr)
        return completion
