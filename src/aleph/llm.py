"""Thin LLM adapter. Anthropic by default; replaceable.

The interface is just two functions: `complete` (prose) and `complete_json` (structured).
If you want OpenAI or a local model, rewrite this file only.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Optional

try:
    from anthropic import (
        Anthropic,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    _RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore
    _RETRYABLE = ()  # type: ignore

from .log import log


DEFAULT_MODEL = os.environ.get("ALEPH_MODEL", "claude-opus-4-7")
# Set at runtime from CLI if you prefer something smaller/faster for bulk ingest.

# Transient-error retry budget. Tune via env for long ingest runs against a
# flaky network. Only transport/429/5xx errors are retried — 4xx client errors
# (bad request, auth, etc.) still fail immediately.
_DEFAULT_MAX_ATTEMPTS = int(os.environ.get("ALEPH_LLM_MAX_ATTEMPTS", "3"))
_DEFAULT_BACKOFF_BASE = float(os.environ.get("ALEPH_LLM_BACKOFF_BASE", "0.5"))


class LLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
    ):
        if Anthropic is None:
            raise RuntimeError("Install `anthropic` to use the default LLM adapter.")
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_attempts = max(1, max_attempts)
        self.backoff_base = backoff_base

    def _create(self, system: str, user: str, max_tokens: int):
        """Call messages.create with exponential-backoff retry on transient errors.

        Retries: transport failures, timeouts, 429s, and 5xx responses. Other
        errors (4xx, validation) bubble up immediately.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            except _RETRYABLE as e:
                last_err = e
                if attempt == self.max_attempts - 1:
                    log(
                        "llm_retry_exhausted",
                        level="error",
                        attempts=self.max_attempts,
                        error=type(e).__name__,
                    )
                    raise
                delay = self.backoff_base * (2 ** attempt) + random.uniform(0, 0.25)
                log(
                    "llm_retry",
                    level="warning",
                    attempt=attempt + 1,
                    max_attempts=self.max_attempts,
                    error=type(e).__name__,
                    delay_s=round(delay, 3),
                )
                time.sleep(delay)
        # unreachable: the loop either returns or raises on the final attempt
        raise last_err  # type: ignore[misc]

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._create(system, user, max_tokens)
        # concatenate all text blocks
        parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip()

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> list | dict:
        """Ask for JSON; strip fences; parse. Retries once on parse failure."""
        raw = self.complete(system + "\n\nReturn ONLY valid JSON. No prose, no markdown fences.",
                            user, max_tokens=max_tokens)
        try:
            return _parse_json(raw)
        except ValueError:
            # one retry with a sharper nudge
            fix_prompt = (
                f"The previous response was not valid JSON. Here it is:\n\n{raw}\n\n"
                "Re-emit as strict JSON only, with no other text."
            )
            raw2 = self.complete(system, fix_prompt, max_tokens=max_tokens)
            return _parse_json(raw2)


def _parse_json(text: str) -> list | dict:
    text = text.strip()
    # strip common fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # fallback: find the first { or [ and match to the last } or ]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{]", text)
        if not m:
            raise ValueError(f"No JSON found in response: {text[:200]}")
        start = m.start()
        # try trailing } or ]
        for end in range(len(text), start, -1):
            chunk = text[start:end]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
        raise ValueError(f"Could not parse JSON: {text[:200]}")
