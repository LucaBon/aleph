"""Thin LLM adapter. Anthropic by default; replaceable.

The interface is just two functions: `complete` (prose) and `complete_json` (structured).
If you want OpenAI or a local model, rewrite this file only.

A deterministic :class:`MockLLM` adapter (P2.1) is also provided here for
tests and offline CI. It reads a fixtures file mapping
``(system_prompt_hash, user_prompt_hash)`` to a canned response, so the
full pipeline (ingest, ask, verifier, contradiction-scan,
concept-validate, …) can exercise its golden paths without a live API
key.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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


# First-party API list prices, USD per million tokens: (input, output).
# Cache writes (5-minute TTL) bill at 1.25x input, cache reads at 0.1x input.
# Taken from Anthropic's model/pricing table as cached 2026-06-24; re-check
# before quoting costs. A model missing here reports cost as unknown (None)
# rather than a guess.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


@dataclass
class Usage:
    """Token usage accumulated over an adapter's calls."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, **tokens) -> None:
        self.calls += 1
        for k, v in tokens.items():
            setattr(self, k, getattr(self, k) + int(v or 0))

    def minus(self, earlier: "Usage") -> "Usage":
        return Usage(**{k: v - getattr(earlier, k) for k, v in asdict(self).items()})

    def copy(self) -> "Usage":
        return Usage(**asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)


def cost_usd(model: str, usage: Usage) -> Optional[float]:
    """List-price cost of ``usage`` on ``model``, or None if the model's price
    is unknown."""
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    inp, out = price
    return (
        usage.input_tokens * inp
        + usage.cache_creation_input_tokens * inp * 1.25
        + usage.cache_read_input_tokens * inp * 0.1
        + usage.output_tokens * out
    ) / 1_000_000


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
        self.usage = Usage()

    def _create(self, system: str, user: str, max_tokens: int):
        """Call messages.create with exponential-backoff retry on transient errors.

        Retries: transport failures, timeouts, 429s, and 5xx responses. Other
        errors (4xx, validation) bubble up immediately.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                u = getattr(resp, "usage", None)
                if u is not None:
                    self.usage.add(**{
                        k: getattr(u, k, 0)
                        for k in ("input_tokens", "output_tokens",
                                  "cache_creation_input_tokens",
                                  "cache_read_input_tokens")
                    })
                return resp
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


# ---------------------------------------------------------------------------
# MockLLM (P2.1)
#
# A deterministic, API-key-free adapter for CI and developer smoke tests.
# Fixtures are keyed by (sha256 of system prompt, sha256 of user prompt);
# misses fall back to any configured ``default`` entry, then to a
# contract-preserving stub. ``complete_json`` returns parsed JSON like the
# real adapter does.
#
# Fixture file format (JSON or YAML):
#
#   [
#     {
#       "match": {
#         "system_contains": "You extract atomic claims",
#         "user_contains": "tesla"
#       },
#       "response": "<string OR object — object serializes to JSON>"
#     },
#     {
#       "match": {
#         "system_hash": "<sha256 hex>",
#         "user_hash":  "<sha256 hex>"
#       },
#       "response": {...}
#     }
#   ]
#
# Two equivalent matcher styles are supported so hand-authored fixtures can
# use substring matching (readable) while recorded fixtures can pin exact
# hashes (airtight).
# ---------------------------------------------------------------------------


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class MockLLM:
    """Deterministic offline LLM adapter driven by a fixtures file.

    Accepts the same two methods as :class:`LLM`. Loads fixtures from the
    path passed to the constructor (YAML or JSON; YAML requires PyYAML).
    """

    def __init__(self, fixtures_path: str | os.PathLike, model: str = "mock"):
        self.model = model
        self.calls = 0
        self.fixtures_path = Path(fixtures_path)
        self._fixtures: list[dict] = self._load(self.fixtures_path)
        # Keep the same surface as LLM so callers can't tell the difference
        # when they only touch .complete / .complete_json.
        self.max_attempts = 1
        self.backoff_base = 0.0
        self.usage = Usage()  # canned responses cost nothing

    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.is_file():
            raise FileNotFoundError(f"MockLLM fixtures not found: {path}")
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "MockLLM YAML fixtures require PyYAML "
                    "(`pip install pyyaml`); use .json instead"
                ) from e
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(
                f"MockLLM fixtures must be a list at top level, got "
                f"{type(data).__name__}"
            )
        return data

    def _match(self, system: str, user: str) -> Optional[dict]:
        """Find the first fixture whose ``match`` block is satisfied."""
        s_hash = _sha256(system)
        u_hash = _sha256(user)
        default: Optional[dict] = None
        for entry in self._fixtures:
            match = entry.get("match") or {}
            if match.get("default"):
                default = entry
                continue
            sh = match.get("system_hash")
            uh = match.get("user_hash")
            if sh and sh != s_hash:
                continue
            if uh and uh != u_hash:
                continue
            sc = match.get("system_contains")
            uc = match.get("user_contains")
            if sc and sc not in system:
                continue
            if uc and uc not in user:
                continue
            return entry
        return default

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        self.calls += 1
        entry = self._match(system, user)
        if entry is None:
            log("mockllm_miss", level="warning",
                system_hash=_sha256(system), user_hash=_sha256(user))
            # Contract-preserving stub: UNKNOWN for prose, empty JSON
            # structure for complete_json's re-parse path.
            return "UNKNOWN"
        resp = entry.get("response")
        if isinstance(resp, str):
            return resp
        return json.dumps(resp, ensure_ascii=False)

    def complete_json(
        self, system: str, user: str, max_tokens: int = 4096,
    ) -> list | dict:
        self.calls += 1
        entry = self._match(system, user)
        if entry is None:
            log("mockllm_miss_json", level="warning",
                system_hash=_sha256(system), user_hash=_sha256(user))
            # UNGROUNDED-safe empty default: empty list is accepted by
            # extractors and detectors as "no items found".
            return []
        resp = entry.get("response")
        if isinstance(resp, (list, dict)):
            return resp
        if isinstance(resp, str):
            return _parse_json(resp)
        raise ValueError(
            f"MockLLM fixture response must be str/list/dict, got "
            f"{type(resp).__name__}"
        )


def from_env(model: Optional[str] = None) -> "LLM | MockLLM":
    """Construct an LLM from environment: honours ``ALEPH_LLM_FIXTURES`` so
    that setting it swaps in the MockLLM transparently for every caller."""
    fixtures = os.environ.get("ALEPH_LLM_FIXTURES")
    if fixtures:
        return MockLLM(fixtures)
    return LLM(model=model or DEFAULT_MODEL)
