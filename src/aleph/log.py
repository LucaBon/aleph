"""Structured logging for Aleph: one JSON object per line on stderr.

Aleph's correctness depends on operator introspection — the worst outcomes
(silently dropped claims, stale caches, verifier errors coerced to UNGROUNDED)
are invisible without this. Keep the surface tiny: one `log(event, **fields)`
helper, one `set_level` escape hatch. Default level is WARNING so normal runs
stay quiet; set `ALEPH_LOG=info` or `ALEPH_LOG=debug` for verbose traces.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

_LOGGER_NAME = "aleph"
_DEFAULT_LEVEL = os.environ.get("ALEPH_LOG", "warning").upper()

_logger = logging.getLogger(_LOGGER_NAME)
_logger.setLevel(getattr(logging, _DEFAULT_LEVEL, logging.WARNING))
_logger.propagate = False
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)


def set_level(level: str) -> None:
    """Override the log level at runtime (e.g. from a test)."""
    _logger.setLevel(getattr(logging, level.upper(), logging.WARNING))


def log(event: str, level: str = "info", **fields) -> None:
    """Emit a single JSONL event. Fields are arbitrary structured context."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    if not _logger.isEnabledFor(lvl):
        return
    payload = {"t": round(time.time(), 3), "level": level, "event": event, **fields}
    _logger.log(lvl, json.dumps(payload, default=str, ensure_ascii=False))
