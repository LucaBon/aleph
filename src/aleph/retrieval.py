"""Pluggable retrieval: the `Retriever` protocol and two implementations.

`query.query()` depends on this protocol, not on `Store.search_claims` directly.
Swapping in BM25 / FTS / embeddings no longer requires rewriting `Store`.

- `KeywordRetriever` wraps `Store.search_claims` (substring LIKE over triples).
- `FTSRetriever` uses SQLite FTS5 (BM25, word-tokenized, stopword-aware).
- `default_retriever(store)` picks FTS if the store reports FTS5 is enabled.
"""
from __future__ import annotations

import sqlite3
from typing import Protocol

from .db import Store


class Retriever(Protocol):
    """Returns ranked claim rows for a keyword query.

    Each returned row must expose at minimum the columns used by the synthesizer:
    `id`, `subject`, `predicate`, `object`, `confidence`, and `span_text`.
    """

    def search(self, keywords: list[str], limit: int = 30) -> list[sqlite3.Row]:
        ...


class KeywordRetriever:
    """LIKE-substring keyword search with count-based ranking."""

    def __init__(self, store: Store):
        self.store = store

    def search(self, keywords: list[str], limit: int = 30) -> list[sqlite3.Row]:
        return self.store.search_claims(keywords, limit=limit)


class FTSRetriever:
    """FTS5-based retrieval ranked by BM25.

    Better than the keyword retriever at: word-boundary matching (won't match
    "ion" inside "million"), stopword handling, and multi-term scoring. Requires
    the store's FTS5 index to be initialized (see `Store.fts_enabled`).
    """

    def __init__(self, store: Store):
        self.store = store

    def search(self, keywords: list[str], limit: int = 30) -> list[sqlite3.Row]:
        return self.store.search_claims_fts(keywords, limit=limit)


def default_retriever(store: Store) -> Retriever:
    """Pick the best retriever available for this store.

    Prefers FTS5 when the store advertises it; falls back to keyword otherwise
    so callers don't have to branch.
    """
    if getattr(store, "fts_enabled", False):
        return FTSRetriever(store)
    return KeywordRetriever(store)
