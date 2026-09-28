"""Pluggable retrieval: the ``Retriever`` protocol and its implementations.

``query.query()`` depends on this protocol, not on ``Store.search_claims``
directly. Swapping in BM25 / FTS / embeddings no longer requires rewriting
``Store``.

Implementations shipped here:

- ``KeywordRetriever`` wraps ``Store.search_claims`` (substring LIKE over
  triples). The lowest-common-denominator backend; always available.
- ``FTSRetriever`` uses SQLite FTS5 (BM25, word-tokenized, stopword-aware).
  The default when the store advertises ``fts_enabled``.
- ``EmbeddingRetriever`` (P2.3) uses ``sentence-transformers`` to compute
  dense embeddings of claim triples and ranks by cosine similarity to the
  query. Vectors are cached in a new ``claim_embeddings`` table keyed by
  ``(claim_id, model_name)`` and recomputed lazily when the claim text
  changes. This is an *optional* dependency; importing the class without
  the package raises a clear ``RuntimeError``.

Selecting a retriever:

- Code: pass ``retriever=MyRetriever(store)`` to ``query.query()``.
- Env: set ``ALEPH_RETRIEVER=fts`` | ``keyword`` | ``embedding`` and let
  ``default_retriever(store)`` pick. Defaults to FTS when available.

Writing a new retriever
-----------------------

Implement the single-method ``Retriever`` protocol::

    class MyRetriever:
        def search(self, keywords: list[str], limit: int = 30) -> list[Row]:
            ...

Each returned row must expose, at minimum: ``id``, ``subject``,
``predicate``, ``object``, ``confidence``, ``source_id``, ``span_text``.
Returning ``sqlite3.Row`` objects (as the built-ins do) is the simplest
path; any mapping/Row-like object that supports ``row["..."]`` access also
works. Ranking is the retriever's job — the rest of the pipeline treats
the list as ordered.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from typing import Optional, Protocol

from .db import Store, _statuses_sql


class Retriever(Protocol):
    """Returns ranked claim rows for a keyword query.

    Each returned row must expose at minimum the columns used by the synthesizer:
    ``id``, ``subject``, ``predicate``, ``object``, ``confidence``,
    ``source_id``, and ``span_text``. The rest of the pipeline is agnostic
    about ranking, so a new retriever only has to produce an ordered list.
    """

    def search(self, keywords: list[str], limit: int = 30) -> list[sqlite3.Row]:
        """Return active claims only. Retrievers that also accept an
        ``include_retracted: bool`` keyword support the ``include_retracted``
        query context; ``query()`` passes it only when a context asks."""
        ...


class KeywordRetriever:
    """LIKE-substring keyword search with count-based ranking.

    When ``expand_aliases=True`` (WS-E.1), the keyword list is augmented with
    every alias-from predicate that resolves to one of the supplied keywords
    in ``domain``. Default behavior is unchanged so existing callers continue
    to work.
    """

    def __init__(
        self, store: Store, *,
        expand_aliases: bool = False,
        domain: Optional[str] = None,
    ):
        self.store = store
        self.expand_aliases = expand_aliases
        self.domain = domain

    def search(
        self, keywords: list[str], limit: int = 30, *,
        include_retracted: bool = False,
    ) -> list[sqlite3.Row]:
        return self.store.search_claims(
            keywords, limit=limit,
            expand_aliases=self.expand_aliases, domain=self.domain,
            include_retracted=include_retracted,
        )


class FTSRetriever:
    """FTS5-based retrieval ranked by BM25.

    Better than the keyword retriever at: word-boundary matching (won't match
    "ion" inside "million"), stopword handling, and multi-term scoring. Requires
    the store's FTS5 index to be initialized (see ``Store.fts_enabled``).

    Mirrors :class:`KeywordRetriever` for the ``expand_aliases`` / ``domain``
    kwargs (WS-E.1).
    """

    def __init__(
        self, store: Store, *,
        expand_aliases: bool = False,
        domain: Optional[str] = None,
    ):
        self.store = store
        self.expand_aliases = expand_aliases
        self.domain = domain

    def search(
        self, keywords: list[str], limit: int = 30, *,
        include_retracted: bool = False,
    ) -> list[sqlite3.Row]:
        return self.store.search_claims_fts(
            keywords, limit=limit,
            expand_aliases=self.expand_aliases, domain=self.domain,
            include_retracted=include_retracted,
        )


# ---------------------------------------------------------------------------
# EmbeddingRetriever (P2.3)
#
# Stores dense vectors in an auxiliary table; caches per-(claim_id, model)
# so vectors don't regenerate on every query. The table is created lazily
# the first time the retriever is constructed, keeping the core schema
# free of a sentence-transformers dependency.
# ---------------------------------------------------------------------------

_EMBEDDING_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS claim_embeddings (
    claim_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    triple_sha256 TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (claim_id, model_name)
);
CREATE INDEX IF NOT EXISTS idx_claim_embeddings_model
    ON claim_embeddings(model_name);
"""


def _triple_hash(subject: str, predicate: str, object_: str) -> str:
    """Hash of the triple so a changed claim invalidates its vector."""
    raw = f"{subject}\x1f{predicate}\x1f{object_}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EmbeddingRetriever:
    """Dense-embedding retrieval with cached per-claim vectors.

    Requires ``sentence-transformers``. The default model is multilingual
    so it works with the English + Italian corpora the project targets.
    Pass ``model=`` to override. Vectors are cached in the store's
    ``claim_embeddings`` table and transparently re-encoded when a claim's
    triple text changes.
    """

    _DEFAULT_MODEL = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    def __init__(
        self,
        store: Store,
        *,
        model: str = _DEFAULT_MODEL,
        top_pool: int = 500,
    ):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "EmbeddingRetriever requires sentence-transformers "
                "(`pip install sentence-transformers`). "
                "Use FTSRetriever or KeywordRetriever if you don't want "
                "the extra dependency."
            ) from e

        self.store = store
        self.model_name = model
        self.top_pool = top_pool
        # Import lazily so the module is importable even without the
        # optional package — only constructing the retriever pays the cost.
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)
        self._ensure_table()
        try:
            import numpy  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "EmbeddingRetriever requires numpy"
            ) from e

    def _ensure_table(self) -> None:
        self.store.conn.executescript(_EMBEDDING_TABLE_SCHEMA)
        self.store.conn.commit()

    def _encode(self, text: str):
        import numpy as np
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")

    def _vector_for(self, claim) -> bytes:
        """Return cached bytes for a claim, (re)encoding if stale."""
        import numpy as np
        th = _triple_hash(claim["subject"], claim["predicate"], claim["object"])
        row = self.store.conn.execute(
            "SELECT triple_sha256, vector FROM claim_embeddings "
            "WHERE claim_id = ? AND model_name = ?",
            (claim["id"], self.model_name),
        ).fetchone()
        if row is not None and row["triple_sha256"] == th:
            return row["vector"]
        triple = f"{claim['subject']} {claim['predicate']} {claim['object']}"
        vec = self._encode(triple)
        blob = vec.tobytes()
        with self.store.tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO claim_embeddings "
                "(claim_id, model_name, triple_sha256, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (claim["id"], self.model_name, th, blob, time.time()),
            )
        return blob

    def search(
        self, keywords: list[str], limit: int = 30, *,
        include_retracted: bool = False,
    ) -> list[sqlite3.Row]:
        if not keywords:
            return []
        import numpy as np
        query_vec = self._encode(" ".join(keywords))

        # Candidate pool: take the top ``top_pool`` most recent active claims
        # so encoding cost stays bounded on large stores. Scored against the
        # full query vector. For smaller stores this is effectively "all".
        rows = self.store.conn.execute(
            "SELECT c.*, "
            "  SUBSTR(s.content, c.span_start + 1, c.span_end - c.span_start) "
            "    AS span_text "
            "FROM claims c JOIN sources s ON c.source_id = s.id "
            f"WHERE c.status IN ({_statuses_sql(include_retracted)}) "
            "ORDER BY c.extracted_at DESC LIMIT ?",
            (self.top_pool,),
        ).fetchall()
        if not rows:
            return []

        vectors = np.stack([
            np.frombuffer(self._vector_for(r), dtype="float32")
            for r in rows
        ])
        scores = vectors @ query_vec  # cosine since both are unit-normalized
        order = np.argsort(-scores)[:limit]
        return [rows[int(i)] for i in order]


def default_retriever(store: Store) -> Retriever:
    """Pick the best retriever available for this store.

    Respects the ``ALEPH_RETRIEVER`` env var (``keyword`` | ``fts`` |
    ``embedding``). Falls back to FTS5 when the store advertises it, then
    to keyword. Retrievers are constructed lazily — nothing pays the
    embedding-model import cost unless the caller asks for it.
    """
    pick = os.environ.get("ALEPH_RETRIEVER", "").lower() or None
    if pick == "embedding":
        return EmbeddingRetriever(store)
    if pick == "keyword":
        return KeywordRetriever(store)
    if pick == "fts":
        return FTSRetriever(store)
    if getattr(store, "fts_enabled", False):
        return FTSRetriever(store)
    return KeywordRetriever(store)
