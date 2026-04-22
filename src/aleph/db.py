"""Persistent store for sources, claims, contradictions, view cache, and aliases.

The schema is the whole architecture in one place:
- sources: immutable, hashed. the ground truth.
- claims: atomic propositions that point back to exact source spans.
- subject_aliases: maps raw subject strings to canonical forms.
- contradictions: explicit nodes when two claims conflict.
- view_cache: generated prose answers, invalidated when claims change.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .log import log


def normalize_subject(s: str) -> str:
    """Deterministic subject normalization: case, whitespace, simple plurals.

    Fast, no LLM. Catches the easy cases so we only need an alias table for the
    genuinely hard ones (e.g. "Model S" vs "Tesla Model S").

    Examples:
        "Tesla Batteries" -> "tesla battery"
        "NCA cells" -> "nca cell"
        "lithium-ion packs" -> "lithium-ion pack"
    """
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # keep letters, digits, hyphens, whitespace
    s = re.sub(r"[^\w\s\-]", "", s)
    words = s.split()
    if not words:
        return ""
    last = words[-1]
    # very simple English plural->singular on the last word
    if len(last) > 4 and last.endswith("ies"):
        words[-1] = last[:-3] + "y"
    elif len(last) > 4 and last.endswith("sses"):
        words[-1] = last[:-2]  # glasses -> glass
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss") and not last.endswith("us"):
        words[-1] = last[:-1]
    return " ".join(words)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    ingested_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',  -- active | superseded | contradicted
    superseded_by INTEGER REFERENCES claims(id),
    extracted_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject);
CREATE INDEX IF NOT EXISTS idx_claims_predicate ON claims(predicate);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_source ON claims(source_id);

CREATE TABLE IF NOT EXISTS subject_aliases (
    alias_from TEXT PRIMARY KEY,   -- normalized raw subject
    canonical_to TEXT NOT NULL,    -- normalized canonical subject
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_aliases_canonical ON subject_aliases(canonical_to);

CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    claim_b_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'open',  -- open | resolved
    resolved_to INTEGER REFERENCES claims(id),
    detected_at REAL NOT NULL,
    UNIQUE(claim_a_id, claim_b_id)
);

CREATE TABLE IF NOT EXISTS view_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL UNIQUE,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    claim_ids TEXT NOT NULL,  -- JSON list of claim ids used
    generated_at REAL NOT NULL
);
"""


def _init_fts(conn: sqlite3.Connection) -> bool:
    """Set up FTS5 virtual table + triggers + backfill. Returns True on success.

    FTS5 is bundled with SQLite in modern builds; if this fails (ancient or
    stripped build), we return False and the store falls back to LIKE-based
    keyword search.
    """
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5("
            "subject, predicate, object, "
            "content='claims', content_rowid='id')"
        )
    except sqlite3.OperationalError:
        return False
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS claims_ai AFTER INSERT ON claims BEGIN
            INSERT INTO claims_fts(rowid, subject, predicate, object)
            VALUES (new.id, new.subject, new.predicate, new.object);
        END;
        CREATE TRIGGER IF NOT EXISTS claims_ad AFTER DELETE ON claims BEGIN
            INSERT INTO claims_fts(claims_fts, rowid, subject, predicate, object)
            VALUES ('delete', old.id, old.subject, old.predicate, old.object);
        END;
        CREATE TRIGGER IF NOT EXISTS claims_au AFTER UPDATE ON claims BEGIN
            INSERT INTO claims_fts(claims_fts, rowid, subject, predicate, object)
            VALUES ('delete', old.id, old.subject, old.predicate, old.object);
            INSERT INTO claims_fts(rowid, subject, predicate, object)
            VALUES (new.id, new.subject, new.predicate, new.object);
        END;
    """)
    # backfill any pre-existing rows that are missing from the FTS index
    # (happens when upgrading a store that was created before FTS existed)
    conn.execute(
        "INSERT INTO claims_fts(rowid, subject, predicate, object) "
        "SELECT id, subject, predicate, object FROM claims "
        "WHERE id NOT IN (SELECT rowid FROM claims_fts)"
    )
    return True


def _invalidate_cache_for_claims(
    cx: sqlite3.Connection, claim_ids, cause: str = "unspecified"
) -> int:
    """Delete view_cache rows whose claim_ids JSON intersects the given set.

    Runs inside the caller's transaction. Mirrors the walk used by remove_source.
    `cause` is only used for logging (e.g. "supersede", "alias").
    """
    ids = set(claim_ids)
    if not ids:
        return 0
    rows = cx.execute("SELECT id, claim_ids FROM view_cache").fetchall()
    to_drop = [
        row["id"] for row in rows
        if ids.intersection(set(json.loads(row["claim_ids"])))
    ]
    if not to_drop:
        return 0
    cx.executemany(
        "DELETE FROM view_cache WHERE id = ?", [(i,) for i in to_drop]
    )
    log(
        "cache_invalidated",
        level="info",
        cause=cause,
        claim_ids=sorted(ids),
        dropped=len(to_drop),
    )
    return len(to_drop)


class Store:
    """Thin wrapper around a SQLite database. Everything a CLI needs lives here."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets readers and writers coexist; busy_timeout makes concurrent
        # writers wait instead of immediately erroring with `database is locked`.
        # Both together are the minimum viable multi-agent story for SQLite.
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.executescript(SCHEMA)
        self.fts_enabled = _init_fts(self.conn)
        self.conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---------- sources ----------

    def add_source(self, path: str, content: str) -> Optional[int]:
        """Insert a source. Returns source_id, or None if already ingested (by hash)."""
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self.tx() as cx:
            existing = cx.execute("SELECT id FROM sources WHERE sha256 = ?", (sha,)).fetchone()
            if existing:
                return None
            cur = cx.execute(
                "INSERT INTO sources (path, sha256, content, ingested_at) VALUES (?, ?, ?, ?)",
                (path, sha, content, time.time()),
            )
            return cur.lastrowid

    def get_source(self, source_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()

    def list_sources(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM sources ORDER BY ingested_at DESC").fetchall()

    def remove_source(self, source_id: int) -> int:
        """Cascade-deletes claims and any view_cache entries referencing them."""
        with self.tx() as cx:
            # invalidate cached views that used any claim from this source
            claim_rows = cx.execute(
                "SELECT id FROM claims WHERE source_id = ?", (source_id,)
            ).fetchall()
            claim_ids = {r["id"] for r in claim_rows}
            if claim_ids:
                all_cached = cx.execute("SELECT id, claim_ids FROM view_cache").fetchall()
                to_drop = [
                    row["id"] for row in all_cached
                    if claim_ids.intersection(set(json.loads(row["claim_ids"])))
                ]
                if to_drop:
                    cx.executemany(
                        "DELETE FROM view_cache WHERE id = ?", [(i,) for i in to_drop]
                    )
            cur = cx.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            return cur.rowcount

    # ---------- claims ----------

    def resolve_subject(self, raw: str) -> str:
        """Normalize then resolve through the alias table (transitive, with cycle guard)."""
        s = normalize_subject(raw)
        seen = {s}
        for _ in range(10):  # cycle guard
            row = self.conn.execute(
                "SELECT canonical_to FROM subject_aliases WHERE alias_from = ?", (s,)
            ).fetchone()
            if not row:
                return s
            s = row["canonical_to"]
            if s in seen:
                return s
            seen.add(s)
        return s

    def add_alias(self, from_subject: str, to_subject: str) -> dict:
        """Record that from_subject -> to_subject, and rewrite existing claims.

        Returns {'claims_rewritten': N, 'cache_cleared': bool, ...}. If adding
        the alias would create a resolution cycle (e.g. A->B already exists
        and we're trying to add B->A), the alias is rejected with
        `note: 'would-create-cycle'`.
        """
        f = normalize_subject(from_subject)
        t = normalize_subject(to_subject)
        if not f or not t or f == t:
            return {"claims_rewritten": 0, "cache_cleared": False, "note": "no-op"}
        # cycle check: if resolving `t` through the current alias chain ever
        # reaches `f`, adding f -> t would make resolve(f) loop. Reject.
        if self.resolve_subject(t) == f:
            return {
                "claims_rewritten": 0,
                "cache_cleared": False,
                "note": "would-create-cycle",
            }
        with self.tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO subject_aliases (alias_from, canonical_to, created_at) "
                "VALUES (?, ?, ?)",
                (f, t, time.time()),
            )
            affected = [
                r["id"] for r in cx.execute(
                    "SELECT id FROM claims WHERE subject = ?", (f,)
                ).fetchall()
            ]
            cur = cx.execute(
                "UPDATE claims SET subject = ? WHERE subject = ?", (t, f)
            )
            rewritten = cur.rowcount
            cache_cleared = (
                _invalidate_cache_for_claims(cx, affected, cause="alias")
                if affected else 0
            )
        return {"claims_rewritten": rewritten, "cache_cleared": cache_cleared > 0}

    def list_aliases(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM subject_aliases ORDER BY canonical_to, alias_from"
        ).fetchall()

    def add_claim(
        self,
        source_id: int,
        subject: str,
        predicate: str,
        object_: str,
        span_start: int,
        span_end: int,
        confidence: float,
    ) -> int:
        canonical_subject = self.resolve_subject(subject)
        canonical_predicate = normalize_subject(predicate)  # same rules work fine
        with self.tx() as cx:
            cur = cx.execute(
                """
                INSERT INTO claims (source_id, subject, predicate, object,
                                    span_start, span_end, confidence, extracted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, canonical_subject, canonical_predicate, object_.strip(),
                 span_start, span_end, confidence, time.time()),
            )
            return cur.lastrowid

    def get_claim(self, claim_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()

    def get_span_text(self, claim_id: int) -> Optional[str]:
        """Fetch the exact source text the claim points at."""
        row = self.conn.execute(
            """
            SELECT s.content, c.span_start, c.span_end
            FROM claims c JOIN sources s ON c.source_id = s.id
            WHERE c.id = ?
            """,
            (claim_id,),
        ).fetchone()
        if not row:
            return None
        return row["content"][row["span_start"] : row["span_end"]]

    def search_claims(self, keywords: list[str], limit: int = 30) -> list[sqlite3.Row]:
        """Simple keyword search across subject/predicate/object.

        Ranks by number of matching keywords, then confidence, then recency.
        Returns each row with an inline `span_text` column (joined from the
        source) so callers don't need to round-trip per claim for spans.
        """
        if not keywords:
            return []
        kws = [k.lower() for k in keywords if len(k) > 2]
        if not kws:
            return []
        # build a CASE expression that counts matches
        match_exprs = " + ".join(
            ["(CASE WHEN (c.subject||' '||c.predicate||' '||c.object) LIKE ? "
             "THEN 1 ELSE 0 END)" for _ in kws]
        )
        params = [f"%{k}%" for k in kws]
        sql = f"""
            SELECT c.*,
                   ({match_exprs}) AS hits,
                   SUBSTR(s.content, c.span_start + 1, c.span_end - c.span_start)
                       AS span_text
            FROM claims c JOIN sources s ON c.source_id = s.id
            WHERE c.status = 'active' AND ({match_exprs}) > 0
            ORDER BY hits DESC, c.confidence DESC, c.extracted_at DESC
            LIMIT ?
        """
        return self.conn.execute(sql, params + params + [limit]).fetchall()

    def search_claims_fts(self, keywords: list[str], limit: int = 30) -> list[sqlite3.Row]:
        """BM25-ranked search over the FTS5 index of claim triples.

        Falls back to an empty list (not an exception) if FTS5 isn't enabled
        on this store, so callers can safely call it unconditionally and then
        inspect `fts_enabled` if they want to distinguish.
        """
        if not self.fts_enabled or not keywords:
            return []
        cleaned = [re.sub(r"[^\w-]", "", k.lower()) for k in keywords]
        cleaned = [k for k in cleaned if len(k) > 2]
        if not cleaned:
            return []
        # double-quote each term to avoid FTS5 interpreting punctuation/syntax
        match_query = " OR ".join(f'"{k}"' for k in cleaned)
        sql = """
            SELECT c.*,
                   SUBSTR(s.content, c.span_start + 1, c.span_end - c.span_start)
                       AS span_text,
                   bm25(claims_fts) AS bm25_score
            FROM claims_fts
            JOIN claims c ON c.id = claims_fts.rowid
            JOIN sources s ON c.source_id = s.id
            WHERE claims_fts MATCH ? AND c.status = 'active'
            ORDER BY bm25(claims_fts), c.confidence DESC, c.extracted_at DESC
            LIMIT ?
        """
        try:
            return self.conn.execute(sql, (match_query, limit)).fetchall()
        except sqlite3.OperationalError:
            # malformed FTS query (e.g. all tokens stripped by tokenizer) —
            # return no hits rather than erroring
            return []

    def all_active_claims(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM claims WHERE status = 'active' ORDER BY subject, predicate"
        ).fetchall()

    def supersede_claim(self, old_id: int, new_id: int) -> None:
        with self.tx() as cx:
            cx.execute(
                "UPDATE claims SET status = 'superseded', superseded_by = ? WHERE id = ?",
                (new_id, old_id),
            )
            # any cached view that cited the now-superseded claim is stale
            _invalidate_cache_for_claims(cx, [old_id], cause="supersede")

    # ---------- contradictions ----------

    def add_contradiction(self, a_id: int, b_id: int) -> Optional[int]:
        lo, hi = sorted([a_id, b_id])
        with self.tx() as cx:
            existing = cx.execute(
                "SELECT id FROM contradictions WHERE claim_a_id = ? AND claim_b_id = ?",
                (lo, hi),
            ).fetchone()
            if existing:
                return None
            cur = cx.execute(
                "INSERT INTO contradictions (claim_a_id, claim_b_id, detected_at) VALUES (?, ?, ?)",
                (lo, hi, time.time()),
            )
            return cur.lastrowid

    def list_contradictions(self, only_open: bool = True) -> list[sqlite3.Row]:
        sql = "SELECT * FROM contradictions"
        if only_open:
            sql += " WHERE status = 'open'"
        sql += " ORDER BY detected_at DESC"
        return self.conn.execute(sql).fetchall()

    # ---------- view cache ----------

    def get_cached_view(self, query: str) -> Optional[sqlite3.Row]:
        qh = hashlib.sha256(query.strip().lower().encode()).hexdigest()
        return self.conn.execute(
            "SELECT * FROM view_cache WHERE query_hash = ?", (qh,)
        ).fetchone()

    def cache_view(self, query: str, response: str, claim_ids: list[int]) -> None:
        qh = hashlib.sha256(query.strip().lower().encode()).hexdigest()
        with self.tx() as cx:
            cx.execute(
                """
                INSERT OR REPLACE INTO view_cache (query_hash, query, response, claim_ids, generated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (qh, query, response, json.dumps(claim_ids), time.time()),
            )

    def clear_cache(self) -> int:
        with self.tx() as cx:
            cur = cx.execute("DELETE FROM view_cache")
            return cur.rowcount

    def stats(self) -> dict:
        return {
            "sources": self.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            "claims_active": self.conn.execute(
                "SELECT COUNT(*) FROM claims WHERE status = 'active'"
            ).fetchone()[0],
            "claims_superseded": self.conn.execute(
                "SELECT COUNT(*) FROM claims WHERE status = 'superseded'"
            ).fetchone()[0],
            "contradictions_open": self.conn.execute(
                "SELECT COUNT(*) FROM contradictions WHERE status = 'open'"
            ).fetchone()[0],
            "cached_views": self.conn.execute("SELECT COUNT(*) FROM view_cache").fetchone()[0],
        }

    def close(self) -> None:
        self.conn.close()
