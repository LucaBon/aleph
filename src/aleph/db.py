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
from typing import Callable, Iterator, Optional

from .fidelity import _context_window_v2
from .log import log
from .normalization import Normalizer, get_normalizer

# Default normalizer for the module-level back-compat helper. Per-store
# normalization goes through ``Store.normalize_subject``, which dispatches to
# whichever locale the store was opened with.
_DEFAULT_LOCALE = "en"
_DEFAULT_NORMALIZER: Normalizer = get_normalizer(_DEFAULT_LOCALE)


def normalize_subject(s: str) -> str:
    """English-locale normalization (module-level back-compat wrapper).

    Retained for call-sites that do not have a :class:`Store` in scope.
    Code that does have a store should prefer ``store.normalize_subject`` so
    that a non-English locale configured on the store applies correctly.
    """
    return _DEFAULT_NORMALIZER.normalize(s)

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
    status TEXT NOT NULL DEFAULT 'active',  -- active | superseded | retracted
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

-- Alias merge log: every alias-add is an event, and every claim whose subject
-- it rewrote is recorded with its old subject, so merges can be undone.
CREATE TABLE IF NOT EXISTS alias_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alias_from TEXT NOT NULL,
    canonical_to TEXT NOT NULL,
    previous_to TEXT,               -- prior target when an alias was overwritten
    created_at REAL NOT NULL,
    undone_at REAL
);

CREATE TABLE IF NOT EXISTS alias_rewrites (
    event_id INTEGER NOT NULL REFERENCES alias_events(id) ON DELETE CASCADE,
    claim_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    old_subject TEXT NOT NULL,
    PRIMARY KEY (event_id, claim_id)
);

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

CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    statement TEXT NOT NULL,
    inference_type TEXT NOT NULL DEFAULT 'summary',  -- summary | generalization | synthesis
    confidence REAL NOT NULL DEFAULT 0.7,
    status TEXT NOT NULL DEFAULT 'draft',
        -- draft | attested | active | stale | superseded | invalidated
    superseded_by INTEGER REFERENCES concepts(id),
    validation_verdict TEXT,   -- GROUNDED | PARTIAL | UNGROUNDED | null
    validation_reason TEXT,
    last_validated_at REAL,
    derived_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_concepts_subject ON concepts(subject);
CREATE INDEX IF NOT EXISTS idx_concepts_status ON concepts(status);

CREATE TABLE IF NOT EXISTS concept_supports (
    concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    claim_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'premise',
        -- premise | corroborating | qualifying | counterexample
    PRIMARY KEY (concept_id, claim_id)
);

CREATE INDEX IF NOT EXISTS idx_concept_supports_claim ON concept_supports(claim_id);

CREATE TABLE IF NOT EXISTS claim_conditions (
    claim_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    condition_claim_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'scope',
        -- sample | method | scope | limitation | assumption
    explicit INTEGER NOT NULL DEFAULT 1,  -- 0/1 bool (sqlite)
    confidence REAL NOT NULL DEFAULT 0.9,
    PRIMARY KEY (claim_id, condition_claim_id),
    CHECK (claim_id != condition_claim_id)
);

CREATE INDEX IF NOT EXISTS idx_claim_conditions_condition
    ON claim_conditions(condition_claim_id);

CREATE TABLE IF NOT EXISTS source_metadata (
    source_id INTEGER PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
    domain TEXT NOT NULL,                    -- legal | scientific | policy | corporate | generic
    metadata TEXT NOT NULL DEFAULT '{}',     -- JSON blob, schema per domain
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_source_metadata_domain ON source_metadata(domain);

CREATE TABLE IF NOT EXISTS contradiction_rules (
    contradiction_id INTEGER PRIMARY KEY
        REFERENCES contradictions(id) ON DELETE CASCADE,
    rule TEXT NOT NULL,
    applies_when TEXT,                     -- JSON predicate (optional)
    rationale_concept_id INTEGER REFERENCES concepts(id),
    decided_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_contradiction_rules_concept
    ON contradiction_rules(rationale_concept_id);

CREATE TABLE IF NOT EXISTS store_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- WS-E.1: domain-scoped predicate aliases
CREATE TABLE IF NOT EXISTS predicate_aliases (
    domain TEXT NOT NULL,            -- legal | scientific | policy
                                     -- | corporate | generic | *
    alias_from TEXT NOT NULL,        -- normalized predicate (raw form)
    canonical_to TEXT NOT NULL,      -- normalized canonical predicate
    created_at REAL NOT NULL,
    PRIMARY KEY (domain, alias_from)
);

CREATE INDEX IF NOT EXISTS idx_predicate_aliases_canonical
    ON predicate_aliases(canonical_to);

-- WS-E.2: predicate sense catalog
CREATE TABLE IF NOT EXISTS predicate_senses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical TEXT NOT NULL,
    sense_tag TEXT NOT NULL,
    domain TEXT NOT NULL,
    definition TEXT,
    parent_id INTEGER REFERENCES predicate_senses(id),
    inverse_id INTEGER REFERENCES predicate_senses(id),
    is_symmetric INTEGER NOT NULL DEFAULT 0,
    is_transitive INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE(canonical, sense_tag, domain)
);

CREATE INDEX IF NOT EXISTS idx_predicate_senses_canonical
    ON predicate_senses(canonical);
CREATE INDEX IF NOT EXISTS idx_predicate_senses_domain
    ON predicate_senses(domain);

-- WS-E.2: per-claim sense assignment with attributed provenance
CREATE TABLE IF NOT EXISTS claim_predicate_senses (
    claim_id INTEGER PRIMARY KEY REFERENCES claims(id) ON DELETE CASCADE,
    sense_id INTEGER NOT NULL REFERENCES predicate_senses(id),
    assigned_by TEXT NOT NULL,        -- agent | llm | human
    assigned_at REAL NOT NULL,
    confidence REAL NOT NULL,
    explicit INTEGER NOT NULL DEFAULT 0,  -- 0/1 bool (sqlite)
    rationale TEXT
);

CREATE INDEX IF NOT EXISTS idx_cps_sense
    ON claim_predicate_senses(sense_id);
"""


def _run_alters(conn: sqlite3.Connection) -> None:
    """Idempotently add columns via ALTER TABLE.

    Each ALTER is wrapped in try/except: if the column already exists SQLite
    raises OperationalError with 'duplicate column' in the message. We swallow
    that and move on. This is the only safe idempotent pattern for ALTERs on
    SQLite.
    """
    alters = [
        "ALTER TABLE contradictions ADD COLUMN kind TEXT DEFAULT 'categorical'",
        "ALTER TABLE contradictions ADD COLUMN disposition TEXT DEFAULT 'unresolved'",
        "ALTER TABLE contradictions ADD COLUMN disposition_at REAL",
        "ALTER TABLE contradictions ADD COLUMN candidate_disposition TEXT",
        "ALTER TABLE contradictions ADD COLUMN overlap_score REAL",
        "ALTER TABLE view_cache ADD COLUMN concept_ids TEXT NOT NULL DEFAULT '[]'",
        # P1.6: cross-subject contradiction escape valve. `cross_subject=1` marks a
        # pair whose subjects differ but which represents a legitimate doctrinal
        # tension (regime-supersedes, rule-limits-rule, doctrinal-cross-ref). The
        # justification is free text; relation_kind is a small vocabulary.
        "ALTER TABLE contradictions ADD COLUMN cross_subject INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE contradictions ADD COLUMN relation_kind TEXT",
        "ALTER TABLE contradictions ADD COLUMN justification TEXT",
        # P1.2: agent-driven promotion path for concepts. `attested` status is a
        # principled escape hatch for fully-agent-mode runs where no LLM validator
        # is available; the operator attests the concept is grounded, recorded
        # here with attestor + rationale and surfaced when the concept is cited.
        "ALTER TABLE concepts ADD COLUMN attested_by TEXT",
        "ALTER TABLE concepts ADD COLUMN attested_at REAL",
        "ALTER TABLE concepts ADD COLUMN attestation_rationale TEXT",
        # Retraction cascade: a resolved contradiction whose grounds disappear
        # is reopened, not left silently resolved. The prior disposition and
        # the cause are kept for audit; replicate bumps are recorded so they
        # can be reverted exactly.
        "ALTER TABLE contradictions ADD COLUMN reopened_at REAL",
        "ALTER TABLE contradictions ADD COLUMN reopened_cause TEXT",
        "ALTER TABLE contradictions ADD COLUMN reopened_from TEXT",
        "ALTER TABLE contradictions ADD COLUMN confidence_delta_a REAL",
        "ALTER TABLE contradictions ADD COLUMN confidence_delta_b REAL",
        # Why a claim is `retracted`: 'source' (source-retract) or
        # 'disposition' (a `retracted` contradiction disposition dropped it).
        # source-unretract only revives the former.
        "ALTER TABLE claims ADD COLUMN retracted_cause TEXT",
    ]
    for stmt in alters:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                continue
            raise


REVIEW_ITEM_TYPES = ("claim", "contradiction", "concept", "alias")
REVIEW_DECISIONS = ("accepted", "rejected")


def _add_column(conn: sqlite3.Connection, stmt: str) -> None:
    try:
        conn.execute(stmt)
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise


# One row per item under review. Exactly one target column is set, matching
# item_type; the foreign keys drop a claim/contradiction/concept review with
# its target. Alias reviews are closed as `obsolete` when the alias is undone.
_REVIEW_QUEUE_DDL = """
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type TEXT NOT NULL,   -- claim | contradiction | concept | alias
    claim_id INTEGER REFERENCES claims(id) ON DELETE CASCADE,
    contradiction_id INTEGER REFERENCES contradictions(id) ON DELETE CASCADE,
    concept_id INTEGER REFERENCES concepts(id) ON DELETE CASCADE,
    alias_from TEXT,
    reason TEXT NOT NULL,      -- fidelity | manual | ...
    details TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'open',
        -- open | accepted | rejected | obsolete
    created_at REAL NOT NULL,
    resolved_at REAL,
    resolved_by TEXT,
    resolution_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status);
CREATE INDEX IF NOT EXISTS idx_review_claim ON review_queue(claim_id);
CREATE INDEX IF NOT EXISTS idx_review_contradiction
    ON review_queue(contradiction_id);
CREATE INDEX IF NOT EXISTS idx_review_concept ON review_queue(concept_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_review_one_open
    ON review_queue(item_type, IFNULL(claim_id, 0), IFNULL(contradiction_id, 0),
                    IFNULL(concept_id, 0), IFNULL(alias_from, ''), reason)
    WHERE status = 'open';
"""


def _migrate_v2_proposition_context_review(conn: sqlite3.Connection) -> None:
    """Claims gain a ``proposition`` (the claim as a sentence; the triple
    becomes an index) and a context window around the span, which the
    fidelity checker reads. Existing claims get a computed context window;
    their proposition stays NULL rather than being fabricated from the triple.
    Adds the review queue.
    """
    _add_column(conn, "ALTER TABLE claims ADD COLUMN proposition TEXT")
    _add_column(conn, "ALTER TABLE claims ADD COLUMN context_start INTEGER")
    _add_column(conn, "ALTER TABLE claims ADD COLUMN context_end INTEGER")
    rows = conn.execute(
        "SELECT c.id, c.span_start, c.span_end, s.content FROM claims c "
        "JOIN sources s ON s.id = c.source_id"
    ).fetchall()
    for claim_id, start, end, content in rows:
        c_start, c_end = _context_window_v2(content, start, end)
        conn.execute(
            "UPDATE claims SET context_start = ?, context_end = ? WHERE id = ?",
            (c_start, c_end, claim_id),
        )
    # executescript would COMMIT mid-migration; run statements one by one so
    # they stay inside _migrate's transaction.
    for stmt in _REVIEW_QUEUE_DDL.split(";"):
        if stmt.strip():
            conn.execute(stmt)


# Numbered schema migrations. SCHEMA + _run_alters define version 1; each
# later change that alters what existing rows *mean* (not just a new nullable
# column) gets an entry here: (version, description, fn(conn)). Entries run in
# order, each in its own transaction, and bump `store_config.schema_version`.
# Never edit or reorder a released entry — append a new one.
MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (2, "claim proposition + context window; review queue",
     _migrate_v2_proposition_context_review),
]

SCHEMA_VERSION = max((v for v, _, _ in MIGRATIONS), default=1)


def _fidelity_text(proposition: Optional[str], predicate: str, object_: str) -> str:
    """What the fidelity checker reads for a claim: its proposition, else
    predicate + object. The subject is an index label ("aggravante art 577"),
    not text the span has to contain."""
    return proposition or f"{predicate} {object_}"


def view_cache_key(question: str, context: Optional[dict] = None) -> str:
    """The one definition of a view's cache key, shared by ``ask`` and the
    agent-mode ``view-get`` / ``view-cache`` commands."""
    raw = question.strip().lower() + "||" + json.dumps(context or {}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _statuses_sql(include_retracted: bool) -> str:
    """Claim statuses a search returns, as a SQL IN-list literal."""
    return "'active', 'retracted'" if include_retracted else "'active'"


class SchemaVersionError(RuntimeError):
    """The store was written by a newer aleph than this one."""


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring the store up to :data:`SCHEMA_VERSION`.

    A store with no recorded version was built by SCHEMA + _run_alters alone,
    which is version 1 by definition. A store recorded at a newer version than
    this code knows is refused: its rows may mean something this code would
    misread.
    """
    row = conn.execute(
        "SELECT value FROM store_config WHERE key = 'schema_version'"
    ).fetchone()
    current = int(row[0]) if row else 1
    if current > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"store schema_version is {current}, but this aleph only knows "
            f"up to {SCHEMA_VERSION}; upgrade aleph to open it"
        )
    for version, _desc, fn in sorted(MIGRATIONS, key=lambda m: m[0]):
        if version <= current:
            continue
        # An explicit BEGIN: in sqlite3's legacy transaction mode DDL (ALTER,
        # CREATE) would otherwise autocommit, leaving a half-applied migration.
        conn.commit()
        conn.execute("BEGIN")
        try:
            fn(conn)
            conn.execute(
                "INSERT OR REPLACE INTO store_config (key, value, updated_at) "
                "VALUES ('schema_version', ?, ?)",
                (str(version), time.time()),
            )
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
        current = version
    if row is None:
        conn.execute(
            "INSERT OR REPLACE INTO store_config (key, value, updated_at) "
            "VALUES ('schema_version', ?, ?)",
            (str(current), time.time()),
        )


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
    # Backfill a store created before FTS existed. `SELECT rowid FROM
    # claims_fts` reads the external content table (claims), not the index,
    # so compare against the index's own docsize table instead; a missing
    # entry would make the update trigger's 'delete' corrupt the index.
    indexed = conn.execute("SELECT count(*) FROM claims_fts_docsize").fetchone()[0]
    total = conn.execute("SELECT count(*) FROM claims").fetchone()[0]
    if indexed != total:
        conn.execute("INSERT INTO claims_fts(claims_fts) VALUES('rebuild')")
    return True


def _invalidate_cache_for_claims(
    cx: sqlite3.Connection, claim_ids, cause: str = "unspecified"
) -> int:
    """Delete view_cache rows whose claim_ids OR concept_ids JSON intersects the given set.

    Runs inside the caller's transaction. Mirrors the walk used by remove_source.
    `cause` is only used for logging (e.g. "supersede", "alias").
    """
    ids = set(claim_ids)
    if not ids:
        return 0
    rows = cx.execute("SELECT id, claim_ids, concept_ids FROM view_cache").fetchall()
    to_drop = []
    for row in rows:
        cached_claim_ids = set(json.loads(row["claim_ids"]))
        # Also check concept_ids column — concepts that reference these claims
        # may have been cached in views
        try:
            cached_concept_ids = set(json.loads(row["concept_ids"]))
        except (json.JSONDecodeError, TypeError):
            cached_concept_ids = set()
        if ids.intersection(cached_claim_ids):
            to_drop.append(row["id"])
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


def _invalidate_cache_for_concepts(
    cx: sqlite3.Connection, concept_ids, cause: str = "unspecified"
) -> int:
    """Delete view_cache rows whose concept_ids JSON intersects the given set.

    Runs inside the caller's transaction. Sibling of _invalidate_cache_for_claims.
    """
    ids = set(concept_ids)
    if not ids:
        return 0
    rows = cx.execute("SELECT id, concept_ids FROM view_cache").fetchall()
    to_drop = []
    for row in rows:
        try:
            cached = set(json.loads(row["concept_ids"]))
        except (json.JSONDecodeError, TypeError):
            cached = set()
        if ids.intersection(cached):
            to_drop.append(row["id"])
    if not to_drop:
        return 0
    cx.executemany(
        "DELETE FROM view_cache WHERE id = ?", [(i,) for i in to_drop]
    )
    log(
        "cache_invalidated",
        level="info",
        cause=cause,
        concept_ids=sorted(ids),
        dropped=len(to_drop),
    )
    return len(to_drop)



# ---------- retraction cascade ----------
#
# Every path that takes claims out of the active set (supersede, retract,
# remove, contradiction disposition) funnels through
# `_on_claims_deactivated_tx`, so the downstream effects are defined once:
#   1. cached views citing the claims are dropped;
#   2. concepts supported by them go stale (and views citing those concepts
#      are dropped, and contradictions citing those concepts as rationale
#      reopen);
#   3. claims that were superseded by a claim that is now retracted or
#      removed come back to active: the correction they yielded to is gone;
#   4. resolved contradictions whose resolution rested on the claims reopen.
# `check_invariants` checks exactly the states this cascade is meant to rule
# out, so the two must be kept in step.

_KEEP_DROP_DISPOSITIONS = ("supersede", "retracted")


def _retracted_source_ids(cx: sqlite3.Connection) -> set[int]:
    """Sources whose metadata marks them retracted (scientific domain)."""
    from .authority import is_retracted
    out = set()
    for row in cx.execute("SELECT source_id, domain, metadata FROM source_metadata"):
        try:
            meta = json.loads(row["metadata"])
        except (json.JSONDecodeError, TypeError):
            continue
        if is_retracted({"domain": row["domain"], "metadata": meta}):
            out.add(row["source_id"])
    return out


def _orphaned_supersessions(
    cx: sqlite3.Connection, removing: frozenset = frozenset(),
) -> list[int]:
    """Superseded claims whose supersession chain no longer ends in an active claim.

    A chain ends at the first claim that isn't `superseded`. If that claim is
    missing, retracted, or about to be deleted (`removing`), the supersession
    has lost its grounds. Claims in retracted sources are left alone: they're
    out of the active set either way.
    """
    rows = cx.execute(
        "SELECT id, source_id, status, superseded_by FROM claims"
    ).fetchall()
    by_id = {r["id"]: r for r in rows}
    retracted_sources = _retracted_source_ids(cx)
    out = []
    for r in rows:
        if r["status"] != "superseded" or r["id"] in removing:
            continue
        if r["source_id"] in retracted_sources:
            continue
        seen = {r["id"]}
        cur = r["superseded_by"]
        terminal_ok = False
        while cur is not None and cur not in seen:
            seen.add(cur)
            t = by_id.get(cur)
            if t is None or cur in removing:
                break
            if t["status"] == "superseded":
                cur = t["superseded_by"]
                continue
            terminal_ok = t["status"] == "active"
            break
        if not terminal_ok:
            out.append(r["id"])
    return out


def _revive_orphaned_supersessions_tx(
    cx: sqlite3.Connection, removing: frozenset = frozenset(), cause: str = "unspecified",
) -> list[int]:
    ids = _orphaned_supersessions(cx, removing)
    if ids:
        ph = ",".join("?" for _ in ids)
        cx.execute(
            f"UPDATE claims SET status = 'active', superseded_by = NULL WHERE id IN ({ph})",
            ids,
        )
        log("claims_revived", level="info", cause=cause, claim_ids=sorted(ids))
    return ids


def _contradiction_keep_id(cx: sqlite3.Connection, row) -> Optional[int]:
    """The claim a keep/drop resolution kept, or None if it can't be determined."""
    if row["resolved_to"] is not None:
        return row["resolved_to"]
    a, b = row["claim_a_id"], row["claim_b_id"]
    ca = cx.execute("SELECT status, superseded_by FROM claims WHERE id = ?", (a,)).fetchone()
    cb = cx.execute("SELECT status, superseded_by FROM claims WHERE id = ?", (b,)).fetchone()
    if row["disposition"] == "supersede":
        if ca is not None and ca["superseded_by"] == b:
            return b
        if cb is not None and cb["superseded_by"] == a:
            return a
    if row["disposition"] == "retracted":
        if ca is not None and ca["status"] == "retracted" and (cb is None or cb["status"] != "retracted"):
            return b
        if cb is not None and cb["status"] == "retracted" and (ca is None or ca["status"] != "retracted"):
            return a
    return None


def _is_resolved(row) -> bool:
    return row["status"] == "resolved" or (row["disposition"] or "unresolved") != "unresolved"


def _is_keep_drop(row) -> bool:
    return row["disposition"] in _KEEP_DROP_DISPOSITIONS or (
        (row["disposition"] or "unresolved") == "unresolved" and row["resolved_to"] is not None
    )


def _revert_replicate_bump_tx(cx: sqlite3.Connection, row) -> None:
    """Undo the confidence bump a `replicate` disposition recorded on ``row``.
    Callers must then clear or overwrite the recorded deltas."""
    if row["disposition"] != "replicate":
        return
    for claim_col, delta_col in (("claim_a_id", "confidence_delta_a"),
                                 ("claim_b_id", "confidence_delta_b")):
        delta = row[delta_col]
        if delta:
            cx.execute(
                "UPDATE claims SET confidence = MAX(confidence - ?, 0.0) WHERE id = ?",
                (delta, row[claim_col]),
            )


def _reopen_contradictions_tx(
    cx: sqlite3.Connection, contradiction_ids, cause: str,
) -> list[int]:
    """Set resolved contradictions back to open/unresolved and revert replicate bumps."""
    reopened = []
    for cid in contradiction_ids:
        row = cx.execute("SELECT * FROM contradictions WHERE id = ?", (cid,)).fetchone()
        if row is None or not _is_resolved(row):
            continue
        _revert_replicate_bump_tx(cx, row)
        cx.execute(
            "UPDATE contradictions SET status = 'open', disposition = 'unresolved', "
            "resolved_to = NULL, confidence_delta_a = NULL, confidence_delta_b = NULL, "
            "reopened_at = ?, reopened_cause = ?, reopened_from = ? WHERE id = ?",
            (time.time(), cause, row["disposition"] or "resolved", cid),
        )
        reopened.append(cid)
    if reopened:
        log("contradictions_reopened", level="info", cause=cause, contradiction_ids=reopened)
    return reopened


def _reopen_contradictions_for_claims_tx(
    cx: sqlite3.Connection, claim_ids, cause: str,
) -> list[int]:
    """Reopen resolved contradictions whose resolution depended on these claims.

    For keep/drop resolutions only the kept claim matters: the dropped claim
    leaving the active set is the resolution working as intended.
    """
    ids = set(claim_ids)
    if not ids:
        return []
    ph = ",".join("?" for _ in ids)
    rows = cx.execute(
        f"SELECT * FROM contradictions WHERE claim_a_id IN ({ph}) OR claim_b_id IN ({ph})",
        list(ids) + list(ids),
    ).fetchall()
    to_reopen = []
    for row in rows:
        if not _is_resolved(row):
            continue
        if _is_keep_drop(row):
            keep = _contradiction_keep_id(cx, row)
            if keep is not None and keep not in ids:
                continue
        to_reopen.append(row["id"])
    return _reopen_contradictions_tx(cx, to_reopen, cause)


def _reopen_contradictions_for_concepts_tx(
    cx: sqlite3.Connection, concept_ids, cause: str,
) -> list[int]:
    """Reopen resolved contradictions whose rationale concept lost its standing."""
    ids = list(set(concept_ids))
    if not ids:
        return []
    ph = ",".join("?" for _ in ids)
    rows = cx.execute(
        f"SELECT contradiction_id FROM contradiction_rules WHERE rationale_concept_id IN ({ph})",
        ids,
    ).fetchall()
    return _reopen_contradictions_tx(cx, [r["contradiction_id"] for r in rows], cause)


def check_invariants(cx: sqlite3.Connection) -> list[dict]:
    """Return every violation of the cascade invariants (empty list = clean).

    The invariant: nothing that is presented as current (a cached view, an
    active/attested concept, a resolved contradiction, a supersession)
    depends on a claim that is no longer active, or on a concept that is no
    longer active/attested.
    """
    violations: list[dict] = []
    claim_status = {
        r["id"]: r["status"] for r in cx.execute("SELECT id, status FROM claims")
    }
    concept_status = {
        r["id"]: r["status"] for r in cx.execute("SELECT id, status FROM concepts")
    }
    citable = ("active", "attested")

    for row in cx.execute("SELECT id, query, claim_ids, concept_ids FROM view_cache"):
        for cid in json.loads(row["claim_ids"] or "[]"):
            if claim_status.get(cid) != "active":
                violations.append({"kind": "view_cites_inactive_claim", "view_id": row["id"],
                                   "claim_id": cid, "claim_status": claim_status.get(cid)})
        try:
            concept_ids = json.loads(row["concept_ids"] or "[]")
        except (json.JSONDecodeError, TypeError):
            concept_ids = []
        for kid in concept_ids:
            if concept_status.get(kid) not in citable:
                violations.append({"kind": "view_cites_uncitable_concept", "view_id": row["id"],
                                   "concept_id": kid, "concept_status": concept_status.get(kid)})

    for cid, status in concept_status.items():
        if status not in citable:
            continue
        supports = [r["claim_id"] for r in cx.execute(
            "SELECT claim_id FROM concept_supports WHERE concept_id = ?", (cid,))]
        if not supports:
            violations.append({"kind": "concept_without_supports", "concept_id": cid})
        for sid in supports:
            if claim_status.get(sid) != "active":
                violations.append({"kind": "concept_supported_by_inactive_claim",
                                   "concept_id": cid, "claim_id": sid,
                                   "claim_status": claim_status.get(sid)})

    for row in cx.execute(
        "SELECT ct.*, cr.rationale_concept_id FROM contradictions ct "
        "LEFT JOIN contradiction_rules cr ON cr.contradiction_id = ct.id"
    ).fetchall():
        if not _is_resolved(row):
            continue
        members = (row["claim_a_id"], row["claim_b_id"])
        if _is_keep_drop(row):
            keep = _contradiction_keep_id(cx, row)
            must_be_active = members if keep is None else (keep,)
        else:
            must_be_active = members
        for m in must_be_active:
            if claim_status.get(m) != "active":
                violations.append({"kind": "resolution_rests_on_inactive_claim",
                                   "contradiction_id": row["id"],
                                   "disposition": row["disposition"],
                                   "claim_id": m, "claim_status": claim_status.get(m)})
        rc = row["rationale_concept_id"]
        if rc is not None and concept_status.get(rc) not in citable:
            violations.append({"kind": "resolution_rests_on_uncitable_concept",
                               "contradiction_id": row["id"], "concept_id": rc,
                               "concept_status": concept_status.get(rc)})

    for cid in _orphaned_supersessions(cx):
        violations.append({"kind": "supersession_without_active_successor", "claim_id": cid})

    for row in cx.execute(
        "SELECT c.id, c.subject FROM claims c JOIN subject_aliases a ON a.alias_from = c.subject"
    ):
        violations.append({"kind": "claim_subject_is_alias", "claim_id": row["id"],
                           "subject": row["subject"]})

    # Claim/contradiction/concept reviews go with their target by foreign key;
    # an alias review must be closed when the merge it was filed on changes.
    for row in cx.execute(
        "SELECT r.id, r.alias_from, json_extract(r.details, '$.canonical_to') AS filed_to, "
        "a.canonical_to AS current_to FROM review_queue r "
        "LEFT JOIN subject_aliases a ON a.alias_from = r.alias_from "
        "WHERE r.item_type = 'alias' AND r.status = 'open'"
    ):
        if row["current_to"] is None or row["current_to"] != row["filed_to"]:
            violations.append({"kind": "open_review_without_target", "review_id": row["id"],
                               "alias_from": row["alias_from"],
                               "filed_to": row["filed_to"], "current_to": row["current_to"]})
    return violations

class Store:
    """Thin wrapper around a SQLite database. Everything a CLI needs lives here."""

    def __init__(self, db_path: Path, locale: Optional[str] = None):
        """Open (or create) the SQLite store.

        ``locale`` configures subject/predicate normalization. If the store
        already has a configured locale, passing a different value raises
        :class:`ValueError` — switching locales mid-corpus would strand
        existing aliases (their keys are stored normalized under the old
        rules). Use ``config-set locale`` to change it deliberately.

        If ``locale`` is ``None``, the stored value is used (default: ``en``
        for fresh stores).
        """
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
        _run_alters(self.conn)
        self.fts_enabled = _init_fts(self.conn)
        self.conn.commit()
        _migrate(self.conn)
        self.conn.commit()
        self._init_locale(locale)

    # ---------- store config ----------

    def _init_locale(self, requested: Optional[str]) -> None:
        """Resolve the effective locale and load the normalizer.

        Conflict policy: if the store already has a configured locale and
        ``requested`` differs, raise. Matching values or a ``None`` request
        are accepted.
        """
        existing = self.get_config("locale")
        if requested is None:
            self._locale = existing or _DEFAULT_LOCALE
            if existing is None:
                self.set_config("locale", self._locale)
        else:
            if existing is not None and existing != requested:
                raise ValueError(
                    f"store locale is {existing!r}; refusing to switch to "
                    f"{requested!r} implicitly (use config-set to change)"
                )
            self._locale = requested
            if existing is None:
                self.set_config("locale", requested)
        self._normalizer = get_normalizer(self._locale)

    @property
    def locale(self) -> str:
        return self._locale

    def normalize_subject(self, s: str) -> str:
        """Dispatch to the store's configured normalizer."""
        return self._normalizer.normalize(s)

    def get_config(self, key: str) -> Optional[str]:
        """Return the stored value for ``key`` or ``None`` if unset."""
        row = self.conn.execute(
            "SELECT value FROM store_config WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_config(self, key: str, value: str) -> None:
        """Upsert a config key. Callers are responsible for validating the
        value (e.g. that ``locale`` is in :data:`normalization.NORMALIZERS`)."""
        with self.tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO store_config (key, value, updated_at) "
                "VALUES (?, ?, ?)",
                (key, value, time.time()),
            )

    def list_config(self) -> dict[str, str]:
        """Return all config key/value pairs."""
        rows = self.conn.execute(
            "SELECT key, value FROM store_config"
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_locale(self, new_locale: str) -> dict:
        """Change the store's locale, reloading the normalizer.

        This does NOT rewrite existing claims or aliases. Callers should
        expect that aliases written under the old rules may no longer match
        after the switch. Returns ``{"old": ..., "new": ...}``.
        """
        old = self._locale
        # Validates by raising if unknown
        self._normalizer = get_normalizer(new_locale)
        self._locale = new_locale
        self.set_config("locale", new_locale)
        return {"old": old, "new": new_locale}

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
        """Cascade-deletes claims and everything that depended on them.

        Runs the full deactivation cascade before the delete, so claims in
        other sources that were superseded by one of these come back to
        active instead of pointing at a deleted row.
        """
        with self.tx() as cx:
            return self._remove_source_tx(cx, source_id)

    def _remove_source_tx(self, cx: sqlite3.Connection, source_id: int) -> int:
        """:meth:`remove_source` inside an open transaction."""
        claim_rows = cx.execute(
            "SELECT id FROM claims WHERE source_id = ?", (source_id,)
        ).fetchall()
        claim_ids = [r["id"] for r in claim_rows]
        if claim_ids:
            self._on_claims_deactivated_tx(
                cx, claim_ids, cause="remove_source", removing=frozenset(claim_ids),
            )
        cur = cx.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        return cur.rowcount

    def replace_source(self, path: str, content: str) -> dict:
        """Replace every source at ``path`` with ``content``, in one transaction.

        Unchanged content keeps its source (and claims) and drops any other
        rows at the path. Content identical to a source at another path is
        refused before anything is removed (``ValueError``).
        """
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self.tx() as cx:
            old_ids = [r["id"] for r in cx.execute(
                "SELECT id FROM sources WHERE path = ? ORDER BY id", (path,)
            ).fetchall()]
            same = cx.execute(
                "SELECT id FROM sources WHERE sha256 = ?", (sha,)
            ).fetchone()
            if same is not None and same["id"] not in old_ids:
                raise ValueError(f"duplicate_content:{same['id']}")
            keep = same["id"] if same is not None else None
            removed_ids = [sid for sid in old_ids if sid != keep]
            removed_claims = 0
            for sid in removed_ids:
                removed_claims += cx.execute(
                    "SELECT COUNT(*) FROM claims WHERE source_id = ?", (sid,)
                ).fetchone()[0]
                self._remove_source_tx(cx, sid)
            if keep is None:
                keep = cx.execute(
                    "INSERT INTO sources (path, sha256, content, ingested_at) "
                    "VALUES (?, ?, ?, ?)",
                    (path, sha, content, time.time()),
                ).lastrowid
                status = "replaced" if old_ids else "ingested"
            else:
                status = "unchanged"
        return {
            "status": status,
            "old_source_ids": removed_ids,
            "old_claims_removed": removed_claims,
            "source_id": keep,
        }

    def _on_claims_deactivated_tx(
        self, cx: sqlite3.Connection, claim_ids, cause: str,
        removing: frozenset = frozenset(),
    ) -> dict:
        """Apply the retraction cascade for claims that just left the active set.

        Must run inside the caller's transaction, after the claims' status
        has been updated (or, for removal, just before they're deleted, with
        their ids passed as ``removing``).
        """
        ids = list(claim_ids)
        views = _invalidate_cache_for_claims(cx, ids, cause=cause)
        concepts = self._mark_concepts_stale_for_claims_tx(cx, ids, cause=cause)
        revived = _revive_orphaned_supersessions_tx(cx, removing, cause=cause)
        # Claims in retracted sources are never revived, but they can't keep
        # pointing at a row that's about to be deleted.
        if removing:
            ph = ",".join("?" for _ in removing)
            cx.execute(
                f"UPDATE claims SET superseded_by = NULL "
                f"WHERE superseded_by IN ({ph}) AND id NOT IN ({ph})",
                list(removing) + list(removing),
            )
        reopened = _reopen_contradictions_for_claims_tx(cx, ids, cause=cause)
        return {
            "views_invalidated": views,
            "concepts_staled": concepts,
            "claims_revived": revived,
            "contradictions_reopened": reopened,
        }

    def check_invariants(self) -> list[dict]:
        """See :func:`check_invariants`."""
        return check_invariants(self.conn)

    # ---------- claims ----------

    def resolve_subject(self, raw: str) -> str:
        """Normalize then resolve through the alias table (transitive, with cycle guard)."""
        s = self.normalize_subject(raw)
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

    def _alias_chain(self, normalized: str) -> list[str]:
        """Every subject visited while resolving ``normalized`` (inclusive)."""
        chain = [normalized]
        s = normalized
        for _ in range(10):  # cycle guard, as in resolve_subject
            row = self.conn.execute(
                "SELECT canonical_to FROM subject_aliases WHERE alias_from = ?", (s,)
            ).fetchone()
            if not row or row["canonical_to"] in chain:
                break
            s = row["canonical_to"]
            chain.append(s)
        return chain

    def add_alias(self, from_subject: str, to_subject: str, *, force: bool = False) -> dict:
        """Record that from_subject -> to_subject, and rewrite existing claims.

        Returns {'claims_rewritten': N, 'cache_cleared': bool, 'event_id': id}.
        Every rewrite is logged in ``alias_rewrites`` so :meth:`undo_alias`
        can restore the old subjects. If adding the alias would create a
        resolution cycle (e.g. A->B already exists and we're trying to add
        B->A), the alias is rejected with `note: 'would-create-cycle'`. An
        existing alias with a different target is not overwritten unless
        ``force`` is set (`note: 'alias-exists'`).
        """
        f = self.normalize_subject(from_subject)
        t = self.normalize_subject(to_subject)
        if not f or not t or f == t:
            return {"claims_rewritten": 0, "cache_cleared": False, "note": "no-op"}
        # cycle check: if resolving `t` through the current alias chain ever
        # reaches `f`, adding f -> t would make resolve(f) loop. Reject. The
        # whole chain matters, not just its end: when f -> x already exists,
        # the chain from t can pass through f and continue past it.
        if f in self._alias_chain(t):
            return {
                "claims_rewritten": 0,
                "cache_cleared": False,
                "note": "would-create-cycle",
            }
        existing = self.conn.execute(
            "SELECT canonical_to FROM subject_aliases WHERE alias_from = ?", (f,)
        ).fetchone()
        previous_to = existing["canonical_to"] if existing else None
        if previous_to == t:
            return {"claims_rewritten": 0, "cache_cleared": False, "note": "no-op"}
        if previous_to is not None and not force:
            return {
                "claims_rewritten": 0,
                "cache_cleared": False,
                "note": "alias-exists",
                "existing_to": previous_to,
            }
        with self.tx() as cx:
            now = time.time()
            cx.execute(
                "INSERT OR REPLACE INTO subject_aliases (alias_from, canonical_to, created_at) "
                "VALUES (?, ?, ?)",
                (f, t, now),
            )
            if previous_to is not None:
                self._obsolete_alias_reviews_tx(cx, f, "alias overwritten")
            event_id = cx.execute(
                "INSERT INTO alias_events (alias_from, canonical_to, previous_to, created_at) "
                "VALUES (?, ?, ?, ?)",
                (f, t, previous_to, now),
            ).lastrowid
            affected = [
                r["id"] for r in cx.execute(
                    "SELECT id FROM claims WHERE subject = ?", (f,)
                ).fetchall()
            ]
            cx.executemany(
                "INSERT INTO alias_rewrites (event_id, claim_id, old_subject) VALUES (?, ?, ?)",
                [(event_id, cid, f) for cid in affected],
            )
            # Rewrite to the end of the chain: if `t` is itself an alias,
            # writing `t` would leave claims under a non-canonical subject.
            cur = cx.execute(
                "UPDATE claims SET subject = ? WHERE subject = ?",
                (self.resolve_subject(t), f),
            )
            rewritten = cur.rowcount
            cache_cleared = (
                _invalidate_cache_for_claims(cx, affected, cause="alias")
                if affected else 0
            )
            # mark dependent concepts stale
            if affected:
                self._mark_concepts_stale_for_claims_tx(cx, affected, cause="alias")
        log("alias_added", level="info", event_id=event_id, alias_from=f,
            canonical_to=t, previous_to=previous_to, claims_rewritten=rewritten)
        result = {"claims_rewritten": rewritten, "cache_cleared": cache_cleared > 0,
                  "event_id": event_id}
        if previous_to is not None:
            result["overwrote"] = previous_to
        return result

    def undo_alias(self, from_subject: str) -> dict:
        """Undo the most recent live merge of ``from_subject``.

        Restores the old subject on every claim the merge rewrote that still
        carries the merged-into subject (claims edited since are reported,
        not touched), removes the alias (or restores the target it
        overwrote), invalidates views citing the restored claims and marks
        their concepts stale.
        """
        f = self.normalize_subject(from_subject)
        event = self.conn.execute(
            "SELECT * FROM alias_events WHERE alias_from = ? AND undone_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (f,),
        ).fetchone()
        if event is None:
            return {"undone": False, "note": "no-live-alias", "alias_from": f}
        # Restoring an overwritten target gets the same cycle check as
        # add_alias: an alias added since may now lead back to f.
        prev = event["previous_to"]
        if prev is not None and f in self._alias_chain(prev):
            return {"undone": False, "note": "would-create-cycle", "alias_from": f,
                    "restore_to": prev}
        with self.tx() as cx:
            current_target = self.resolve_subject(event["canonical_to"])
            rewrites = cx.execute(
                "SELECT ar.claim_id, ar.old_subject, c.subject FROM alias_rewrites ar "
                "JOIN claims c ON c.id = ar.claim_id WHERE ar.event_id = ?",
                (event["id"],),
            ).fetchall()
            restored = [r["claim_id"] for r in rewrites if r["subject"] == current_target]
            skipped = [r["claim_id"] for r in rewrites if r["subject"] != current_target]
            for r in rewrites:
                if r["claim_id"] in restored:
                    cx.execute("UPDATE claims SET subject = ? WHERE id = ?",
                               (r["old_subject"], r["claim_id"]))
            if event["previous_to"] is not None:
                cx.execute(
                    "UPDATE subject_aliases SET canonical_to = ? WHERE alias_from = ?",
                    (event["previous_to"], f),
                )
            else:
                cx.execute("DELETE FROM subject_aliases WHERE alias_from = ?", (f,))
            cx.execute("UPDATE alias_events SET undone_at = ? WHERE id = ?",
                       (time.time(), event["id"]))
            self._obsolete_alias_reviews_tx(cx, f, "alias undone")
            if restored:
                _invalidate_cache_for_claims(cx, restored, cause="alias_undo")
                self._mark_concepts_stale_for_claims_tx(cx, restored, cause="alias_undo")
        log("alias_undone", level="info", event_id=event["id"], alias_from=f,
            claims_restored=len(restored), claims_skipped=len(skipped))
        return {
            "undone": True,
            "event_id": event["id"],
            "alias_from": f,
            "canonical_to": event["canonical_to"],
            "restored_alias_to": event["previous_to"],
            "claims_restored": restored,
            "claims_skipped": skipped,
        }

    def list_alias_events(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT e.*, (SELECT COUNT(*) FROM alias_rewrites r WHERE r.event_id = e.id) "
            "AS claims_rewritten FROM alias_events e ORDER BY e.id DESC LIMIT ?",
            (limit,),
        ).fetchall()

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
        proposition: Optional[str] = None,
    ) -> int:
        """Insert a claim. Its context window is computed from the source,
        and the fidelity checker runs on the proposition (else predicate +
        object, as written): any issue queues the claim for review in the
        same transaction. Fidelity issues never refuse the write."""
        from .fidelity import check_fidelity, context_window

        canonical_subject = self.resolve_subject(subject)
        # Predicates get the same locale-aware rules: they are English words
        # in English corpora, Italian verbs in Italian corpora, etc.
        canonical_predicate = self.normalize_subject(predicate)
        proposition = (proposition or "").strip() or None
        src = self.get_source(source_id)
        content = src["content"] if src else ""
        c_start, c_end = context_window(content, span_start, span_end)
        issues = check_fidelity(
            _fidelity_text(proposition, predicate, object_),
            content[span_start:span_end], content[c_start:c_end],
        )
        with self.tx() as cx:
            cur = cx.execute(
                """
                INSERT INTO claims (source_id, subject, predicate, object,
                                    span_start, span_end, confidence, extracted_at,
                                    proposition, context_start, context_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, canonical_subject, canonical_predicate, object_.strip(),
                 span_start, span_end, confidence, time.time(),
                 proposition, c_start, c_end),
            )
            claim_id = cur.lastrowid
            if issues:
                self._enqueue_review_tx(
                    cx, "claim", claim_id, reason="fidelity",
                    details={"issues": [i.to_dict() for i in issues]},
                )
            return claim_id

    def get_claim(self, claim_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()

    def get_context_text(self, claim_id: int) -> Optional[str]:
        """The source text of the claim's context window (the span plus its
        neighbouring sentences)."""
        row = self.conn.execute(
            "SELECT s.content, c.context_start, c.context_end, c.span_start, c.span_end "
            "FROM claims c JOIN sources s ON c.source_id = s.id WHERE c.id = ?",
            (claim_id,),
        ).fetchone()
        if not row:
            return None
        start = row["context_start"] if row["context_start"] is not None else row["span_start"]
        end = row["context_end"] if row["context_end"] is not None else row["span_end"]
        return row["content"][start:end]

    def claim_text(self, claim_id: int) -> Optional[str]:
        """The claim as a sentence: its proposition, or the triple when the
        claim predates propositions."""
        row = self.get_claim(claim_id)
        if not row:
            return None
        return row["proposition"] or f"{row['subject']} {row['predicate']} {row['object']}"

    def check_claim_fidelity(self, claim_id: int) -> list:
        """Re-run the fidelity checker on a stored claim."""
        from .fidelity import check_fidelity

        row = self.get_claim(claim_id)
        if row is None:
            raise LookupError(f"no claim with id {claim_id}")
        return check_fidelity(
            _fidelity_text(row["proposition"], row["predicate"], row["object"]),
            self.get_span_text(claim_id) or "", self.get_context_text(claim_id) or "")

    # ---------- review queue ----------

    _REVIEW_TARGET_COLUMN = {
        "claim": ("claim_id", "claims", "id"),
        "contradiction": ("contradiction_id", "contradictions", "id"),
        "concept": ("concept_id", "concepts", "id"),
        "alias": ("alias_from", "subject_aliases", "alias_from"),
    }

    @staticmethod
    def _obsolete_alias_reviews_tx(cx: sqlite3.Connection, alias_from: str, note: str) -> None:
        """Close open reviews of an alias whose mapping just changed: they
        were filed on a merge that no longer exists."""
        cx.execute(
            "UPDATE review_queue SET status = 'obsolete', resolved_at = ?, "
            "resolution_note = ? "
            "WHERE item_type = 'alias' AND alias_from = ? AND status = 'open'",
            (time.time(), note, alias_from),
        )

    def _enqueue_review_tx(
        self, cx: sqlite3.Connection, item_type: str, target, *,
        reason: str, details: Optional[dict] = None,
    ) -> int:
        if item_type not in REVIEW_ITEM_TYPES:
            raise ValueError(
                f"item_type must be one of {REVIEW_ITEM_TYPES}, got {item_type!r}")
        column, table, key = self._REVIEW_TARGET_COLUMN[item_type]
        details = dict(details or {})
        if item_type == "alias":
            target = self.normalize_subject(str(target))
            row = cx.execute("SELECT canonical_to FROM subject_aliases WHERE alias_from = ?",
                             (target,)).fetchone()
            if row is None:
                raise LookupError(f"no alias {target!r}")
            # The merge under review, so a later overwrite can't change it.
            details["canonical_to"] = row["canonical_to"]
        elif cx.execute(f"SELECT 1 FROM {table} WHERE {key} = ?", (target,)).fetchone() is None:
            raise LookupError(f"no {item_type} {target!r}")
        details_json = json.dumps(details, ensure_ascii=False, sort_keys=True)
        existing = cx.execute(
            f"SELECT id FROM review_queue WHERE item_type = ? AND {column} = ? "
            "AND reason = ? AND status = 'open'",
            (item_type, target, reason),
        ).fetchone()
        if existing:
            return existing["id"]
        # A decision already made on exactly this finding stands; re-running
        # a check must not re-queue what a reviewer accepted or rejected.
        decided = cx.execute(
            f"SELECT id FROM review_queue WHERE item_type = ? AND {column} = ? "
            "AND reason = ? AND status IN ('accepted', 'rejected') AND details = ? "
            "ORDER BY id DESC LIMIT 1",
            (item_type, target, reason, details_json),
        ).fetchone()
        if decided:
            return decided["id"]
        cur = cx.execute(
            f"INSERT INTO review_queue (item_type, {column}, reason, details, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (item_type, target, reason, details_json, time.time()),
        )
        return cur.lastrowid

    def enqueue_review(
        self, item_type: str, target, *, reason: str, details: Optional[dict] = None,
    ) -> int:
        """Queue an item for human/agent review. ``target`` is the claim,
        contradiction or concept id, or the alias's ``from`` subject. An
        identical open item (same target and reason) is returned, not
        duplicated."""
        with self.tx() as cx:
            return self._enqueue_review_tx(cx, item_type, target, reason=reason,
                                           details=details)

    def open_claim_review(self, claim_id: int, reason: str) -> Optional[sqlite3.Row]:
        """The open review item for ``claim_id`` with ``reason``, if any."""
        return self.conn.execute(
            "SELECT * FROM review_queue WHERE item_type = 'claim' AND claim_id = ? "
            "AND reason = ? AND status = 'open'",
            (claim_id, reason),
        ).fetchone()

    def get_review(self, review_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM review_queue WHERE id = ?", (review_id,)).fetchone()

    def list_reviews(
        self, *, status: str = "open", item_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        """Review items, oldest first. ``status='all'`` lists every item."""
        where, params = [], []
        if status != "all":
            where.append("status = ?")
            params.append(status)
        if item_type:
            where.append("item_type = ?")
            params.append(item_type)
        sql = "SELECT * FROM review_queue"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return self.conn.execute(sql + " ORDER BY id LIMIT ?", (*params, limit)).fetchall()

    def resolve_review(
        self, review_id: int, decision: str, *, resolved_by: str,
        note: Optional[str] = None,
    ) -> None:
        """Record a decision on an open item: ``accepted`` (the item is
        right as stored) or ``rejected`` (it is wrong). Recording a decision
        changes nothing else; fixing a rejected item is a separate, explicit
        step (supersede the claim, re-dispose the contradiction, undo the
        alias, invalidate the concept)."""
        if decision not in REVIEW_DECISIONS:
            raise ValueError(
                f"decision must be one of {REVIEW_DECISIONS}, got {decision!r}")
        row = self.get_review(review_id)
        if row is None:
            raise LookupError(f"no review item {review_id}")
        if row["status"] != "open":
            raise ValueError(f"review item {review_id} is already {row['status']}")
        with self.tx() as cx:
            cur = cx.execute(
                "UPDATE review_queue SET status = ?, resolved_at = ?, resolved_by = ?, "
                "resolution_note = ? WHERE id = ? AND status = 'open'",
                (decision, time.time(), resolved_by, note, review_id),
            )
            if cur.rowcount != 1:  # closed by someone else since we read it
                raise ValueError(f"review item {review_id} is no longer open")

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

    def search_claims(
        self, keywords: list[str], limit: int = 30, *,
        expand_aliases: bool = False,
        domain: Optional[str] = None,
        include_retracted: bool = False,
    ) -> list[sqlite3.Row]:
        """Simple keyword search across subject/predicate/object.

        Ranks by number of matching keywords, then confidence, then recency.
        Returns each row with an inline `span_text` column (joined from the
        source) so callers don't need to round-trip per claim for spans.

        When ``expand_aliases=True`` the keyword list is expanded with all
        surface predicates that resolve to any of the given keywords (WS-E.1).
        ``domain`` selects which predicate alias rows participate in the
        expansion; falls back to ``*`` when no domain-specific row matches.

        ``include_retracted=True`` also returns ``retracted`` claims, for the
        ``include_retracted`` query context.
        """
        if not keywords:
            return []
        if expand_aliases:
            from . import predicates
            keywords = predicates.expand_keywords(self, keywords, domain=domain)
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
            WHERE c.status IN ({_statuses_sql(include_retracted)})
              AND ({match_exprs}) > 0
            ORDER BY hits DESC, c.confidence DESC, c.extracted_at DESC
            LIMIT ?
        """
        return self.conn.execute(sql, params + params + [limit]).fetchall()

    def search_claims_fts(
        self, keywords: list[str], limit: int = 30, *,
        expand_aliases: bool = False,
        domain: Optional[str] = None,
        include_retracted: bool = False,
    ) -> list[sqlite3.Row]:
        """BM25-ranked search over the FTS5 index of claim triples.

        Falls back to an empty list (not an exception) if FTS5 isn't enabled
        on this store, so callers can safely call it unconditionally and then
        inspect `fts_enabled` if they want to distinguish.

        When ``expand_aliases=True`` the keyword list is expanded via the
        WS-E.1 ``predicate_aliases`` table; same semantics as
        :meth:`search_claims`.
        """
        if not self.fts_enabled or not keywords:
            return []
        if expand_aliases:
            from . import predicates
            keywords = predicates.expand_keywords(self, keywords, domain=domain)
        cleaned = [re.sub(r"[^\w-]", "", k.lower()) for k in keywords]
        cleaned = [k for k in cleaned if len(k) > 2]
        if not cleaned:
            return []
        # double-quote each term to avoid FTS5 interpreting punctuation/syntax
        match_query = " OR ".join(f'"{k}"' for k in cleaned)
        sql = f"""
            SELECT c.*,
                   SUBSTR(s.content, c.span_start + 1, c.span_end - c.span_start)
                       AS span_text,
                   bm25(claims_fts) AS bm25_score
            FROM claims_fts
            JOIN claims c ON c.id = claims_fts.rowid
            JOIN sources s ON c.source_id = s.id
            WHERE claims_fts MATCH ?
              AND c.status IN ({_statuses_sql(include_retracted)})
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
            self._on_claims_deactivated_tx(cx, [old_id], cause="supersede")

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
        qh = view_cache_key(query)
        return self.conn.execute(
            "SELECT * FROM view_cache WHERE query_hash = ?", (qh,)
        ).fetchone()

    def cache_view(self, query: str, response: str, claim_ids: list[int]) -> None:
        """Cache a view. Every cited claim must be active: a view citing an
        inactive claim would be stale the moment it's written."""
        qh = view_cache_key(query)
        if claim_ids:
            ph = ",".join("?" for _ in claim_ids)
            active = {
                r["id"] for r in self.conn.execute(
                    f"SELECT id FROM claims WHERE id IN ({ph}) AND status = 'active'",
                    list(claim_ids),
                )
            }
            inactive = sorted(set(claim_ids) - active)
            if inactive:
                raise ValueError(
                    "view_cites_inactive_claim:" + ",".join(map(str, inactive))
                )
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

    # ---------- concepts ----------

    def add_concept(
        self,
        subject: str,
        statement: str,
        inference_type: str,
        confidence: float,
        support_claim_ids: list[tuple[int, str]],
        status: str = "draft",
    ) -> int:
        """Insert a concept with its support rows in one transaction.
        Returns the new concept id. Does NOT run validation -- caller (WS-A)
        decides when to validate and transition from draft -> active."""
        with self.tx() as cx:
            cur = cx.execute(
                "INSERT INTO concepts "
                "(subject, statement, inference_type, confidence, status, derived_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (subject, statement, inference_type, confidence, status, time.time()),
            )
            concept_id = cur.lastrowid
            for claim_id, role in support_claim_ids:
                cx.execute(
                    "INSERT INTO concept_supports (concept_id, claim_id, role) "
                    "VALUES (?, ?, ?)",
                    (concept_id, claim_id, role),
                )
            return concept_id

    def get_concept(self, concept_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM concepts WHERE id = ?", (concept_id,)
        ).fetchone()

    def get_concept_supports(self, concept_id: int) -> list[sqlite3.Row]:
        """Return rows joining concept_supports with claims."""
        return self.conn.execute(
            "SELECT cs.*, c.subject, c.predicate, c.object, "
            "c.confidence AS claim_confidence, c.status AS claim_status "
            "FROM concept_supports cs "
            "JOIN claims c ON cs.claim_id = c.id "
            "WHERE cs.concept_id = ?",
            (concept_id,),
        ).fetchall()

    def list_concepts(
        self, *, subject: Optional[str] = None,
        status: Optional[str] = None, limit: int = 100,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM concepts WHERE 1=1"
        params: list = []
        if subject is not None:
            sql += " AND subject = ?"
            params.append(subject)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY derived_at DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def update_concept_status(
        self, concept_id: int, status: str, *,
        validation_verdict: Optional[str] = None,
        validation_reason: Optional[str] = None,
        superseded_by: Optional[int] = None,
    ) -> None:
        """Transition a concept's status. If the new status is one of
        {superseded, invalidated}, also invalidate any cached view that cited
        this concept (use _invalidate_cache_for_concepts).

        ``superseded_by`` is kept unless a new value is given. A
        ``validation_reason`` without a verdict (e.g. ``concept-invalidate
        --reason``) is still recorded.
        """
        with self.tx() as cx:
            if validation_verdict is not None:
                cx.execute(
                    "UPDATE concepts SET status = ?, validation_verdict = ?, "
                    "validation_reason = ?, last_validated_at = ?, "
                    "superseded_by = COALESCE(?, superseded_by) WHERE id = ?",
                    (status, validation_verdict, validation_reason,
                     time.time(), superseded_by, concept_id),
                )
            else:
                cx.execute(
                    "UPDATE concepts SET status = ?, "
                    "validation_reason = COALESCE(?, validation_reason), "
                    "superseded_by = COALESCE(?, superseded_by) WHERE id = ?",
                    (status, validation_reason, superseded_by, concept_id),
                )
            if status in ("active", "attested"):
                self._require_active_supports_tx(cx, concept_id)
            if status in ("superseded", "invalidated", "stale"):
                _invalidate_cache_for_concepts(cx, [concept_id], cause=f"concept_{status}")
                _reopen_contradictions_for_concepts_tx(
                    cx, [concept_id], cause=f"concept_{status}",
                )

    def _require_active_supports_tx(self, cx: sqlite3.Connection, concept_id: int) -> None:
        """Refuse to make a concept citable while any support claim is inactive."""
        bad = cx.execute(
            "SELECT cs.claim_id, c.status FROM concept_supports cs "
            "LEFT JOIN claims c ON c.id = cs.claim_id "
            "WHERE cs.concept_id = ? AND (c.status IS NULL OR c.status != 'active')",
            (concept_id,),
        ).fetchall()
        if bad:
            raise ValueError(
                "concept_support_inactive:"
                + ",".join(f"{r['claim_id']}={r['status']}" for r in bad)
            )

    def _mark_concepts_stale_for_claims_tx(
        self, cx: sqlite3.Connection, claim_ids: list[int], cause: str,
    ) -> int:
        """Inner helper that runs inside an existing transaction (cx).

        Any concept whose support set intersects claim_ids -> status='stale'.
        Also invalidates cached views citing those concepts.
        """
        if not claim_ids:
            return 0
        placeholders = ",".join("?" for _ in claim_ids)
        rows = cx.execute(
            f"SELECT DISTINCT concept_id FROM concept_supports "
            f"WHERE claim_id IN ({placeholders})",
            claim_ids,
        ).fetchall()
        concept_ids = [r["concept_id"] for r in rows]
        if not concept_ids:
            return 0
        c_placeholders = ",".join("?" for _ in concept_ids)
        # Attested concepts also transition to stale when their supports change:
        # the attestation was over the *old* support set, and it should be
        # re-attested (or rebuilt) under the new one.
        cx.execute(
            f"UPDATE concepts SET status = 'stale' "
            f"WHERE id IN ({c_placeholders}) AND status NOT IN ('superseded', 'invalidated')",
            concept_ids,
        )
        _invalidate_cache_for_concepts(cx, concept_ids, cause=cause)
        _reopen_contradictions_for_concepts_tx(cx, concept_ids, cause=cause)
        return len(concept_ids)

    def mark_concepts_stale_for_claims(
        self, claim_ids: list[int], cause: str,
    ) -> int:
        """Any concept whose support set intersects claim_ids -> status='stale'.
        Also invalidates cached views citing those concepts. Call from the
        transactions in supersede_claim, remove_source, add_alias."""
        with self.tx() as cx:
            return self._mark_concepts_stale_for_claims_tx(cx, claim_ids, cause)

    # ---------- claim conditions ----------

    def add_claim_condition(
        self, claim_id: int, condition_claim_id: int,
        kind: str, explicit: bool, confidence: float,
    ) -> None:
        """Upsert a claim_conditions row."""
        with self.tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO claim_conditions "
                "(claim_id, condition_claim_id, kind, explicit, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (claim_id, condition_claim_id, kind, 1 if explicit else 0, confidence),
            )

    def get_claim_conditions(self, claim_id: int) -> list[sqlite3.Row]:
        """Return the conditions of a claim, joined with the condition claim."""
        return self.conn.execute(
            "SELECT cc.*, c.subject, c.predicate, c.object, "
            "c.confidence AS cond_confidence, c.status AS cond_status "
            "FROM claim_conditions cc "
            "JOIN claims c ON cc.condition_claim_id = c.id "
            "WHERE cc.claim_id = ?",
            (claim_id,),
        ).fetchall()

    def conditions_overlap(self, claim_a: int, claim_b: int) -> float:
        """Jaccard overlap over condition_claim_ids. Returns 0.0 if either
        claim has no conditions (informational, not an error)."""
        a_rows = self.conn.execute(
            "SELECT condition_claim_id FROM claim_conditions WHERE claim_id = ?",
            (claim_a,),
        ).fetchall()
        b_rows = self.conn.execute(
            "SELECT condition_claim_id FROM claim_conditions WHERE claim_id = ?",
            (claim_b,),
        ).fetchall()
        a_set = {r["condition_claim_id"] for r in a_rows}
        b_set = {r["condition_claim_id"] for r in b_rows}
        if not a_set or not b_set:
            return 0.0
        intersection = len(a_set & b_set)
        union = len(a_set | b_set)
        return intersection / union if union > 0 else 0.0

    # ---------- source metadata ----------

    _VALID_DOMAINS = {"legal", "scientific", "policy", "corporate", "generic"}

    def set_source_metadata(
        self, source_id: int, domain: str, metadata: dict,
    ) -> None:
        """Upsert. Validates domain is one of the known values; stores metadata
        as json.dumps(metadata)."""
        if domain not in self._VALID_DOMAINS:
            raise ValueError(
                f"Unknown domain {domain!r}; expected one of {sorted(self._VALID_DOMAINS)}"
            )
        with self.tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO source_metadata "
                "(source_id, domain, metadata, updated_at) VALUES (?, ?, ?, ?)",
                (source_id, domain, json.dumps(metadata), time.time()),
            )

    def get_source_metadata(self, source_id: int) -> Optional[dict]:
        """Return {'domain': str, 'metadata': dict} or None."""
        row = self.conn.execute(
            "SELECT * FROM source_metadata WHERE source_id = ?", (source_id,)
        ).fetchone()
        if not row:
            return None
        return {"domain": row["domain"], "metadata": json.loads(row["metadata"])}

    def list_sources_by_domain(self, domain: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT sm.*, s.path, s.sha256, s.ingested_at "
            "FROM source_metadata sm "
            "JOIN sources s ON sm.source_id = s.id "
            "WHERE sm.domain = ? ORDER BY sm.updated_at DESC",
            (domain,),
        ).fetchall()

    # ---------- contradiction disposition ----------

    _DISPOSITION_STATUS_MAP = {
        "unresolved": "open",
        "supersede": "resolved",
        "coexist": "resolved",
        "distinguish": "resolved",
        "reconcile": "resolved",
        "replicate": "resolved",
        "dispute": "resolved",
        "retracted": "resolved",
        "gap": "resolved",
    }

    def update_contradiction_disposition(
        self, contradiction_id: int, disposition: str, *,
        rule: Optional[str] = None,
        applies_when: Optional[dict] = None,
        rationale_concept_id: Optional[int] = None,
        overlap_score: Optional[float] = None,
        keep: Optional[int] = None,
        confidence_deltas: Optional[tuple[float, float]] = None,
    ) -> None:
        """Set the disposition, write contradiction_rules if rule is provided,
        update status appropriately.

        ``keep`` records the claim a keep/drop resolution kept (stored as
        ``resolved_to``); ``confidence_deltas`` records the exact replicate
        bump per claim so a reopen can revert it.
        """
        status = self._DISPOSITION_STATUS_MAP.get(disposition, "open")
        da, db_ = confidence_deltas if confidence_deltas else (None, None)
        with self.tx() as cx:
            cx.execute(
                "UPDATE contradictions SET disposition = ?, disposition_at = ?, "
                "overlap_score = COALESCE(?, overlap_score), status = ?, "
                "resolved_to = ?, confidence_delta_a = ?, confidence_delta_b = ? "
                "WHERE id = ?",
                (disposition, time.time(), overlap_score, status,
                 keep, da, db_, contradiction_id),
            )
            if rule is not None:
                cx.execute(
                    "INSERT OR REPLACE INTO contradiction_rules "
                    "(contradiction_id, rule, applies_when, rationale_concept_id, decided_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (contradiction_id, rule,
                     json.dumps(applies_when) if applies_when else None,
                     rationale_concept_id, time.time()),
                )

    def add_contradiction_cross_subject(
        self, a_id: int, b_id: int, *,
        relation_kind: str, justification: str,
    ) -> Optional[int]:
        """Cross-subject contradiction (P1.6).

        Bypasses the same-subject invariant that ``add_contradiction`` enforces.
        ``relation_kind`` is a small vocabulary (regime-supersedes,
        rule-limits-rule, doctrinal-cross-ref) that tells downstream readers how
        to interpret the pair; ``justification`` is a required free-text
        explanation. Returns None if the pair already exists.
        """
        lo, hi = sorted([a_id, b_id])
        with self.tx() as cx:
            existing = cx.execute(
                "SELECT id FROM contradictions WHERE claim_a_id = ? AND claim_b_id = ?",
                (lo, hi),
            ).fetchone()
            if existing:
                return None
            cur = cx.execute(
                "INSERT INTO contradictions "
                "(claim_a_id, claim_b_id, detected_at, kind, "
                " cross_subject, relation_kind, justification) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (lo, hi, time.time(), "unknown", relation_kind, justification),
            )
            return cur.lastrowid

    def attest_concept(
        self, concept_id: int, *, attested_by: str, rationale: str,
    ) -> None:
        """Record an agent-driven attestation that promotes a draft concept
        to ``attested`` (P1.2).

        Unlike ``concept-validate`` (which calls the LLM to check GROUNDED
        against the union of support spans), attestation is the honest
        agent-mode pathway: the operator takes responsibility for the
        grounding. Only ``draft`` concepts can be attested; the transition is
        ``draft -> attested``. ``active -> attested`` would be a downgrade and
        is disallowed. Cached views citing this concept are invalidated so
        subsequent reads reflect the new status.
        """
        row = self.conn.execute(
            "SELECT status FROM concepts WHERE id = ?", (concept_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"no concept with id {concept_id}")
        if row["status"] != "draft":
            raise ValueError(
                f"only draft concepts can be attested; current status is {row['status']!r}"
            )
        with self.tx() as cx:
            self._require_active_supports_tx(cx, concept_id)
            cx.execute(
                "UPDATE concepts SET status = 'attested', attested_by = ?, "
                "attested_at = ?, attestation_rationale = ? WHERE id = ?",
                (attested_by, time.time(), rationale, concept_id),
            )
            _invalidate_cache_for_concepts(cx, [concept_id], cause="concept_attested")

    def add_contradiction_with_kind(
        self, a_id: int, b_id: int, kind: str, *,
        candidate_disposition: Optional[str] = None,
        overlap_score: Optional[float] = None,
    ) -> Optional[int]:
        """Same as add_contradiction but carries detection metadata.
        Returns None if the pair already exists."""
        lo, hi = sorted([a_id, b_id])
        with self.tx() as cx:
            existing = cx.execute(
                "SELECT id FROM contradictions WHERE claim_a_id = ? AND claim_b_id = ?",
                (lo, hi),
            ).fetchone()
            if existing:
                return None
            cur = cx.execute(
                "INSERT INTO contradictions "
                "(claim_a_id, claim_b_id, detected_at, kind, candidate_disposition, overlap_score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (lo, hi, time.time(), kind, candidate_disposition, overlap_score),
            )
            return cur.lastrowid

    def list_contradictions_full(
        self, *, disposition: Optional[str] = None,
        subject: Optional[str] = None, limit: int = 100,
    ) -> list[sqlite3.Row]:
        """Left-joins contradictions with contradiction_rules."""
        sql = (
            "SELECT ct.*, cr.rule, cr.applies_when, cr.rationale_concept_id, cr.decided_at "
            "FROM contradictions ct "
            "LEFT JOIN contradiction_rules cr ON ct.id = cr.contradiction_id "
            "WHERE 1=1"
        )
        params: list = []
        if disposition is not None:
            sql += " AND ct.disposition = ?"
            params.append(disposition)
        if subject is not None:
            sql += (
                " AND (ct.claim_a_id IN (SELECT id FROM claims WHERE subject = ?) "
                "OR ct.claim_b_id IN (SELECT id FROM claims WHERE subject = ?))"
            )
            params.extend([subject, subject])
        sql += " ORDER BY ct.detected_at DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    # ---------- retract source ----------

    def retract_source(self, source_id: int) -> int:
        """Mark all active claims from this source as status='retracted'.
        Mark dependent concepts stale. Invalidate views citing those claims.
        Returns the number of claims transitioned. Does NOT delete the source
        (retraction is part of the record)."""
        with self.tx() as cx:
            return self._retract_source_claims_tx(cx, source_id)

    def _retract_source_claims_tx(self, cx: sqlite3.Connection, source_id: int) -> int:
        """:meth:`retract_source` inside an open transaction."""
        claim_rows = cx.execute(
            "SELECT id FROM claims WHERE source_id = ? AND status = 'active'",
            (source_id,),
        ).fetchall()
        claim_ids = [r["id"] for r in claim_rows]
        if not claim_ids:
            return 0
        placeholders = ",".join("?" for _ in claim_ids)
        cx.execute(
            f"UPDATE claims SET status = 'retracted', retracted_cause = 'source' "
            f"WHERE id IN ({placeholders})",
            claim_ids,
        )
        self._on_claims_deactivated_tx(cx, claim_ids, cause="retract_source")
        return len(claim_ids)

    # ---------- WS-E.1: predicate aliases ----------

    def _resolve_predicate_chain(
        self, cx: sqlite3.Connection, predicate: str, domain: Optional[str],
    ) -> str:
        """Walk the alias chain for ``predicate`` inside an open transaction.

        Tries domain-specific rows first, then ``*`` wildcards. Cycle-guarded
        with a 10-step ceiling, mirroring :meth:`resolve_subject`.
        """
        return self._predicate_chain(cx, predicate, domain)[-1]

    def _predicate_chain(
        self, cx: sqlite3.Connection, predicate: str, domain: Optional[str],
    ) -> list[str]:
        """Every predicate visited while resolving ``predicate`` (inclusive)."""
        chain = [predicate]
        s = predicate
        for _ in range(10):
            row = None
            if domain:
                row = cx.execute(
                    "SELECT canonical_to FROM predicate_aliases "
                    "WHERE alias_from = ? AND domain = ?",
                    (s, domain),
                ).fetchone()
            if row is None:
                row = cx.execute(
                    "SELECT canonical_to FROM predicate_aliases "
                    "WHERE alias_from = ? AND domain = '*'",
                    (s,),
                ).fetchone()
            if not row or row["canonical_to"] in chain:
                break
            s = row["canonical_to"]
            chain.append(s)
        return chain

    def resolve_predicate(
        self, predicate: str, *, domain: Optional[str] = None,
    ) -> str:
        """Normalize then resolve through the predicate alias table.

        Domain-specific rows are tried first and fall back to the ``*``
        wildcard. Transitive resolution with a 10-step cycle guard. If
        ``domain`` is ``None`` only ``*`` aliases apply.
        """
        s = self.normalize_subject(predicate)
        return self._resolve_predicate_chain(self.conn, s, domain)

    def add_predicate_alias(
        self, domain: str, from_predicate: str, to_predicate: str,
    ) -> dict:
        """Record that ``(domain, from_predicate) -> to_predicate``.

        Both sides are normalized via :meth:`normalize_subject`. The alias is
        rejected if it would create a per-domain resolution cycle. On success,
        atomically rewrites every matching claim's predicate, invalidates
        cached views that cite affected claims, and marks dependent concepts
        stale — all in one transaction.

        Returns ``{"claims_rewritten": N, "cache_cleared": bool, "domain":
        str, "from_canonical": str, "to_canonical": str}`` on success, or
        ``{"claims_rewritten": 0, "cache_cleared": False,
        "note": "would-create-cycle"}`` if the alias would form a cycle.
        """
        if domain != "*" and domain not in self._VALID_DOMAINS:
            raise ValueError(
                f"Unknown domain {domain!r}; expected one of "
                f"{sorted(self._VALID_DOMAINS)} or '*'"
            )
        f = self.normalize_subject(from_predicate)
        t = self.normalize_subject(to_predicate)
        if not f or not t or f == t:
            return {
                "claims_rewritten": 0, "cache_cleared": False,
                "domain": domain, "from_canonical": f, "to_canonical": t,
                "note": "no-op",
            }
        # cycle check: if resolving `t` through the current alias chain ever
        # passes through `f` (not just ends there), adding f -> t would loop.
        chain = self._predicate_chain(self.conn, t, domain)
        if f in chain:
            return {
                "claims_rewritten": 0, "cache_cleared": False,
                "domain": domain, "from_canonical": f, "to_canonical": t,
                "note": "would-create-cycle",
            }
        with self.tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO predicate_aliases "
                "(domain, alias_from, canonical_to, created_at) VALUES (?, ?, ?, ?)",
                (domain, f, t, time.time()),
            )
            # Find every active claim whose predicate matches `f` and whose
            # source belongs to this domain (or `*` wildcard rewrites all).
            if domain == "*":
                affected_rows = cx.execute(
                    "SELECT id FROM claims WHERE predicate = ?", (f,),
                ).fetchall()
            else:
                affected_rows = cx.execute(
                    "SELECT c.id FROM claims c "
                    "LEFT JOIN source_metadata sm ON sm.source_id = c.source_id "
                    "WHERE c.predicate = ? AND sm.domain = ?",
                    (f, domain),
                ).fetchall()
            affected = [r["id"] for r in affected_rows]
            rewritten = 0
            cache_cleared = 0
            if affected:
                placeholders = ",".join("?" for _ in affected)
                # Rewrite to the end of t's chain, so claims never sit under
                # a predicate that is itself an alias.
                cur = cx.execute(
                    f"UPDATE claims SET predicate = ? WHERE id IN ({placeholders})",
                    [chain[-1], *affected],
                )
                rewritten = cur.rowcount
                cache_cleared = _invalidate_cache_for_claims(
                    cx, affected, cause="predicate_alias",
                )
                self._mark_concepts_stale_for_claims_tx(
                    cx, affected, cause="predicate_alias",
                )
        return {
            "claims_rewritten": rewritten,
            "cache_cleared": cache_cleared > 0,
            "domain": domain,
            "from_canonical": f,
            "to_canonical": t,
        }

    def list_predicate_aliases(
        self, domain: Optional[str] = None,
    ) -> list[sqlite3.Row]:
        """All alias rows, optionally filtered by ``domain`` (None = all)."""
        if domain is None:
            return self.conn.execute(
                "SELECT * FROM predicate_aliases ORDER BY domain, alias_from"
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM predicate_aliases WHERE domain = ? "
            "ORDER BY alias_from",
            (domain,),
        ).fetchall()

    def remove_predicate_alias(
        self, domain: str, from_predicate: str,
    ) -> dict:
        """Delete a single alias row.

        Does NOT un-rewrite previously rewritten claims — alias commitments
        mirror :meth:`add_alias` semantics.
        """
        f = self.normalize_subject(from_predicate)
        with self.tx() as cx:
            cur = cx.execute(
                "DELETE FROM predicate_aliases "
                "WHERE domain = ? AND alias_from = ?",
                (domain, f),
            )
            return {"removed": cur.rowcount > 0}

    # ---------- WS-E.2: predicate senses ----------

    def add_predicate_sense(
        self, *, canonical: str, sense_tag: str, domain: str,
        definition: Optional[str] = None,
        parent_id: Optional[int] = None,
        inverse_id: Optional[int] = None,
        is_symmetric: bool = False,
        is_transitive: bool = False,
    ) -> int:
        """Insert a new sense row.

        ``canonical`` is normalized via :meth:`normalize_subject`. The
        ``UNIQUE(canonical, sense_tag, domain)`` constraint enforces "one
        catalog entry per triple"; duplicates raise ``ValueError``. Validates
        ``parent_id`` and ``inverse_id`` reference existing senses.
        """
        if domain not in self._VALID_DOMAINS:
            raise ValueError(
                f"Unknown domain {domain!r}; expected one of "
                f"{sorted(self._VALID_DOMAINS)}"
            )
        canonical_n = self.normalize_subject(canonical)
        if parent_id is not None:
            if not self.get_predicate_sense(parent_id):
                raise ValueError(f"parent_id {parent_id} does not exist")
        if inverse_id is not None:
            if not self.get_predicate_sense(inverse_id):
                raise ValueError(f"inverse_id {inverse_id} does not exist")
        try:
            with self.tx() as cx:
                cur = cx.execute(
                    "INSERT INTO predicate_senses "
                    "(canonical, sense_tag, domain, definition, parent_id, "
                    " inverse_id, is_symmetric, is_transitive, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (canonical_n, sense_tag, domain, definition,
                     parent_id, inverse_id,
                     1 if is_symmetric else 0,
                     1 if is_transitive else 0,
                     time.time()),
                )
                return cur.lastrowid
        except sqlite3.IntegrityError as e:
            raise ValueError(f"duplicate_sense: {e}")

    def get_predicate_sense(self, sense_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM predicate_senses WHERE id = ?", (sense_id,),
        ).fetchone()

    def list_predicate_senses(
        self, *, canonical: Optional[str] = None,
        domain: Optional[str] = None, limit: int = 200,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM predicate_senses WHERE 1=1"
        params: list = []
        if canonical is not None:
            sql += " AND canonical = ?"
            params.append(self.normalize_subject(canonical))
        if domain is not None:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " ORDER BY canonical, domain, sense_tag LIMIT ?"
        params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def set_claim_predicate_sense(
        self, claim_id: int, sense_id: int, *,
        assigned_by: str, confidence: float,
        explicit: bool = False, rationale: Optional[str] = None,
    ) -> None:
        """Upsert a claim's sense assignment.

        Honesty contract (mirrors ``add_claim_condition``):
        - NEVER silently flips ``explicit=0 -> explicit=1``. Promotion
          requires the caller to pass ``explicit=True``.
        - NEVER silently demotes ``explicit=1 -> explicit=0``. If the
          existing row is ``explicit=1`` and the caller passes
          ``explicit=False``, raises ``ValueError`` with code
          ``would_silently_demote``. The caller must
          :meth:`unset_claim_predicate_sense` first.

        Invalidates cached views citing the claim on every successful write.
        """
        if assigned_by not in {"agent", "llm", "human"}:
            raise ValueError(
                f"assigned_by must be agent|llm|human, got {assigned_by!r}"
            )
        if not self.get_predicate_sense(sense_id):
            raise ValueError(f"sense_id {sense_id} does not exist")
        if not self.get_claim(claim_id):
            raise ValueError(f"claim_id {claim_id} does not exist")
        existing = self.conn.execute(
            "SELECT explicit FROM claim_predicate_senses WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        if existing is not None and existing["explicit"] and not explicit:
            raise ValueError("would_silently_demote")
        with self.tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO claim_predicate_senses "
                "(claim_id, sense_id, assigned_by, assigned_at, "
                " confidence, explicit, rationale) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (claim_id, sense_id, assigned_by, time.time(),
                 confidence, 1 if explicit else 0, rationale),
            )
            _invalidate_cache_for_claims(
                cx, [claim_id], cause="predicate_sense_set",
            )

    def unset_claim_predicate_sense(self, claim_id: int) -> bool:
        """Delete the row. Returns True if a row was deleted. Invalidates
        cached views citing the claim."""
        with self.tx() as cx:
            cur = cx.execute(
                "DELETE FROM claim_predicate_senses WHERE claim_id = ?",
                (claim_id,),
            )
            removed = cur.rowcount > 0
            if removed:
                _invalidate_cache_for_claims(
                    cx, [claim_id], cause="predicate_sense_unset",
                )
            return removed

    def get_claim_predicate_sense(
        self, claim_id: int,
    ) -> Optional[sqlite3.Row]:
        """Return the joined sense row for a claim, or None."""
        return self.conn.execute(
            "SELECT cps.*, ps.canonical, ps.sense_tag, ps.domain, "
            "  ps.definition, ps.parent_id, ps.inverse_id, "
            "  ps.is_symmetric, ps.is_transitive "
            "FROM claim_predicate_senses cps "
            "JOIN predicate_senses ps ON ps.id = cps.sense_id "
            "WHERE cps.claim_id = ?",
            (claim_id,),
        ).fetchone()

    def claims_with_sense(self, sense_id: int) -> list[int]:
        """All claim IDs whose ``claim_predicate_senses`` row points at this
        sense. Used to cascade staleness on sense reparenting/invalidation."""
        rows = self.conn.execute(
            "SELECT claim_id FROM claim_predicate_senses WHERE sense_id = ?",
            (sense_id,),
        ).fetchall()
        return [r["claim_id"] for r in rows]

    def close(self) -> None:
        self.conn.close()
