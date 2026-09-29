"""Phase 2 store layer: claim propositions, context windows, the review
queue, and fidelity flags raised at write time."""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from aleph import db as dbmod
from aleph.db import SCHEMA, SCHEMA_VERSION, Store, _run_alters

TEXT = (
    "Tesla was founded in 2003. Model S packs retain about 90% of their "
    "original capacity after 200,000 miles. Degradation is not linear."
)
SPAN = "Model S packs retain about 90% of their original capacity after 200,000 miles."


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def _add(store, *, obj="about 90% capacity after 200,000 miles", proposition=None,
         span=SPAN, source_text=TEXT):
    sid = store.add_source(f"s{time.time_ns()}.txt", source_text)
    start = source_text.index(span)
    return sid, store.add_claim(sid, "Model S pack", "retains", obj, start,
                                start + len(span), 0.9, proposition=proposition)


# ---------- schema ----------

def test_fresh_store_is_at_current_schema_version(store):
    assert SCHEMA_VERSION >= 2
    assert store.get_config("schema_version") == str(SCHEMA_VERSION)
    cols = {r[1] for r in store.conn.execute("PRAGMA table_info(claims)")}
    assert {"proposition", "context_start", "context_end"} <= cols


def test_v1_store_is_migrated_and_backfilled(tmp_path):
    path = tmp_path / "v1.db"
    cx = sqlite3.connect(path)
    cx.executescript(SCHEMA)
    _run_alters(cx)
    cx.execute("INSERT INTO sources (path, sha256, content, ingested_at) "
               "VALUES ('a', 'h', ?, 0)", (TEXT,))
    start = TEXT.index(SPAN)
    cx.execute("INSERT INTO claims (source_id, subject, predicate, object, span_start,"
               " span_end, confidence, extracted_at) VALUES (1,'s','p','o',?,?,0.9,0)",
               (start, start + len(SPAN)))
    cx.execute("INSERT INTO store_config VALUES ('schema_version', '1', 0)")
    cx.commit()
    cx.close()

    s = Store(path)
    try:
        assert s.get_config("schema_version") == str(SCHEMA_VERSION)
        row = s.get_claim(1)
        assert row["proposition"] is None  # never fabricated from the triple
        assert row["context_start"] <= start and start + len(SPAN) <= row["context_end"]
        assert s.get_context_text(1) == TEXT
    finally:
        s.close()


def test_pre_fts_store_is_indexed_on_open(tmp_path):
    # Claims written before the FTS table existed must be backfilled into
    # the index; otherwise the update trigger's 'delete' corrupts it.
    path = tmp_path / "prefts.db"
    cx = sqlite3.connect(path)
    cx.executescript(SCHEMA)
    _run_alters(cx)
    cx.execute("INSERT INTO sources (path, sha256, content, ingested_at) "
               "VALUES ('a', 'h', ?, 0)", (TEXT,))
    cx.execute("INSERT INTO claims (source_id, subject, predicate, object, span_start,"
               " span_end, confidence, extracted_at) VALUES (1,'pack','retains','90%',0,5,0.9,0)")
    cx.commit()
    cx.close()
    s = Store(path)
    try:
        if not s.fts_enabled:
            pytest.skip("SQLite build without FTS5")
        assert [r["id"] for r in s.search_claims_fts(["pack"])] == [1]
        s.conn.execute("UPDATE claims SET object = 'ninety' WHERE id = 1")
        s.conn.commit()
        assert s.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        s.conn.execute("INSERT INTO claims_fts(claims_fts) VALUES('integrity-check')")
    finally:
        s.close()


# ---------- proposition + context ----------

def test_add_claim_stores_proposition_and_context(store):
    _, cid = _add(store, proposition="Model S packs keep about 90% capacity after 200,000 miles.")
    row = store.get_claim(cid)
    assert row["proposition"].startswith("Model S packs keep")
    assert store.get_context_text(cid) == TEXT
    assert store.claim_text(cid) == row["proposition"]


def test_claim_text_falls_back_to_the_triple(store):
    _, cid = _add(store)
    row = store.get_claim(cid)
    assert store.claim_text(cid) == f"{row['subject']} {row['predicate']} {row['object']}"


# ---------- fidelity at write time ----------

def test_faithful_claim_raises_no_review_item(store):
    _add(store)
    assert store.list_reviews() == []


def test_unfaithful_claim_is_queued_for_review_not_refused(store):
    _, cid = _add(store, obj="about 80% capacity after 200,000 miles")
    assert store.get_claim(cid)["status"] == "active"
    [item] = store.list_reviews()
    assert item["item_type"] == "claim"
    assert item["claim_id"] == cid
    assert item["reason"] == "fidelity"
    issues = json.loads(item["details"])["issues"]
    assert [i["kind"] for i in issues] == ["number"]
    assert issues[0]["value"] == "80"


def test_fidelity_uses_the_context_window(store):
    # "Tesla" is outside the span but inside its context window.
    _, _cid = _add(store, proposition="Tesla's Model S packs retain about 90% capacity.")
    assert store.list_reviews() == []


def test_recheck_fidelity_of_existing_claim(store):
    _, cid = _add(store, proposition="Model S packs retain about 80% capacity.")
    assert [i.kind for i in store.check_claim_fidelity(cid)] == ["number"]


# ---------- review queue ----------

def test_enqueue_each_item_type_and_filter(store):
    sid, c1 = _add(store)
    _, c2 = _add(store, obj="90% after 200,000 miles", source_text=TEXT + " Copy.")
    x = store.add_contradiction(c1, c2)
    concept = store.add_concept("model s pack", "Packs age slowly.", "summary", 0.7,
                                [(c1, "premise")])
    store.add_alias("model s battery", "model s pack")

    store.enqueue_review("claim", c1, reason="manual")
    store.enqueue_review("contradiction", x, reason="manual")
    store.enqueue_review("concept", concept, reason="manual")
    store.enqueue_review("alias", "model s battery", reason="manual")

    assert {r["item_type"] for r in store.list_reviews()} == {
        "claim", "contradiction", "concept", "alias"}
    [a] = store.list_reviews(item_type="alias")
    assert a["alias_from"] == "model s battery"


def test_enqueue_rejects_unknown_type_and_missing_target(store):
    with pytest.raises(ValueError):
        store.enqueue_review("view", 1, reason="manual")
    with pytest.raises(LookupError):
        store.enqueue_review("claim", 999, reason="manual")
    with pytest.raises(LookupError):
        store.enqueue_review("alias", "no such alias", reason="manual")


def test_identical_open_item_is_not_duplicated(store):
    _, cid = _add(store)
    first = store.enqueue_review("claim", cid, reason="manual")
    assert store.enqueue_review("claim", cid, reason="manual") == first
    assert len(store.list_reviews()) == 1


def test_resolve_review(store):
    _, cid = _add(store)
    rid = store.enqueue_review("claim", cid, reason="manual", details={"why": "check"})
    store.resolve_review(rid, "rejected", resolved_by="luca", note="object is wrong")
    assert store.list_reviews() == []
    [r] = store.list_reviews(status="rejected")
    assert r["resolved_by"] == "luca" and r["resolution_note"] == "object is wrong"
    with pytest.raises(ValueError):
        store.resolve_review(rid, "accepted", resolved_by="luca")  # already closed
    with pytest.raises(ValueError):
        store.resolve_review(store.enqueue_review("claim", cid, reason="x"),
                             "maybe", resolved_by="luca")
    with pytest.raises(LookupError):
        store.resolve_review(12345, "accepted", resolved_by="luca")


def test_removing_a_source_drops_its_claim_reviews(store):
    sid, cid = _add(store, obj="about 80% capacity after 200,000 miles")
    assert len(store.list_reviews()) == 1
    store.remove_source(sid)
    assert store.list_reviews(status="all") == []
    assert store.check_invariants() == []


def test_undoing_an_alias_closes_its_open_reviews(store):
    _add(store)
    store.add_alias("model s battery", "model s pack")
    store.enqueue_review("alias", "model s battery", reason="manual")
    store.undo_alias("model s battery")
    assert store.list_reviews() == []
    [r] = store.list_reviews(status="obsolete")
    assert r["resolution_note"] == "alias undone"


def test_schema_version_constant_matches_migrations():
    assert SCHEMA_VERSION == max(v for v, _, _ in dbmod.MIGRATIONS)


def test_triple_fallback_does_not_check_the_subject(store):
    # Subjects are index labels ("aggravante art 577"); their numbers need not
    # appear in the span. The object is still checked.
    text = "La pena è l'ergastolo se il fatto è commesso contro il coniuge."
    sid = store.add_source("art577.txt", text)
    store.add_claim(sid, "aggravante art 577", "comporta", "ergastolo contro il coniuge",
                    0, len(text), 0.9)
    assert store.list_reviews() == []
    cid = store.add_claim(sid, "aggravante art 577", "comporta", "30 anni di reclusione",
                          0, len(text), 0.9)
    [item] = store.list_reviews()
    assert item["claim_id"] == cid
    assert [i.kind for i in store.check_claim_fidelity(cid)] == ["number"]


# ---------- review findings ----------

def test_accepted_fidelity_review_is_not_requeued_for_the_same_issues(store):
    _, cid = _add(store, obj="about 80% capacity after 200,000 miles")
    [item] = store.list_reviews()
    store.resolve_review(item["id"], "accepted", resolved_by="luca")
    issues = [i.to_dict() for i in store.check_claim_fidelity(cid)]
    assert store.enqueue_review("claim", cid, reason="fidelity",
                                details={"issues": issues}) == item["id"]
    assert store.list_reviews() == []
    # different issues are a new finding
    store.enqueue_review("claim", cid, reason="fidelity", details={"issues": [{"x": 1}]})
    assert len(store.list_reviews()) == 1


def test_alias_review_records_its_target_and_is_obsoleted_on_overwrite(store):
    _add(store)
    store.add_alias("model s battery", "model s pack")
    rid = store.enqueue_review("alias", "model s battery", reason="manual")
    assert json.loads(store.get_review(rid)["details"])["canonical_to"] == "model s pack"
    store.add_alias("model s battery", "model x pack", force=True)
    assert store.get_review(rid)["status"] == "obsolete"
    rid2 = store.enqueue_review("alias", "model s battery", reason="manual")
    store.undo_alias("model s battery")  # restores the previous target
    assert store.get_review(rid2)["status"] == "obsolete"
    assert store.check_invariants() == []


def test_resolve_review_refuses_a_concurrently_closed_item(store):
    _, cid = _add(store)
    rid = store.enqueue_review("claim", cid, reason="manual")
    store.conn.execute("UPDATE review_queue SET status = 'rejected' WHERE id = ?", (rid,))
    store.conn.commit()
    with pytest.raises(ValueError):
        store.resolve_review(rid, "accepted", resolved_by="x")


def test_only_one_open_review_per_target_and_reason(store):
    _, cid = _add(store)
    store.enqueue_review("claim", cid, reason="manual")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO review_queue (item_type, claim_id, reason, created_at) "
            "VALUES ('claim', ?, 'manual', 0)", (cid,))


def test_migration_2_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "v1.db"
    cx = sqlite3.connect(path)
    cx.executescript(SCHEMA)
    _run_alters(cx)
    cx.execute("INSERT INTO sources (path, sha256, content, ingested_at) VALUES ('a','h',?,0)", (TEXT,))
    for _ in range(2):
        cx.execute("INSERT INTO claims (source_id, subject, predicate, object, span_start,"
                   " span_end, confidence, extracted_at) VALUES (1,'s','p','o',0,5,0.9,0)")
    cx.execute("INSERT INTO store_config VALUES ('schema_version', '1', 0)")
    cx.commit()
    cx.close()
    calls = []

    def boom(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("mid-migration failure")
        return 0, 5
    monkeypatch.setattr(dbmod, "_context_window_v2", boom)
    with pytest.raises(RuntimeError):
        Store(path)
    cx = sqlite3.connect(path)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(claims)")}
    tables = {r[0] for r in cx.execute("SELECT name FROM sqlite_master")}
    cx.close()
    assert "proposition" not in cols and "review_queue" not in tables
