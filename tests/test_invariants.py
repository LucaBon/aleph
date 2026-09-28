"""Property-based test of the retraction-cascade invariant.

Drives a Store through random sequences of the mutations an agent can make
(add/remove/retract sources, add/supersede claims, aliases and their undo,
concepts, contradiction dispositions, cached views) and checks after every
step that `Store.check_invariants()` reports nothing: no cached view,
active/attested concept, resolved contradiction or supersession rests on a
claim that is no longer active.

Run: pytest tests/test_invariants.py
Bump examples: HYPOTHESIS_PROFILE=thorough pytest tests/test_invariants.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from aleph import authority
from aleph.contradictions import dispose
from aleph.db import Store
from aleph.query import _cache_view

settings.register_profile(
    "default", max_examples=200, stateful_step_count=40, deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "thorough", max_examples=1000, stateful_step_count=60, deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

SUBJECTS = ["battery", "batteries", "cell", "pack", "motor"]
WORDS = ["retains", "capacity", "ninety", "percent", "after", "miles", "degrades", "slowly"]


class CascadeMachine(RuleBasedStateMachine):
    @initialize()
    def setup(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._tmp.name) / "t.db")
        self.n_sources = 0

    def teardown(self):
        self.store.close()
        self._tmp.cleanup()

    # ---------- helpers ----------

    def _ids(self, sql, *params):
        return [r[0] for r in self.store.conn.execute(sql, params).fetchall()]

    def _active_claims(self):
        return self._ids("SELECT id FROM claims WHERE status = 'active' ORDER BY id")

    def _sources(self):
        return self._ids("SELECT id FROM sources ORDER BY id")

    # ---------- sources ----------

    @rule(words=st.lists(st.sampled_from(WORDS), min_size=3, max_size=8),
          scientific=st.booleans())
    def add_source(self, words, scientific):
        self.n_sources += 1
        content = f"doc {self.n_sources}: " + " ".join(words) + "."
        sid = self.store.add_source(f"s{self.n_sources}.txt", content)
        if sid is not None and scientific:
            self.store.set_source_metadata(sid, "scientific", {"peer_reviewed": True})

    @precondition(lambda self: self._sources())
    @rule(data=st.data())
    def remove_source(self, data):
        sid = data.draw(st.sampled_from(self._sources()))
        self.store.remove_source(sid)

    @precondition(lambda self: self._sources())
    @rule(data=st.data())
    def retract_source(self, data):
        sid = data.draw(st.sampled_from(self._sources()))
        authority.retract_source(self.store, sid, "test")

    @precondition(lambda self: self._sources())
    @rule(data=st.data())
    def unretract_source(self, data):
        sid = data.draw(st.sampled_from(self._sources()))
        meta = self.store.get_source_metadata(sid)
        if meta and meta["metadata"].get("retracted"):
            authority.unretract_source(self.store, sid, "test")

    # ---------- claims ----------

    @precondition(lambda self: self._sources())
    @rule(data=st.data(), subject=st.sampled_from(SUBJECTS))
    def add_claim(self, data, subject):
        sid = data.draw(st.sampled_from(self._sources()))
        content = self.store.get_source(sid)["content"]
        start = data.draw(st.integers(0, len(content) - 1))
        end = data.draw(st.integers(start + 1, len(content)))
        self.store.add_claim(sid, subject, "has", content[start:end], start, end, 0.8)

    @precondition(lambda self: len(self._active_claims()) >= 2)
    @rule(data=st.data())
    def supersede(self, data):
        old, new = data.draw(st.lists(st.sampled_from(self._active_claims()),
                                      min_size=2, max_size=2, unique=True))
        self.store.supersede_claim(old, new)

    # ---------- aliases ----------

    @rule(a=st.sampled_from(SUBJECTS), b=st.sampled_from(SUBJECTS), force=st.booleans())
    def add_alias(self, a, b, force):
        self.store.add_alias(a, b, force=force)

    @rule(a=st.sampled_from(SUBJECTS))
    def undo_alias(self, a):
        self.store.undo_alias(a)

    # ---------- concepts ----------

    @precondition(lambda self: self._active_claims())
    @rule(data=st.data(), attest=st.booleans())
    def add_concept(self, data, attest):
        supports = data.draw(st.lists(st.sampled_from(self._active_claims()),
                                      min_size=1, max_size=3, unique=True))
        cid = self.store.add_concept("battery", "synthesis", "summary", 0.7,
                                     [(s, "premise") for s in supports])
        if attest:
            self.store.attest_concept(cid, attested_by="test", rationale="checked")

    @precondition(lambda self: self._ids(
        "SELECT id FROM concepts WHERE status IN ('active','attested')"))
    @rule(data=st.data(), status=st.sampled_from(["superseded", "invalidated", "stale"]))
    def retire_concept(self, data, status):
        cid = data.draw(st.sampled_from(self._ids(
            "SELECT id FROM concepts WHERE status IN ('active','attested')")))
        self.store.update_concept_status(cid, status)

    # ---------- contradictions ----------

    @precondition(lambda self: len(self._active_claims()) >= 2)
    @rule(data=st.data())
    def add_contradiction(self, data):
        a, b = data.draw(st.lists(st.sampled_from(self._active_claims()),
                                  min_size=2, max_size=2, unique=True))
        self.store.add_contradiction(a, b)

    @precondition(lambda self: self._ids(
        "SELECT id FROM contradictions WHERE status = 'open'"))
    @rule(data=st.data(), disposition=st.sampled_from(
        ["supersede", "retracted", "replicate", "coexist", "dispute", "gap"]))
    def dispose_contradiction(self, data, disposition):
        ct_id = data.draw(st.sampled_from(self._ids(
            "SELECT id FROM contradictions WHERE status = 'open'")))
        row = self.store.conn.execute(
            "SELECT claim_a_id, claim_b_id FROM contradictions WHERE id = ?", (ct_id,)
        ).fetchone()
        keep, drop = data.draw(st.permutations([row[0], row[1]]))
        kwargs = {}
        if disposition == "supersede":
            kwargs = {"keep": keep, "drop": drop}
        elif disposition == "retracted":
            kwargs = {"drop": drop}
        elif disposition == "coexist":
            kwargs = {"rule": "different scope"}
            citable = self._ids("SELECT id FROM concepts WHERE status IN ('active','attested')")
            if citable and data.draw(st.booleans()):
                kwargs["rationale_concept_id"] = data.draw(st.sampled_from(citable))
        try:
            dispose(self.store, ct_id, disposition, **kwargs)
        except ValueError as e:
            # Refusals are fine (e.g. a member already inactive); silent
            # acceptance of a bad state is what the invariant catches.
            assert "inactive" in str(e) or "not_active" in str(e), e

    # ---------- views ----------

    @precondition(lambda self: self._active_claims())
    @rule(data=st.data())
    def cache_view(self, data):
        claims = data.draw(st.lists(st.sampled_from(self._active_claims()),
                                    min_size=1, max_size=3, unique=True))
        concepts = self._ids("SELECT id FROM concepts WHERE status IN ('active','attested')")
        cited_concepts = data.draw(st.lists(st.sampled_from(concepts), max_size=2,
                                            unique=True)) if concepts else []
        q = f"q{data.draw(st.integers(0, 5))}"
        _cache_view(self.store, q, q, "answer", claims, cited_concepts)

    # ---------- the invariant ----------

    @invariant()
    def cascade_invariant_holds(self):
        if not hasattr(self, "store"):
            return
        violations = self.store.check_invariants()
        assert violations == [], violations


TestCascade = CascadeMachine.TestCase


# ---------- targeted scenarios ----------
#
# Deep states the random walk reaches rarely. Each one is a leakage channel:
# something derived from a claim that outlives the claim.

import pytest


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def _claim(store, sid, text, subject="battery"):
    content = store.get_source(sid)["content"]
    i = content.index(text)
    return store.add_claim(sid, subject, "has", text, i, i + len(text), 0.8)


def _scientific_source(store, name, content):
    sid = store.add_source(name, content)
    store.set_source_metadata(sid, "scientific", {"peer_reviewed": True})
    return sid


def test_supersede_resolution_reopens_when_kept_claim_is_retracted(store):
    real = store.add_source("real.txt", "capacity is 90 percent")
    canary = _scientific_source(store, "canary.txt", "capacity is 47 percent")
    r = _claim(store, real, "90 percent")
    c = _claim(store, canary, "47 percent")
    ct = store.add_contradiction(r, c)
    dispose(store, ct, "supersede", keep=c, drop=r)
    assert store.get_claim(r)["status"] == "superseded"

    authority.retract_source(store, canary, "fabricated")

    assert store.get_claim(r)["status"] == "active"
    row = store.conn.execute("SELECT * FROM contradictions WHERE id = ?", (ct,)).fetchone()
    assert row["disposition"] == "unresolved" and row["reopened_from"] == "supersede"
    assert store.check_invariants() == []


def test_replicate_bump_is_reverted_on_reopen(store):
    real = store.add_source("real.txt", "capacity is 90 percent")
    canary = _scientific_source(store, "canary.txt", "capacity is ninety percent")
    r = _claim(store, real, "90 percent")
    c = _claim(store, canary, "ninety percent")
    ct = store.add_contradiction(r, c)
    dispose(store, ct, "replicate")
    assert store.get_claim(r)["confidence"] == pytest.approx(0.85)

    authority.retract_source(store, canary, "fabricated")

    assert store.get_claim(r)["confidence"] == pytest.approx(0.8)
    assert store.check_invariants() == []


def test_rationale_concept_going_stale_reopens_resolution(store):
    sid = store.add_source("a.txt", "cells degrade slowly; packs degrade fast; cold slows it")
    a = _claim(store, sid, "cells degrade slowly")
    b = _claim(store, sid, "packs degrade fast")
    k = _claim(store, sid, "cold slows it")
    concept = store.add_concept("battery", "scope differs", "synthesis", 0.7, [(k, "premise")])
    store.attest_concept(concept, attested_by="test", rationale="checked")
    ct = store.add_contradiction(a, b)
    dispose(store, ct, "coexist", rule="different units", rationale_concept_id=concept)

    other = _claim(store, sid, "degrade fast")
    store.supersede_claim(k, other)

    row = store.conn.execute("SELECT * FROM contradictions WHERE id = ?", (ct,)).fetchone()
    assert row["disposition"] == "unresolved" and row["reopened_from"] == "coexist"
    assert store.check_invariants() == []


def test_remove_source_revives_claims_it_superseded_elsewhere(store):
    old = store.add_source("old.txt", "retention is 30 days")
    new = store.add_source("new.txt", "retention is 90 days")
    o = _claim(store, old, "30 days")
    n = _claim(store, new, "90 days")
    store.supersede_claim(o, n)

    store.remove_source(new)  # used to fail the superseded_by foreign key

    assert store.get_claim(o)["status"] == "active"
    assert store.check_invariants() == []


def test_alias_undo_restores_subjects_and_invalidates_views(store):
    sid = store.add_source("a.txt", "the pack holds 75 kWh")
    cid = _claim(store, sid, "75 kWh", subject="pack")
    store.add_alias("pack", "battery")
    assert store.get_claim(cid)["subject"] == "battery"
    store.cache_view("how big?", f"75 kWh [claim:{cid}]", [cid])

    out = store.undo_alias("pack")

    assert out["claims_restored"] == [cid]
    assert store.get_claim(cid)["subject"] == "pack"
    assert store.resolve_subject("pack") == "pack"
    assert store.get_cached_view("how big?") is None
    assert store.check_invariants() == []


def test_alias_overwrite_requires_force_and_cannot_cycle(store):
    store.add_alias("cell", "pack")
    assert store.add_alias("cell", "battery")["note"] == "alias-exists"
    store.add_alias("battery", "cell")  # battery -> cell -> pack
    assert store.add_alias("cell", "battery", force=True)["note"] == "would-create-cycle"


# ---------- PR #1 review regressions ----------


def test_predicate_alias_cycle_through_chain_is_refused(store):
    store.add_predicate_alias("*", "a", "x")
    store.add_predicate_alias("*", "b", "a")  # b -> a -> x
    assert store.add_predicate_alias("*", "a", "b")["note"] == "would-create-cycle"


def test_predicate_alias_rewrites_claims_to_end_of_chain(store):
    sid = store.add_source("a.txt", "the pack holds 75 kWh")
    cid = _claim(store, sid, "75 kWh")  # predicate "has"
    store.add_predicate_alias("*", "holds", "contains")
    store.add_predicate_alias("*", "has", "holds")  # has -> holds -> contains
    assert store.get_claim(cid)["predicate"] == store.normalize_subject("contains")


def test_redisposing_replicate_never_compounds_confidence(store):
    sid = store.add_source("a.txt", "capacity is 90 percent; capacity is ninety percent")
    a = _claim(store, sid, "90 percent")
    b = _claim(store, sid, "ninety percent")
    ct = store.add_contradiction(a, b)
    dispose(store, ct, "replicate")
    dispose(store, ct, "replicate")
    assert store.get_claim(a)["confidence"] == pytest.approx(0.85)
    dispose(store, ct, "dispute")
    assert store.get_claim(a)["confidence"] == pytest.approx(0.8)
    assert store.check_invariants() == []


class _GroundedLLM:
    def complete_json(self, system, user):
        return {"verdict": "GROUNDED", "reason": "ok"}


def test_validate_does_not_revive_superseded_or_invalidated_concepts(store):
    from aleph.concepts import invalidate_concept, validate_concept

    sid = store.add_source("a.txt", "cells degrade slowly; packs degrade fast")
    a = _claim(store, sid, "cells degrade slowly")
    old = store.add_concept("battery", "old", "summary", 0.7, [(a, "premise")])
    new = store.add_concept("battery", "new", "summary", 0.7, [(a, "premise")])
    store.update_concept_status(old, "superseded", superseded_by=new)
    validate_concept(store, _GroundedLLM(), old)
    row = store.get_concept(old)
    assert row["status"] == "superseded" and row["superseded_by"] == new

    invalidate_concept(store, new, "wrong scope")
    validate_concept(store, _GroundedLLM(), new)
    row = store.get_concept(new)
    assert row["status"] == "invalidated"

    fresh = store.add_concept("battery", "fresh", "summary", 0.7, [(a, "premise")])
    validate_concept(store, _GroundedLLM(), fresh)
    assert store.get_concept(fresh)["status"] == "active"


def test_invalidate_records_reason(store):
    from aleph.concepts import invalidate_concept

    sid = store.add_source("a.txt", "cells degrade slowly")
    a = _claim(store, sid, "cells degrade slowly")
    c = store.add_concept("battery", "s", "summary", 0.7, [(a, "premise")])
    invalidate_concept(store, c, "wrong scope")
    assert store.get_concept(c)["validation_reason"] == "wrong scope"


def test_resolve_by_recency_leaves_pairs_with_inactive_members_open(store):
    from aleph.lint import resolve_by_recency

    old = store.add_source("old.txt", "retention is 30 days")
    new = _scientific_source(store, "new.txt", "retention is 90 days")
    o = _claim(store, old, "30 days")
    n = _claim(store, new, "90 days")
    ct = store.add_contradiction(o, n)
    authority.retract_source(store, new, "fabricated")

    assert resolve_by_recency(store) == 0
    row = store.conn.execute("SELECT * FROM contradictions WHERE id = ?", (ct,)).fetchone()
    assert row["status"] == "open" and row["resolved_to"] is None
    assert store.get_claim(o)["status"] == "active"
    assert store.check_invariants() == []


def test_unretract_keeps_claims_dropped_by_retracted_disposition(store):
    sid = _scientific_source(store, "a.txt", "capacity is 90 percent; capacity is 47 percent")
    good = _claim(store, sid, "90 percent")
    bad = _claim(store, sid, "47 percent")
    ct = store.add_contradiction(good, bad)
    dispose(store, ct, "retracted", keep=good, drop=bad)

    assert authority.unretract_source(store, sid, "oops")["error"] == "not_retracted"
    assert store.get_claim(bad)["status"] == "retracted"

    authority.retract_source(store, sid, "fabricated")
    authority.unretract_source(store, sid, "restored")
    assert store.get_claim(good)["status"] == "active"
    assert store.get_claim(bad)["status"] == "retracted"
    assert store.check_invariants() == []


def test_set_metadata_cannot_change_retraction_state(store):
    sid = _scientific_source(store, "a.txt", "capacity is 90 percent")
    cid = _claim(store, sid, "90 percent")

    out = authority.set_metadata(store, sid, "scientific",
                                 {"peer_reviewed": True, "retracted": True})
    assert out["error"] == "retraction_state_change"
    assert store.get_claim(cid)["status"] == "active"

    authority.retract_source(store, sid, "fabricated")
    out = authority.set_metadata(store, sid, "scientific", {"peer_reviewed": False})
    assert "error" not in out
    meta = store.get_source_metadata(sid)["metadata"]
    assert meta["retracted"] is True and meta["peer_reviewed"] is False
    out = authority.set_metadata(store, sid, "scientific", {"retracted": False})
    assert out["error"] == "retraction_state_change"


def test_retract_source_is_atomic(store, monkeypatch):
    sid = _scientific_source(store, "a.txt", "capacity is 90 percent")
    cid = _claim(store, sid, "90 percent")

    def boom(cx, source_id):
        raise RuntimeError("cascade failed")

    monkeypatch.setattr(store, "_retract_source_claims_tx", boom)
    with pytest.raises(RuntimeError):
        authority.retract_source(store, sid, "fabricated")
    assert not authority.is_retracted(store.get_source_metadata(sid))
    assert store.get_claim(cid)["status"] == "active"


def test_replace_source_refuses_duplicate_before_removing(store):
    sid = store.add_source("a.txt", "capacity is 90 percent")
    cid = _claim(store, sid, "90 percent")
    store.add_source("b.txt", "capacity is 47 percent")
    with pytest.raises(ValueError):
        store.replace_source("a.txt", "capacity is 47 percent")
    assert store.get_claim(cid)["status"] == "active"

    out = store.replace_source("a.txt", "capacity is 90 percent")
    assert out["status"] == "unchanged" and out["source_id"] == sid
    out = store.replace_source("a.txt", "capacity is 80 percent")
    assert out["status"] == "replaced" and out["old_source_ids"] == [sid]
    assert store.get_claim(cid) is None
    assert store.check_invariants() == []
