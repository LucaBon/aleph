"""Phase 3: evidence considered but not used, and the counter-evidence check.

Both are computed when a view is read, never baked into cached prose: a
claim retracted or a contradiction recorded after caching shows up (or
drops out) on the next read without invalidating the view."""
from __future__ import annotations

import json

import pytest

from aleph.cli import main
from aleph.contradictions import dispose
from aleph.db import Store
from aleph.query import counter_evidence, query

TEXT_A = "Packs retain 90% capacity after 200,000 miles."
TEXT_B = "Packs retain 70% capacity after 200,000 miles."
TEXT_C = "Packs are warrantied for 8 years."


class AskLLM:
    def __init__(self, answer):
        self.answer = answer

    def complete(self, system, user, max_tokens=2048):
        self.last_user = user
        return self.answer

    def complete_json(self, system, user, max_tokens=4096):
        return {"verdict": "GROUNDED", "reason": "ok"}


@pytest.fixture
def kb(tmp_path):
    store = Store(tmp_path / "t.db")
    ids = []
    for i, text in enumerate([TEXT_A, TEXT_B, TEXT_C]):
        sid = store.add_source(f"s{i}.txt", text)
        ids.append(store.add_claim(sid, "pack", "retains", text, 0, len(text), 0.9))
    yield store, ids
    store.close()


def test_unused_claims_are_reported(kb):
    store, (a, b, c) = kb
    r = query(store, AskLLM(f"Packs retain 90% capacity [claim:{a}]."), "pack retains capacity")
    assert r.claim_ids_used == [a]
    assert r.claim_ids_unused == sorted({b, c})


def test_unused_claims_survive_the_cache_and_drop_inactive_ones(kb):
    store, (a, b, c) = kb
    llm = AskLLM(f"Packs retain 90% capacity [claim:{a}].")
    query(store, llm, "pack retains capacity")
    store.remove_source(store.get_claim(c)["source_id"])  # c is unused: view stays
    hit = query(store, llm, "pack retains capacity")
    assert hit.from_cache
    assert hit.claim_ids_unused == [b]


def test_counter_evidence_flags_an_uncited_side_of_an_open_contradiction(kb):
    store, (a, b, c) = kb
    x = store.add_contradiction(a, b)
    [ce] = counter_evidence(store, [a])
    assert ce == {"cited_claim_id": a, "counter_claim_id": b, "contradiction_id": x,
                  "disposition": "unresolved"}
    assert counter_evidence(store, [a, b]) == []  # both sides cited


@pytest.mark.parametrize("disposition,kwargs,flagged", [
    ("dispute", {}, True),
    ("gap", {}, True),
    ("coexist", {"rule": "different climates"}, False),
    ("replicate", {}, False),
])
def test_counter_evidence_by_disposition(kb, disposition, kwargs, flagged):
    store, (a, b, c) = kb
    x = store.add_contradiction(a, b)
    dispose(store, x, disposition, **kwargs)
    assert bool(counter_evidence(store, [a])) is flagged


def test_superseded_counter_claim_is_not_counter_evidence(kb):
    store, (a, b, c) = kb
    x = store.add_contradiction(a, b)
    dispose(store, x, "supersede", keep=a, drop=b)
    assert counter_evidence(store, [a]) == []


def test_counter_evidence_on_a_cached_view_tracks_the_counter_claim(kb):
    # The counter claim isn't cited, so retracting it doesn't invalidate the
    # view; the read-time check drops it on the next (cached) read.
    from aleph import authority
    store, (a, b, c) = kb
    store.add_contradiction(a, b)
    llm = AskLLM(f"Packs retain 90% capacity [claim:{a}].")
    assert [ce["counter_claim_id"] for ce in query(store, llm, "pack retains capacity").counter_evidence] == [b]
    sid = store.get_claim(b)["source_id"]
    authority.set_metadata(store, sid, "scientific", {"peer_reviewed": True})
    authority.retract_source(store, sid, "bad")
    hit = query(store, llm, "pack retains capacity")
    assert hit.from_cache and hit.counter_evidence == []


def test_agent_counter_evidence_command(kb, capsys):
    store, (a, b, c) = kb
    store.add_contradiction(a, b)
    capsys.readouterr()
    assert main(["--db", str(store.db_path), "counter-evidence", "--claim-ids", f"{a},{c}"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert [ce["counter_claim_id"] for ce in env["data"]["counter_evidence"]] == [b]


def test_legacy_resolved_contradiction_is_not_counter_evidence(kb):
    store, (a, b, c) = kb
    x = store.add_contradiction(a, b)
    # `contradiction-resolve --keep a` without --drop: status resolved,
    # disposition left at its 'unresolved' default.
    store.conn.execute("UPDATE contradictions SET status='resolved', resolved_to=? WHERE id=?", (a, x))
    store.conn.commit()
    assert counter_evidence(store, [a]) == []


def test_reopened_contradiction_is_counter_evidence_again(kb):
    store, (a, b, c) = kb
    x = store.add_contradiction(a, b)
    concept = store.add_concept("pack", "s", "summary", 0.7, [(c, "premise")])
    store.attest_concept(concept, attested_by="t", rationale="r")
    dispose(store, x, "coexist", rule="climates", rationale_concept_id=concept)
    assert counter_evidence(store, [a]) == []
    store.remove_source(store.get_claim(c)["source_id"])  # rationale goes stale -> reopen
    assert [ce["counter_claim_id"] for ce in counter_evidence(store, [a])] == [b]


def test_cached_view_is_invalidated_when_a_cited_claims_contradiction_changes(kb):
    store, (a, b, c) = kb
    llm = AskLLM(f"Packs retain 90% capacity [claim:{a}].")
    query(store, llm, "pack retains capacity")
    x = store.add_contradiction(a, b)
    assert not query(store, llm, "pack retains capacity").from_cache
    assert query(store, llm, "pack retains capacity").from_cache
    dispose(store, x, "dispute")
    assert not query(store, llm, "pack retains capacity").from_cache


def test_unverified_views_are_not_cached(kb):
    store, (a, b, c) = kb
    llm = AskLLM(f"Packs retain 90% capacity [claim:{a}].")
    query(store, llm, "pack retains capacity", verify=False)
    assert store.stats()["cached_views"] == 0


def test_cached_unused_respects_include_retracted(kb, tmp_path):
    from aleph import authority
    store, (a, b, c) = kb
    sid = store.get_claim(b)["source_id"]
    authority.set_metadata(store, sid, "scientific", {"peer_reviewed": True})
    authority.retract_source(store, sid, "bad")
    llm = AskLLM(f"Packs retain 90% capacity [claim:{a}].")
    ctx = {"include_retracted": True}
    fresh = query(store, llm, "pack retains capacity", context=ctx)
    hit = query(store, llm, "pack retains capacity", context=ctx)
    assert hit.from_cache and hit.claim_ids_unused == fresh.claim_ids_unused


def test_counter_evidence_respects_the_query_context(kb):
    from aleph import authority
    store, (a, b, c) = kb
    authority.set_metadata(store, store.get_claim(a)["source_id"], "legal",
                           {"jurisdiction": "US-CA", "authority_level": 1})
    authority.set_metadata(store, store.get_claim(b)["source_id"], "legal",
                           {"jurisdiction": "DE", "authority_level": 1})
    store.add_contradiction(a, b)
    ctx = {"jurisdiction": "US-CA"}
    assert counter_evidence(store, [a], ctx) == []
    assert [ce["counter_claim_id"] for ce in counter_evidence(store, [a])] == [b]
    llm = AskLLM(f"Packs retain 90% capacity [claim:{a}].")
    assert query(store, llm, "pack retains capacity", context=ctx).counter_evidence == []


def test_counter_evidence_skips_differing_senses(kb):
    store, (a, b, c) = kb
    s1 = store.add_predicate_sense(canonical="retain", sense_tag="capacity", domain="generic")
    s2 = store.add_predicate_sense(canonical="retain", sense_tag="customers", domain="generic")
    store.set_claim_predicate_sense(a, s1, assigned_by="agent", confidence=0.9, explicit=True)
    store.set_claim_predicate_sense(b, s2, assigned_by="agent", confidence=0.9, explicit=True)
    store.add_contradiction(a, b)
    assert counter_evidence(store, [a]) == []
