"""Phase 3: contradiction- and condition-aware expansion, and unresolved
conflicts shown to the synthesizer.

Retrieval finds claims by their words. A claim that contradicts, or scopes,
a retrieved claim may share none of those words; without expansion the
synthesizer never sees it and the answer silently omits it."""
from __future__ import annotations

import json

import pytest

from aleph import authority
from aleph.cli import main
from aleph.db import Store
from aleph.query import SYNTHESIZE_V2_SYSTEM, query


class RecordingLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def complete(self, system, user, max_tokens=2048):
        self.prompts.append(user)
        return self.answer

    def complete_json(self, system, user, max_tokens=4096):
        return {"verdict": "GROUNDED", "reason": "ok"}


def _claim(store, subject, obj):
    sid = store.add_source(f"{subject}-{obj}.txt", obj)
    return store.add_claim(sid, subject, "states", obj, 0, len(obj), 0.9), sid


@pytest.fixture
def kb(tmp_path):
    store = Store(tmp_path / "t.db")
    a, _ = _claim(store, "pack", "Packs retain ninety percent capacity.")
    # shares no keyword with the question below
    b, b_src = _claim(store, "module", "Modules degrade quickly in heat.")
    c, _ = _claim(store, "fleet sample", "Measured on 300 taxis in Norway.")
    store.add_contradiction(a, b)
    store.add_claim_condition(a, c, "sample", explicit=True, confidence=0.9)
    yield store, a, b, c, b_src
    store.close()


def _ask(store, llm, **kw):
    return query(store, llm, "pack retain capacity", use_cache=False, **kw)


def test_retrieval_alone_misses_the_conflicting_claim(kb):
    store, a, b, c, _ = kb
    rows = store.search_claims_fts(["pack", "retain", "capacity"])
    assert [r["id"] for r in rows] == [a]


def test_contradiction_partner_and_condition_are_shown_with_a_reason(kb):
    store, a, b, c, _ = kb
    llm = RecordingLLM(f"Packs retain ninety percent capacity [claim:{a}].")
    r = _ask(store, llm)
    prompt = llm.prompts[0]
    assert f"[claim:{b}]" in prompt and f"conflicts with [claim:{a}]" in prompt
    assert f"[claim:{c}]" in prompt and f"sample condition of [claim:{a}]" in prompt
    assert f"conditions: [claim:{c}] (sample, explicit)" in prompt
    assert r.claim_ids_unused == sorted({b, c})


def test_unresolved_conflict_is_in_the_dispositions_block(kb):
    store, a, b, c, _ = kb
    llm = RecordingLLM(f"x [claim:{a}].")
    _ask(store, llm)
    assert f"- unresolved: [claim:{a}] vs [claim:{b}]" in llm.prompts[0]
    assert "- unresolved:" in SYNTHESIZE_V2_SYSTEM


def test_expansion_respects_the_context_filter(kb):
    store, a, b, c, b_src = kb
    authority.set_metadata(store, b_src, "scientific", {"peer_reviewed": True})
    authority.retract_source(store, b_src, "fabricated")
    llm = RecordingLLM(f"x [claim:{a}].")
    _ask(store, llm)
    assert f"[claim:{b}]" not in llm.prompts[0]


def test_expansion_can_be_turned_off(kb):
    store, a, b, c, _ = kb
    llm = RecordingLLM(f"x [claim:{a}].")
    _ask(store, llm, expand=False)
    assert f"[claim:{b}]" not in llm.prompts[0]


def test_expansion_is_capped(tmp_path):
    store = Store(tmp_path / "cap.db")
    a, _ = _claim(store, "pack", "Packs retain ninety percent capacity.")
    others = [_claim(store, f"module{i}", f"Module {i} degrades.")[0] for i in range(12)]
    for o in others:
        store.add_contradiction(a, o)
    llm = RecordingLLM(f"x [claim:{a}].")
    query(store, llm, "pack retain capacity", use_cache=False, retrieve_k=8)
    shown = [o for o in others if f"[claim:{o}] (" in llm.prompts[0]]
    assert len(shown) == 4  # retrieve_k // 2
    store.close()


def test_compose_includes_expanded_claims_and_unresolved(kb, capsys):
    store, a, b, c, _ = kb
    capsys.readouterr()
    assert main(["--db", str(store.db_path), "compose", "--query", "pack retain capacity"]) == 0
    d = json.loads(capsys.readouterr().out)["data"]
    by_id = {r["claim_id"]: r for r in d["retrieved_claims"]}
    assert by_id[a]["included_because"] is None
    assert by_id[b]["included_because"] == f"conflicts with [claim:{a}]"
    assert by_id[c]["included_because"] == f"sample condition of [claim:{a}]"
    assert d["dispositions"]["unresolved"] == [{"claim_a": a, "claim_b": b}]


def test_unexpanded_views_stay_out_of_the_cache(kb):
    store, a, b, c, _ = kb
    llm = RecordingLLM(f"x [claim:{a}].")
    query(store, llm, "pack retain capacity", expand=False)
    assert store.stats()["cached_views"] == 0
    query(store, llm, "pack retain capacity")
    assert query(store, llm, "pack retain capacity").from_cache
    assert not query(store, llm, "pack retain capacity", expand=False).from_cache


def test_keep_resolved_partner_is_not_expanded_or_called_unresolved(kb):
    store, a, b, c, _ = kb
    store.conn.execute("UPDATE contradictions SET status='resolved', resolved_to=?", (a,))
    store.conn.commit()
    llm = RecordingLLM(f"x [claim:{a}].")
    query(store, llm, "pack retain capacity", use_cache=False)
    assert f"[claim:{b}]" not in llm.prompts[0]
    assert "- unresolved:" not in llm.prompts[0]


def test_dispositions_are_found_for_old_contradictions(tmp_path):
    # The grouping used to read only the newest 1000 contradictions store-wide.
    from itertools import combinations
    from aleph.query import _claims_with_spans, _group_by_disposition
    store = Store(tmp_path / "many.db")
    a, _ = _claim(store, "pack", "Packs retain ninety percent capacity.")
    b, _ = _claim(store, "module", "Modules degrade quickly in heat.")
    store.add_contradiction(a, b)
    store.conn.execute("UPDATE contradictions SET detected_at = 0")
    filler = [_claim(store, f"f{i}", f"Filler {i}.")[0] for i in range(47)]
    store.conn.executemany(
        "INSERT INTO contradictions (claim_a_id, claim_b_id, detected_at) VALUES (?, ?, 1e12)",
        list(combinations(filler, 2)))  # 1081 newer contradictions
    store.conn.commit()
    rows = _claims_with_spans(store, [a, b], False)
    assert _group_by_disposition(store, rows)["unresolved"] == [(a, b)]
    store.close()
