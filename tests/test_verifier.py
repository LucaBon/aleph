"""Layered verifier (Phase 2): deterministic -> entailment -> LLM, with
SUPPORTED / UNSUPPORTED / UNCERTAIN verdicts. Each layer either decides or
passes to the next; the first decision wins."""
from __future__ import annotations

import pytest

from aleph.verifier import (
    SUPPORTED, UNCERTAIN, UNSUPPORTED, LayeredVerifier, baseline_verdict,
)

SPAN = "Model S packs retain about 90% of their original capacity after 200,000 miles."


class ScriptedLLM:
    def __init__(self, response=None, exc=None):
        self.response, self.exc = response, exc
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, system, user, max_tokens=4096):
        self.prompts.append((system, user))
        if self.exc:
            raise self.exc
        return self.response


class FixedEntailer:
    def __init__(self, entailment=0.0, contradiction=0.0):
        self.scores = {"entailment": entailment, "contradiction": contradiction,
                       "neutral": max(0.0, 1 - entailment - contradiction)}
        self.calls = 0

    def predict(self, premise, hypothesis):
        self.calls += 1
        return self.scores


# ---------- deterministic layer ----------

def test_verbatim_sentence_is_supported_without_an_llm():
    llm = ScriptedLLM({"verdict": "UNSUPPORTED"})
    r = LayeredVerifier(llm).verify(
        "Model S packs retain about 90% of their original capacity after "
        "200,000 miles [claim:4].", SPAN)
    assert (r.verdict, r.layer) == (SUPPORTED, "deterministic")
    assert llm.prompts == []


def test_number_mismatch_is_unsupported_without_an_llm():
    llm = ScriptedLLM({"verdict": "SUPPORTED"})
    r = LayeredVerifier(llm).verify("Packs retain about 80% of their capacity [claim:4].", SPAN)
    assert (r.verdict, r.layer) == (UNSUPPORTED, "deterministic")
    assert [i["kind"] for i in r.issues] == ["number"]
    assert llm.prompts == []


@pytest.mark.parametrize("sentence", [
    "Packs do not retain 90% of their capacity.",           # added negation
    "Packs retain 90% of their capacity after 200,000 km.",  # unit swap
])
def test_negation_and_unit_mismatches_are_unsupported(sentence):
    r = LayeredVerifier(None).verify(sentence, SPAN)
    assert (r.verdict, r.layer) == (UNSUPPORTED, "deterministic")


def test_entity_flags_are_hints_for_the_llm_not_rejections():
    # A capitalized sentence opener absent from the span must not reject.
    llm = ScriptedLLM({"verdict": "SUPPORTED", "reason": "restates the span"})
    r = LayeredVerifier(llm).verify(
        "However, packs retain about 90% of their capacity.", SPAN)
    assert (r.verdict, r.layer) == (SUPPORTED, "llm")
    assert "However" in llm.prompts[0][1]


def test_context_is_shown_to_the_llm():
    llm = ScriptedLLM({"verdict": "SUPPORTED"})
    LayeredVerifier(llm).verify("Packs age slowly.", SPAN, context="CONTEXT-MARKER " + SPAN)
    assert "CONTEXT-MARKER" in llm.prompts[0][1]


# ---------- entailment layer ----------

def test_confident_entailment_decides():
    llm = ScriptedLLM({"verdict": "UNSUPPORTED"})
    r = LayeredVerifier(llm, entailer=FixedEntailer(entailment=0.97)).verify(
        "Packs keep most of their capacity.", SPAN)
    assert (r.verdict, r.layer) == (SUPPORTED, "entailment")
    assert llm.prompts == []


def test_confident_contradiction_decides():
    r = LayeredVerifier(None, entailer=FixedEntailer(contradiction=0.95)).verify(
        "Packs lose most of their capacity.", SPAN)
    assert (r.verdict, r.layer) == (UNSUPPORTED, "entailment")


def test_unsure_entailment_passes_to_the_llm():
    llm = ScriptedLLM({"verdict": "UNSUPPORTED", "reason": "adds a claim"})
    ent = FixedEntailer(entailment=0.6)
    r = LayeredVerifier(llm, entailer=ent).verify("Packs keep most capacity.", SPAN)
    assert (r.verdict, r.layer) == (UNSUPPORTED, "llm")
    assert ent.calls == 1


def test_deterministic_decision_skips_the_entailer():
    ent = FixedEntailer(entailment=0.99)
    r = LayeredVerifier(None, entailer=ent).verify("Packs retain 80% capacity.", SPAN)
    assert r.verdict == UNSUPPORTED and ent.calls == 0


# ---------- LLM layer ----------

@pytest.mark.parametrize("raw,expected", [
    ({"verdict": "SUPPORTED"}, SUPPORTED),
    ({"verdict": "unsupported"}, UNSUPPORTED),
    ({"verdict": "UNCERTAIN"}, UNCERTAIN),
    ({"verdict": "GROUNDED"}, UNCERTAIN),   # off-vocabulary is not a decision
    (["not", "a", "dict"], UNCERTAIN),
])
def test_llm_verdict_parsing(raw, expected):
    r = LayeredVerifier(ScriptedLLM(raw)).verify("Packs age slowly.", SPAN)
    assert r.verdict == expected


def test_llm_failure_is_uncertain_not_unsupported():
    r = LayeredVerifier(ScriptedLLM(exc=RuntimeError("boom"))).verify("Packs age slowly.", SPAN)
    assert (r.verdict, r.layer) == (UNCERTAIN, "llm")
    assert "boom" in r.reason


def test_no_llm_and_no_decision_is_uncertain():
    r = LayeredVerifier(None).verify("Packs age slowly.", SPAN)
    assert (r.verdict, r.layer) == (UNCERTAIN, "none")


# ---------- baseline (the current ask verifier, on the 3-way scale) ----------

@pytest.mark.parametrize("raw,expected", [
    ({"verdict": "GROUNDED"}, SUPPORTED),
    ({"verdict": "PARTIAL"}, UNSUPPORTED),
    ({"verdict": "UNGROUNDED"}, UNSUPPORTED),
    ({"verdict": "banana"}, UNSUPPORTED),  # ask's parser treats unknown as UNGROUNDED
])
def test_baseline_maps_ask_verdicts(raw, expected):
    assert baseline_verdict(ScriptedLLM(raw), "Packs age slowly.", SPAN).verdict == expected


def test_baseline_error_is_uncertain():
    assert baseline_verdict(ScriptedLLM(exc=RuntimeError("x")), "s", SPAN).verdict == UNCERTAIN


# ---------- ask integration (opt-in) ----------

class AskLLM:
    """Synthesizes a fixed answer; any verifier call through it fails loudly."""

    def __init__(self, answer):
        self.answer = answer
        self.json_calls = 0

    def complete(self, system, user, max_tokens=2048):
        return self.answer

    def complete_json(self, system, user, max_tokens=4096):
        self.json_calls += 1
        return {"verdict": "GROUNDED", "reason": "baseline says fine"}


def _ask_store(tmp_path):
    from aleph.db import Store
    store = Store(tmp_path / "q.db")
    sid = store.add_source("t.txt", SPAN)
    cid = store.add_claim(sid, "model s pack", "retains", "about 90% capacity",
                          0, len(SPAN), 0.9)
    return store, cid


def test_ask_with_layered_verifier_flags_what_the_baseline_accepts(tmp_path):
    from aleph.query import query
    store, cid = _ask_store(tmp_path)
    llm = AskLLM(f"Model S packs retain about 80% of their capacity [claim:{cid}].")
    try:
        base = query(store, llm, "model s pack capacity", use_cache=False)
        assert base.citations[0].verdict == "GROUNDED"

        layered = query(store, llm, "model s pack capacity", use_cache=False,
                        verifier=LayeredVerifier(llm))
        c = layered.citations[0]
        assert c.verdict == "UNGROUNDED"
        assert c.per_claim[cid][1].startswith("[deterministic]")
        assert "Verifier flags" in layered.answer
    finally:
        store.close()


def test_ask_layered_uncertain_is_its_own_flag(tmp_path):
    from aleph.query import query
    store, cid = _ask_store(tmp_path)
    llm = AskLLM(f"Packs age slowly [claim:{cid}].")
    try:
        r = query(store, llm, "model s pack capacity", use_cache=False,
                  verifier=LayeredVerifier(None))
        assert r.citations[0].verdict == "UNCERTAIN"
        assert "[UNCERTAIN]" in r.answer
    finally:
        store.close()


def test_truncated_span_is_not_deterministically_supported():
    # A verbatim fragment can drop the qualifier that scopes it.
    span = "Packs retain 90% of capacity only when kept below 30 degrees."
    r = LayeredVerifier(None).verify("Packs retain 90% of capacity.", span)
    assert r.verdict == UNCERTAIN


def test_one_sentence_of_a_multi_sentence_span_is_not_deterministic():
    # "Suppose taxes rise. Revenue falls." shows why: the other sentence can
    # scope this one.
    span = "Packs retain 90% of capacity. They are warrantied for 8 years."
    r = LayeredVerifier(None).verify("They are warrantied for 8 years.", span)
    assert r.verdict == UNCERTAIN


def test_dropped_negation_is_a_hint_not_a_rejection():
    # Restating the positive half of a negated span is faithful.
    span = "Page size is fixed at 50 items and cannot be configured."
    llm = ScriptedLLM({"verdict": "SUPPORTED"})
    r = LayeredVerifier(llm).verify("In v1 the page size is fixed at 50 items.", span)
    assert (r.verdict, r.layer) == (SUPPORTED, "llm")


def test_added_negation_still_rejects():
    r = LayeredVerifier(None).verify("Page size is not fixed.", "Page size is fixed at 50 items.")
    assert (r.verdict, r.layer) == (UNSUPPORTED, "deterministic")


# ---------- review findings ----------

@pytest.mark.parametrize("span,sentence", [
    ("It is false that approx. 90% of cells survive.", "90% of cells survive [claim:1]"),
    ("It is not true that the U.S. economy grew 3% in 2020.", "Economy grew 3% in 2020"),
    ("Suppose taxes rise. Revenue falls by 10%.", "Revenue falls by 10%."),
])
def test_a_fragment_of_the_span_is_never_deterministically_supported(span, sentence):
    assert LayeredVerifier(None).verify(sentence, span).verdict != SUPPORTED


def test_multi_citation_sentence_checks_numbers_against_all_cited_spans(tmp_path):
    from aleph.db import Store
    from aleph.query import query
    # Separate paragraphs, so neither claim's context window holds the other.
    text = "Alpha weighs 5 kg.\n\nBeta weighs 7 kg."
    store = Store(tmp_path / "m.db")
    sid = store.add_source("t.txt", text)
    a = store.add_claim(sid, "alpha", "weighs", "5 kg", 0, 18, 0.9)
    b = store.add_claim(sid, "beta", "weighs", "7 kg", 20, len(text), 0.9)
    llm = AskLLM(f"Alpha weighs 5 kg and Beta weighs 7 kg [claim:{a}][claim:{b}].")
    try:
        r = query(store, llm, "alpha beta weight", use_cache=False,
                  verifier=LayeredVerifier(None))
        assert r.citations[0].verdict != "UNGROUNDED"
    finally:
        store.close()


def test_layered_and_default_ask_do_not_share_cached_views(tmp_path):
    from aleph.query import query
    store, cid = _ask_store(tmp_path)
    llm = AskLLM(f"Model S packs retain about 80% of their capacity [claim:{cid}].")
    try:
        first = query(store, llm, "model s pack capacity")
        assert first.citations[0].verdict == "GROUNDED"
        layered = query(store, llm, "model s pack capacity", verifier=LayeredVerifier(llm))
        assert not layered.from_cache
        assert layered.citations[0].verdict == "UNGROUNDED"
        again = query(store, llm, "model s pack capacity")
        assert again.from_cache
        # the layered run must not have overwritten the default view
        assert "Verifier flags" not in again.answer
    finally:
        store.close()


def test_cross_encoder_reads_label_order_from_the_model():
    from aleph.verifier import CrossEncoderEntailer
    e = CrossEncoderEntailer.__new__(CrossEncoderEntailer)
    e.model = type("M", (), {
        "config": type("C", (), {"id2label": {0: "ENTAILMENT", 1: "neutral", 2: "contradiction"}})(),
        "predict": lambda self, pairs: [[5.0, 0.0, 0.0]],
    })()
    scores = e.predict("p", "h")
    assert scores["entailment"] > 0.9 and scores["contradiction"] < 0.1
