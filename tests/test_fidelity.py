"""Deterministic fidelity checker (Phase 2).

A claim's numbers, dates, units, negations and entities must appear in the
span it cites or in that span's context window. The seeded fixture pairs
each deliberate mismatch with faithful controls; the exit criterion is that
every seeded case is flagged with the right kind and no control is flagged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aleph.fidelity import check_fidelity, context_window

FIXTURE = Path(__file__).parent / "fixtures" / "fidelity_seeded.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
SEEDED = [c for c in CASES if c["seeded"]]
CONTROLS = [c for c in CASES if not c["seeded"]]


def _kinds(case) -> set[str]:
    issues = check_fidelity(case["claim"], case["span"], case.get("context", ""))
    return {i.kind for i in issues}


@pytest.mark.parametrize("case", SEEDED, ids=[c["id"] for c in SEEDED])
def test_seeded_mismatch_is_flagged(case):
    assert case["seeded"] in _kinds(case)


@pytest.mark.parametrize("case", CONTROLS, ids=[c["id"] for c in CONTROLS])
def test_faithful_claim_is_not_flagged(case):
    assert _kinds(case) == set()


def test_fixture_covers_every_kind():
    assert {c["seeded"] for c in SEEDED} == {"number", "date", "unit", "negation", "entity"}


def test_issue_names_the_missing_value():
    [issue] = check_fidelity("Retains 80% capacity.", "retains 90% capacity")
    assert issue.kind == "number"
    assert issue.value == "80"
    assert issue.to_dict() == {"kind": "number", "value": "80", "message": issue.message}


def test_context_rescues_an_element_outside_the_span():
    span = "It was funded by the NIH."
    assert check_fidelity("Stanford was funded by the NIH.", span)
    assert not check_fidelity(
        "Stanford was funded by the NIH.", span,
        "The study ran at Stanford. It was funded by the NIH.",
    )


def test_dropped_negation_checks_the_span_only():
    # A negation in the surrounding context does not make the claim's
    # omission of the span's own negation faithful.
    kinds = {i.kind for i in check_fidelity(
        "Accuracy improved.", "Accuracy did not improve.", "Nothing else changed.")}
    assert "negation" in kinds


def test_identifier_change_is_an_entity_issue():
    kinds = {i.kind for i in check_fidelity("The v3 API is stable.", "The v2 API is stable.")}
    assert kinds == {"entity"}


def test_lowercase_triple_rendering_does_not_raise_entity_flags():
    assert not check_fidelity("tesla battery retains 90% capacity",
                              "Tesla batteries retain 90% capacity.")


# ---------- context window ----------

TEXT = (
    "First sentence here. Second sentence has the span in it. "
    "Third sentence follows. Fourth is far away."
)


def test_context_window_extends_to_neighbouring_sentences():
    start = TEXT.index("the span")
    end = start + len("the span")
    c_start, c_end = context_window(TEXT, start, end)
    assert TEXT[c_start:c_end] == (
        "First sentence here. Second sentence has the span in it. "
        "Third sentence follows."
    )


def test_context_window_always_contains_the_span():
    for start in range(0, len(TEXT) - 5, 7):
        end = start + 5
        c_start, c_end = context_window(TEXT, start, end)
        assert c_start <= start and end <= c_end


def test_context_window_is_capped():
    text = "word " * 2000
    c_start, c_end = context_window(text, 5000, 5010, max_chars=400)
    assert c_end - c_start <= 400 + 10
    assert c_start <= 5000 and 5010 <= c_end


def test_context_window_stops_at_paragraph_breaks():
    text = "Other paragraph.\n\nThe span sentence. Next one."
    start = text.index("span")
    c_start, c_end = context_window(text, start, start + 4)
    assert text[c_start:c_end] == "The span sentence. Next one."


def test_negation_issues_record_their_direction():
    [added] = check_fidelity("Backups are not kept.", "Backups are kept.")
    [dropped] = check_fidelity("Backups are kept.", "Backups are not kept.")
    assert (added.direction, dropped.direction) == ("added", "dropped")
    assert added.to_dict()["direction"] == "added"


def test_added_negation_on_a_word_absent_from_the_span_falls_back_to_presence():
    # The negated word comes from the predicate ("non si applica"); the span
    # states the negation with different words. Some negation is present,
    # so there is nothing local to compare against: no flag.
    span = "La premeditazione non è configurabile quando il momento è occasionale."
    assert not check_fidelity("non si applica la premeditazione", span)
    # ...but with no negation anywhere it is still an added negation.
    assert [i.kind for i in check_fidelity(
        "non si applica la premeditazione", "La premeditazione è configurabile.")] == ["negation"]
