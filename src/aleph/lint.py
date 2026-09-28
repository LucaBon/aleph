"""Lint: detect contradictions and health issues across the claim graph.

Thin wrapper over contradictions.detect_all. Preserves the existing CLI
surface and tier1-3 return shape. The old CONTRADICTION_SYSTEM prompt is
kept here for backward compatibility (the FakeLLM dispatches on its
substring "You judge whether two claims"); detect_all uses the new
DETECT_SYSTEM prompt from contradictions.py which contains the same
dispatch substring.
"""
from __future__ import annotations

from .db import Store
from .log import log
from .llm import LLM
from .contradictions import detect_all, dispose


# Legacy prompt constants kept for import compatibility (nothing uses them
# at runtime any more, but external code may reference them).
CONTRADICTION_SYSTEM = """You judge whether two claims about the same subject
and predicate actually contradict each other.

Two claims may have the same subject+predicate but different objects without
contradicting — e.g. "Tesla — makes — cars" and "Tesla — makes — batteries"
are both true. But "Tesla batteries — last — 5 years" vs "Tesla batteries —
last — 10 years" is a real contradiction.

Reply in JSON: {"contradicts": true|false, "reason": "one short sentence"}"""


CONTRADICTION_USER = """Claim A: {a_subj} — {a_pred} — {a_obj}
(source span: {a_span})

Claim B: {b_subj} — {b_pred} — {b_obj}
(source span: {b_span})

Do these contradict?"""


def lint(store: Store, llm: LLM, verbose: bool = False) -> dict:
    """Scan for contradictions. Returns a summary dict.

    Delegates to contradictions.detect_all which returns a superset of the
    old keys (candidate_groups, pairs_checked, contradictions_confirmed).
    """
    return detect_all(store, llm, verbose=verbose)


def resolve_by_recency(store: Store) -> int:
    """For each open contradiction, mark the older claim as superseded by the newer.

    Delegates to contradictions.dispose per-contradiction with disposition='supersede'.
    """
    resolved = 0
    for row in store.list_contradictions(only_open=True):
        a = store.get_claim(row["claim_a_id"])
        b = store.get_claim(row["claim_b_id"])
        if not a or not b:
            continue
        if a["extracted_at"] >= b["extracted_at"]:
            newer, older = a, b
        else:
            newer, older = b, a
        try:
            dispose(store, row["id"], "supersede",
                    keep=newer["id"], drop=older["id"])
        except ValueError as e:
            # dispose refuses pairs it can't resolve soundly (e.g. a member
            # is no longer active). Leave those open rather than force them.
            log("resolve_by_recency_skipped", level="info",
                contradiction_id=row["id"], reason=str(e))
            continue
        resolved += 1
    return resolved
