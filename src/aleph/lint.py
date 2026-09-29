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
from .authority import source_date
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


def _comparable_jurisdictions(ja, jb) -> bool:
    """Authority levels only compare within one legal order: the same
    jurisdiction, or one nested in the other (US-CA inside US)."""
    if not ja or not jb:
        return True
    return ja == jb or ja.startswith(jb + "-") or jb.startswith(ja + "-")


def _ordered(store: Store, a, b):
    """(keep, drop, rule) for a pair, or (None, None, reason) if it can't be
    decided soundly.

    - Sources of different domains: left open (``mixed_domain``). A paper
      and a statute are not two versions of one fact.
    - Two legal sources: lex superior (higher authority_level), then lex
      posterior (newer source date). Each step needs its field on both
      sides; a missing one leaves the pair open (``unranked`` / ``undated``)
      rather than counting as lowest. Levels from unrelated jurisdictions
      aren't compared (``cross_jurisdiction``). A pair split only by
      specificity is left open (``lex_specialis``): a special rule displaces
      the general one only within its scope, which is `distinguish`.
    - Otherwise the newer source date wins. Extraction time never decides.
    """
    ma = store.get_source_metadata(a["source_id"])
    mb = store.get_source_metadata(b["source_id"])
    if ma and mb and ma["domain"] != mb["domain"]:
        return None, None, "mixed_domain"
    if ma and mb and ma["domain"] == "legal":
        ia, ib = ma["metadata"], mb["metadata"]
        if not _comparable_jurisdictions(ia.get("jurisdiction"), ib.get("jurisdiction")):
            return None, None, "cross_jurisdiction"
        la, lb = ia.get("authority_level"), ib.get("authority_level")
        if la is None or lb is None:
            return None, None, "unranked"
        if la != lb:
            return (a, b, "legal_authority") if la > lb else (b, a, "legal_authority")
        if (ia.get("specificity") or 0) != (ib.get("specificity") or 0):
            return None, None, "lex_specialis"
        da, db = source_date(ma), source_date(mb)
        if da is None or db is None:
            return None, None, "undated"
        if da == db:
            return None, None, "tie"
        return (a, b, "legal_authority") if da > db else (b, a, "legal_authority")
    da, db = source_date(ma), source_date(mb)
    if da is None or db is None:
        return None, None, "undated"
    if da == db:
        return None, None, "tie"
    return (a, b, "source_date") if da > db else (b, a, "source_date")


def resolve_by_source_date(store: Store) -> dict:
    """Resolve each open contradiction as `supersede`, keeping the claim
    whose source is newer (or, between two legal sources, of higher
    authority). Every pair :func:`_ordered` can't decide, cross-subject
    pairs, and pairs ``dispose`` refuses stay open and are reported in
    ``skipped`` with their reason."""
    decisions, skipped = [], []
    for row in store.list_contradictions(only_open=True):
        a = store.get_claim(row["claim_a_id"])
        b = store.get_claim(row["claim_b_id"])
        if not a or not b:
            continue
        # A cross-subject pair (rule-limits-rule, doctrinal-cross-ref, ...)
        # is a relation between two live rules, never a newer version.
        if row["cross_subject"]:
            skipped.append({"contradiction_id": row["id"], "reason": "cross_subject"})
            continue
        keep, drop, rule = _ordered(store, a, b)
        if keep is None:
            skipped.append({"contradiction_id": row["id"], "reason": rule})
            continue
        try:
            dispose(store, row["id"], "supersede", keep=keep["id"], drop=drop["id"])
        except ValueError as e:
            # dispose refuses pairs it can't resolve soundly (e.g. a member
            # is no longer active). Leave those open rather than force them.
            log("resolve_by_recency_skipped", level="info",
                contradiction_id=row["id"], reason=str(e))
            skipped.append({"contradiction_id": row["id"], "reason": "refused",
                            "detail": str(e)})
            continue
        decisions.append({"contradiction_id": row["id"], "keep": keep["id"],
                          "drop": drop["id"], "rule": rule})
    return {"resolved": len(decisions), "decisions": decisions, "skipped": skipped}


def resolve_by_recency(store: Store) -> int:
    """Back-compat wrapper: the number of contradictions
    :func:`resolve_by_source_date` resolved."""
    return resolve_by_source_date(store)["resolved"]
