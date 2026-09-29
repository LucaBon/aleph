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


def _jurisdiction_relation(ja, jb) -> str:
    """same | nested (US-CA inside US) | unknown (only one side says) | unrelated.
    Two sources that both leave jurisdiction unset count as the same order."""
    if not ja and not jb:
        return "same"
    if not ja or not jb:
        return "unknown"
    if ja == jb:
        return "same"
    if ja.startswith(jb + "-") or jb.startswith(ja + "-"):
        return "nested"
    return "unrelated"


def _ordered(store: Store, a, b):
    """(keep, drop, rule) for a pair, or (None, None, reason) if it can't be
    decided soundly.

    - Sources of different domains: left open (``mixed_domain``). A paper
      and a statute are not two versions of one fact.
    - Two legal sources, compared only within one legal order: an unset
      jurisdiction on one side (``unknown_jurisdiction``) or unrelated ones
      (``cross_jurisdiction``) leave the pair open. Then lex superior
      (higher authority_level; also across nested orders, e.g. US over
      US-CA), then lex posterior (newer source date, same jurisdiction
      only: across nested orders it is preemption, ``nested_jurisdiction``).
      Each step needs its field on both sides; a missing one leaves the pair
      open (``unranked`` / ``undated``). A specificity split is left open
      (``lex_specialis``): a special rule displaces the general one only
      within its scope, which is `distinguish`.
    - Otherwise the newer source date wins. Extraction time never decides.
    """
    ma = store.get_source_metadata(a["source_id"])
    mb = store.get_source_metadata(b["source_id"])
    if ma and mb and ma["domain"] != mb["domain"]:
        return None, None, "mixed_domain"
    if ma and mb and ma["domain"] == "legal":
        ia, ib = ma["metadata"], mb["metadata"]
        rel = _jurisdiction_relation(ia.get("jurisdiction"), ib.get("jurisdiction"))
        if rel == "unknown":
            return None, None, "unknown_jurisdiction"
        if rel == "unrelated":
            return None, None, "cross_jurisdiction"
        la, lb = ia.get("authority_level"), ib.get("authority_level")
        if la is None or lb is None:
            return None, None, "unranked"
        if la != lb:
            return (a, b, "lex_superior") if la > lb else (b, a, "lex_superior")
        if rel == "nested":
            return None, None, "nested_jurisdiction"
        sa, sb = ia.get("specificity"), ib.get("specificity")
        if (sa is None) != (sb is None):
            return None, None, "unranked"
        if sa != sb:
            return None, None, "lex_specialis"
        da, db = source_date(ma), source_date(mb)
        if da is None or db is None:
            return None, None, "undated"
        if da == db:
            return None, None, "tie"
        return (a, b, "lex_posterior") if da > db else (b, a, "lex_posterior")
    da, db = source_date(ma), source_date(mb)
    if da is None or db is None:
        return None, None, "undated"
    if da == db:
        return None, None, "tie"
    return (a, b, "source_date") if da > db else (b, a, "source_date")


def _pair_age(store: Store, row) -> tuple:
    """Sort key: pairs whose newer source is oldest go first. In a chain
    A > B > C with pairs (A,B) and (B,C), resolving (B,C) first lets C be
    superseded by B before B is superseded by A; the other order leaves C
    active because (B,C) is refused once B is inactive."""
    dates = [source_date(store.get_source_metadata(c["source_id"]))
             for c in (store.get_claim(row["claim_a_id"]), store.get_claim(row["claim_b_id"])) if c]
    known = [d for d in dates if d is not None]
    return (max(known) if known else float("inf"), row["id"])


def resolve_by_source_date(store: Store) -> dict:
    """Resolve each open contradiction as `supersede`, keeping the claim
    whose source is newer (or, between two legal sources, of higher
    authority). Every pair :func:`_ordered` can't decide, cross-subject
    pairs, and pairs ``dispose`` refuses stay open and are reported in
    ``skipped`` with their reason."""
    decisions, skipped = [], []
    rows = sorted(store.list_contradictions(only_open=True), key=lambda r: _pair_age(store, r))
    for row in rows:
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
