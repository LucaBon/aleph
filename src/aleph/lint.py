"""Lint: detect contradictions and health issues across the claim graph.

Contradiction candidates = same (subject, predicate) with different objects.
We ask the LLM to judge whether they actually conflict (vs. just being different
facets of the same subject), and record confirmed ones.
"""
from __future__ import annotations

from collections import defaultdict

from .db import Store
from .llm import LLM

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

    We group candidates by subject only (not subject+predicate) because predicates
    are free-form LLM text — "last" and "last on average" refer to the same thing.
    The LLM judge filters out false candidates; grouping too narrowly misses real
    contradictions silently, which is the worse failure.
    """
    claims = store.all_active_claims()
    # group by subject
    groups: dict[str, list] = defaultdict(list)
    for c in claims:
        groups[c["subject"]].append(c)

    # only inspect groups with multiple claims
    candidates = [(k, v) for k, v in groups.items() if len(v) >= 2]
    if verbose:
        print(f"scanning {len(candidates)} candidate groups...")

    confirmed = 0
    checked = 0
    for subj, members in candidates:
        # pairwise among distinct (predicate, object) combinations
        seen: set[tuple[str, str]] = set()
        unique = []
        for m in members:
            key = (m["predicate"].lower().strip(), m["object"].lower().strip())
            if key in seen:
                continue
            seen.add(key)
            unique.append(m)
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                a, b = unique[i], unique[j]
                # skip pairs with identical object — no contradiction possible
                if a["object"].lower().strip() == b["object"].lower().strip():
                    continue
                checked += 1
                a_span = store.get_span_text(a["id"]) or ""
                b_span = store.get_span_text(b["id"]) or ""
                try:
                    result = llm.complete_json(
                        CONTRADICTION_SYSTEM,
                        CONTRADICTION_USER.format(
                            a_subj=a["subject"], a_pred=a["predicate"], a_obj=a["object"],
                            a_span=a_span[:400],
                            b_subj=b["subject"], b_pred=b["predicate"], b_obj=b["object"],
                            b_span=b_span[:400],
                        ),
                        max_tokens=256,
                    )
                except Exception:
                    continue
                if bool(result.get("contradicts")):
                    store.add_contradiction(a["id"], b["id"])
                    confirmed += 1
                    if verbose:
                        print(f"  contradiction: [{a['id']}] vs [{b['id']}] — {subj}")
    return {
        "candidate_groups": len(candidates),
        "pairs_checked": checked,
        "contradictions_confirmed": confirmed,
    }


def resolve_by_recency(store: Store) -> int:
    """For each open contradiction, mark the older claim as superseded by the newer.

    Simple default policy. Users can implement other policies (authority, confidence).
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
        store.supersede_claim(older["id"], newer["id"])
        # mark contradiction resolved
        with store.tx() as cx:
            cx.execute(
                "UPDATE contradictions SET status = 'resolved', resolved_to = ? WHERE id = ?",
                (newer["id"], row["id"]),
            )
        resolved += 1
    # supersede_claim already invalidates cache entries citing the superseded
    # claim — no blanket clear_cache needed here.
    return resolved
