"""Contradiction detection and disposition.

Detection: typed pre-filters + LLM confirmer. Pre-filters classify the
kind of conflict (numeric, categorical, negation, temporal, normative);
the LLM confirms and proposes a candidate disposition.

Dispositions: supersede | coexist | distinguish | reconcile | replicate
| dispute | retracted | gap | unresolved. See docs/plan/00-architecture.md
for the full vocabulary and when each applies.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .db import Store
from .llm import LLM
from .log import log


# ----- prompts (verbatim from docs/plan/90-prompts.md) -----

DETECT_SYSTEM = """\
You judge whether two claims with the same subject contradict each other,
and if so, what kind of disposition fits best.

A pair "contradicts" when both claims cannot simultaneously be true of the
same subject under the same conditions. Different facets of the same
subject are NOT contradictions ("Tesla makes cars" vs "Tesla makes
batteries" — both true). Different values for the same quantity or
mutually exclusive categorical claims ARE contradictions.

If the pair contradicts, classify the disposition you would recommend:
- supersede: one claim is an updated/correction of the other (same scope,
  different value, clear recency or authority signal).
- coexist: both hold simultaneously under different scopes (jurisdiction,
  date, level of authority). Only applicable if the scope difference is
  stated or strongly implied.
- distinguish: the apparent conflict is a category error — the claims
  address different things that happen to share a label.
- reconcile: both are valid findings under different premises/conditions.
  Only applicable when the claims have distinguishable conditions.
- replicate: this is NOT a contradiction — the claims assert the same fact
  with the same scope from different sources.
- dispute: same scope, genuine disagreement, no obvious resolution.
- gap: same scope on the surface, but the divergence suggests an unstated
  premise the text does not reveal.

Do NOT try to detect hidden premises. If you cannot locate a stated
scope/condition difference, prefer "dispute" or "gap" over "reconcile".

Reply in JSON:
{
  "contradicts": true | false,
  "kind": "numeric" | "categorical" | "negation" | "temporal" | "normative" | "unknown",
  "candidate_disposition": "supersede" | "coexist" | "distinguish" | "reconcile" | "replicate" | "dispute" | "gap" | "unresolved",
  "reason": "one short sentence"
}

If contradicts=false, set candidate_disposition="unresolved" and kind="unknown"."""

DETECT_USER_TEMPLATE = """\
Claim A: [claim:{a_id}] {a_subj} — {a_pred} — {a_obj}
  source span: {a_span!r}
  source metadata: {a_meta}

Claim B: [claim:{b_id}] {b_subj} — {b_pred} — {b_obj}
  source span: {b_span!r}
  source metadata: {b_meta}

Shared conditions (claim IDs both claims depend on): {shared_conditions}
Distinct conditions A: {a_only_conditions}
Distinct conditions B: {b_only_conditions}

Classify (JSON)."""


# ----- helpers -----

_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def _try_number(s: str) -> Optional[float]:
    """Extract the first number from an object string. Returns None if no
    number or if the string has mixed categorical+numeric content we can't
    safely compare."""
    m = _NUMERIC_RE.search(s)
    if not m:
        return None
    return float(m.group())


_NEGATION_PAIRS = [
    ("is", "is not"), ("has", "has no"), ("can", "cannot"),
    ("requires", "prohibits"), ("permits", "forbids"), ("allows", "disallows"),
    ("causes", "prevents"),
]

_TEMPORAL_KEYWORDS = {
    "year", "years", "month", "months", "day", "days", "decade", "decades",
    "century", "centuries", "before", "after", "since", "until", "date",
    "period", "era", "quarter", "annual", "weekly", "daily",
}

_DEONTIC_VERBS = {
    "requires", "require", "prohibits", "prohibit",
    "permits", "permit", "forbids", "forbid",
    "mandates", "mandate", "shall", "must",
}


@dataclass
class ConflictCandidate:
    claim_a_id: int
    claim_b_id: int
    kind: str          # numeric | categorical | negation | temporal | normative
    overlap: Optional[float]  # conditions overlap; null if not computable


def classify_pair(store: Store, a, b) -> Optional[ConflictCandidate]:
    """Cheap typed classifier. Returns None if the pair is clearly not a
    conflict (e.g. identical objects). Runs before any LLM call.

    WS-E.1: predicates are resolved through the domain-scoped predicate
    alias table before any negation/deontic pre-filter check, so two
    surface forms ("permits"/"authorizes") that share a canonical
    predicate behave as one.

    WS-E.2: if both claims carry sense assignments and the senses differ,
    skip the pair entirely — different relations are not the same kind of
    contradiction even when the surface predicate matches.
    """
    a_obj = a["object"].strip().lower()
    b_obj = b["object"].strip().lower()

    # identical objects => no contradiction
    if a_obj == b_obj:
        return None

    # WS-E.2: sense disagreement suppresses the pre-filter
    a_sense_row = store.conn.execute(
        "SELECT sense_id FROM claim_predicate_senses WHERE claim_id = ?",
        (a["id"],),
    ).fetchone()
    b_sense_row = store.conn.execute(
        "SELECT sense_id FROM claim_predicate_senses WHERE claim_id = ?",
        (b["id"],),
    ).fetchone()
    if (
        a_sense_row is not None and b_sense_row is not None
        and a_sense_row["sense_id"] != b_sense_row["sense_id"]
    ):
        return None

    # WS-E.1: resolve predicates through domain-scoped aliases first.
    a_meta = store.get_source_metadata(a["source_id"]) or {}
    b_meta = store.get_source_metadata(b["source_id"]) or {}
    a_pred = store.resolve_predicate(
        a["predicate"], domain=a_meta.get("domain"),
    ).strip().lower()
    b_pred = store.resolve_predicate(
        b["predicate"], domain=b_meta.get("domain"),
    ).strip().lower()

    # compute overlap
    try:
        overlap = store.conditions_overlap(a["id"], b["id"])
    except Exception:
        overlap = None

    # negation check
    for pos, neg in _NEGATION_PAIRS:
        if (a_pred == pos and b_pred == neg) or (a_pred == neg and b_pred == pos):
            return ConflictCandidate(a["id"], b["id"], "negation", overlap)

    # normative check: deontic predicates
    if a_pred in _DEONTIC_VERBS or b_pred in _DEONTIC_VERBS:
        return ConflictCandidate(a["id"], b["id"], "normative", overlap)

    # numeric check: both objects contain numbers
    a_num = _try_number(a_obj)
    b_num = _try_number(b_obj)
    if a_num is not None and b_num is not None and a_num != b_num:
        return ConflictCandidate(a["id"], b["id"], "numeric", overlap)

    # temporal check: objects reference time-related words
    a_words = set(a_obj.split())
    b_words = set(b_obj.split())
    if a_words & _TEMPORAL_KEYWORDS or b_words & _TEMPORAL_KEYWORDS:
        return ConflictCandidate(a["id"], b["id"], "temporal", overlap)

    # default: categorical
    return ConflictCandidate(a["id"], b["id"], "categorical", overlap)


def _format_conditions(store: Store, claim_id: int) -> list[int]:
    """Return list of condition_claim_ids for a claim."""
    rows = store.conn.execute(
        "SELECT condition_claim_id FROM claim_conditions WHERE claim_id = ?",
        (claim_id,),
    ).fetchall()
    return [r["condition_claim_id"] for r in rows]


def _get_source_meta(store: Store, claim) -> str:
    """Return compact source metadata string or 'none'."""
    meta = store.get_source_metadata(claim["source_id"])
    if meta:
        return json.dumps(meta, default=str)
    return "none"


def _confirm_with_llm(
    store: Store, llm: LLM, a, b, candidate: ConflictCandidate,
) -> Optional[dict]:
    """Ask the LLM to confirm a contradiction and propose a disposition.
    Returns the parsed JSON result dict or None on failure."""
    a_span = store.get_span_text(a["id"]) or ""
    b_span = store.get_span_text(b["id"]) or ""

    a_conds = set(_format_conditions(store, a["id"]))
    b_conds = set(_format_conditions(store, b["id"]))
    shared = sorted(a_conds & b_conds)
    a_only = sorted(a_conds - b_conds)
    b_only = sorted(b_conds - a_conds)

    user_msg = DETECT_USER_TEMPLATE.format(
        a_id=a["id"], a_subj=a["subject"], a_pred=a["predicate"], a_obj=a["object"],
        a_span=a_span[:400],
        a_meta=_get_source_meta(store, a),
        b_id=b["id"], b_subj=b["subject"], b_pred=b["predicate"], b_obj=b["object"],
        b_span=b_span[:400],
        b_meta=_get_source_meta(store, b),
        shared_conditions=", ".join(f"[claim:{c}]" for c in shared) if shared else "none",
        a_only_conditions=", ".join(f"[claim:{c}]" for c in a_only) if a_only else "none",
        b_only_conditions=", ".join(f"[claim:{c}]" for c in b_only) if b_only else "none",
    )

    try:
        result = llm.complete_json(DETECT_SYSTEM, user_msg, max_tokens=256)
    except Exception:
        return None

    if not isinstance(result, dict):
        return None
    return result


def detect_for_claim(
    store: Store, llm: LLM, claim_id: int, *,
    run_llm: bool = True,
) -> list[int]:
    """Run incremental detection for a single new claim. Returns new
    contradiction IDs."""
    claim = store.get_claim(claim_id)
    if not claim or claim["status"] != "active":
        return []

    # find all other active claims on the same subject
    others = store.conn.execute(
        "SELECT * FROM claims WHERE subject = ? AND status = 'active' AND id != ?",
        (claim["subject"], claim_id),
    ).fetchall()

    new_ids: list[int] = []
    for other in others:
        candidate = classify_pair(store, claim, other)
        if candidate is None:
            continue

        if run_llm:
            result = _confirm_with_llm(store, llm, claim, other, candidate)
            if not result or not result.get("contradicts"):
                continue
            kind = result.get("kind", candidate.kind)
            cand_disp = result.get("candidate_disposition", "unresolved")
            # Invariant 3: if overlap is None/0, do not allow reconcile
            if candidate.overlap is None or candidate.overlap == 0.0:
                if cand_disp == "reconcile":
                    cand_disp = "dispute"
        else:
            kind = candidate.kind
            cand_disp = "unresolved"

        cid = store.add_contradiction_with_kind(
            claim["id"], other["id"], kind,
            candidate_disposition=cand_disp,
            overlap_score=candidate.overlap,
        )
        if cid is not None:
            new_ids.append(cid)

    return new_ids


def detect_all(
    store: Store, llm: LLM, *,
    subject: Optional[str] = None,
    kind: Optional[str] = None,
    since: Optional[float] = None,
    verbose: bool = False,
) -> dict:
    """Batch scan. Groups active claims by subject and runs pairwise
    classify_pair -> LLM confirmer. Returns a summary dict."""
    claims = store.all_active_claims()

    # optional: filter by subject
    if subject:
        canonical = store.resolve_subject(subject)
        claims = [c for c in claims if c["subject"] == canonical]

    # optional: filter by since (extracted_at)
    if since is not None:
        claims = [c for c in claims if c["extracted_at"] >= since]

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
    by_kind: dict[str, int] = defaultdict(int)
    by_candidate_disposition: dict[str, int] = defaultdict(int)
    new_contradictions: list[dict] = []

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
                candidate = classify_pair(store, a, b)
                if candidate is None:
                    continue

                # optional: filter by kind
                if kind and candidate.kind != kind:
                    continue

                checked += 1
                result = _confirm_with_llm(store, llm, a, b, candidate)
                if not result:
                    continue

                if bool(result.get("contradicts")):
                    det_kind = result.get("kind", candidate.kind)
                    cand_disp = result.get("candidate_disposition", "unresolved")

                    # Invariant 3: if overlap is None/0, do not allow reconcile
                    if candidate.overlap is None or candidate.overlap == 0.0:
                        if cand_disp == "reconcile":
                            cand_disp = "dispute"

                    cid = store.add_contradiction_with_kind(
                        a["id"], b["id"], det_kind,
                        candidate_disposition=cand_disp,
                        overlap_score=candidate.overlap,
                    )
                    if cid is not None:
                        confirmed += 1
                        by_kind[det_kind] += 1
                        by_candidate_disposition[cand_disp] += 1
                        new_contradictions.append({
                            "id": cid,
                            "claim_a": a["id"],
                            "claim_b": b["id"],
                            "kind": det_kind,
                            "candidate_disposition": cand_disp,
                        })
                        if verbose:
                            print(f"  contradiction: [{a['id']}] vs [{b['id']}] "
                                  f"— {subj} ({det_kind}, {cand_disp})")
                    else:
                        # already existed
                        confirmed += 1
                        by_kind[det_kind] += 1
                        by_candidate_disposition[cand_disp] += 1

    return {
        "candidate_groups": len(candidates),
        "pairs_checked": checked,
        "contradictions_confirmed": confirmed,
        "by_kind": dict(by_kind),
        "by_candidate_disposition": dict(by_candidate_disposition),
        "new_contradictions": new_contradictions,
    }


# ----- allowed dispositions -----

_ALLOWED_DISPOSITIONS = {
    "supersede", "coexist", "distinguish", "reconcile", "replicate",
    "dispute", "retracted", "gap", "unresolved",
}

_RULE_REQUIRED = {"coexist", "distinguish", "reconcile"}
_KEEP_DROP_ALLOWED = {"supersede", "retracted"}


def dispose(
    store: Store, contradiction_id: int, disposition: str, *,
    rule: Optional[str] = None,
    applies_when: Optional[dict] = None,
    rationale_concept_id: Optional[int] = None,
    keep: Optional[int] = None,
    drop: Optional[int] = None,
) -> dict:
    """Apply a disposition to a contradiction."""
    # validate disposition
    if disposition not in _ALLOWED_DISPOSITIONS:
        raise ValueError(f"invalid_disposition:{disposition}")

    # validate keep/drop
    if (keep is not None or drop is not None) and disposition not in _KEEP_DROP_ALLOWED:
        raise ValueError("keep_drop_not_allowed_for_disposition")

    # validate rule requirement
    if disposition in _RULE_REQUIRED and not rule:
        raise ValueError("rule_required_for_disposition")

    # validate rationale concept. P1.2: `attested` is an accepted status for
    # agent-mode runs — the concept carries an attestation trail (attested_by,
    # attestation_rationale) that replaces the LLM GROUNDED verdict as the
    # honesty signal. `active` remains the preferred/strongest status.
    if rationale_concept_id is not None:
        concept = store.get_concept(rationale_concept_id)
        if not concept:
            raise ValueError("rationale_concept_not_found")
        if concept["status"] not in ("active", "attested"):
            raise ValueError("rationale_concept_not_active")

    # fetch the contradiction
    crow = store.conn.execute(
        "SELECT * FROM contradictions WHERE id = ?", (contradiction_id,)
    ).fetchone()
    if not crow:
        raise ValueError("contradiction_not_found")

    # A resolution must rest on claims that are live. The only exception is
    # the dropped side of a `retracted` disposition, which may already be out
    # of the active set (e.g. its source was retracted first).
    pair = (crow["claim_a_id"], crow["claim_b_id"])
    if keep is not None and keep not in pair:
        raise ValueError("keep_not_in_contradiction")
    if drop is not None and drop not in pair:
        raise ValueError("drop_not_in_contradiction")
    if disposition == "retracted" and drop is not None and keep is None:
        keep = pair[0] if drop == pair[1] else pair[1]
    exempt = {drop} if disposition == "retracted" else set()
    for cid in pair:
        if cid in exempt:
            continue
        c = store.get_claim(cid)
        if c is None or c["status"] != "active":
            raise ValueError(
                f"contradiction_member_inactive:{cid}={c['status'] if c else 'missing'}"
            )

    # apply side effects based on disposition
    if disposition == "supersede":
        if keep is None or drop is None:
            raise ValueError("keep_drop_required_for_supersede")
        store.supersede_claim(drop, keep)
        store.update_contradiction_disposition(
            contradiction_id, disposition,
            rule=rule, applies_when=applies_when,
            rationale_concept_id=rationale_concept_id,
            keep=keep,
        )

    elif disposition == "retracted":
        if drop is None:
            raise ValueError("drop_required_for_retracted")
        # retracted: update claim status to 'retracted' directly (not superseded)
        with store.tx() as cx:
            cur = cx.execute(
                "UPDATE claims SET status = 'retracted' WHERE id = ? AND status = 'active'",
                (drop,),
            )
            if cur.rowcount:
                store._on_claims_deactivated_tx(cx, [drop], cause="retracted")
        store.update_contradiction_disposition(
            contradiction_id, disposition,
            rule=rule, applies_when=applies_when,
            rationale_concept_id=rationale_concept_id,
            keep=keep,
        )

    elif disposition == "replicate":
        # bump both claims' confidence by 0.05, capped at 1.0
        # Record the bump actually applied (it's capped at 1.0) so that a
        # reopen can revert it exactly.
        deltas = []
        with store.tx() as cx:
            for cid in pair:
                before = cx.execute(
                    "SELECT confidence FROM claims WHERE id = ?", (cid,)
                ).fetchone()["confidence"]
                after = min(before + 0.05, 1.0)
                cx.execute(
                    "UPDATE claims SET confidence = ? WHERE id = ?", (after, cid),
                )
                deltas.append(after - before)
        store.update_contradiction_disposition(
            contradiction_id, disposition,
            rule=rule, applies_when=applies_when,
            rationale_concept_id=rationale_concept_id,
            confidence_deltas=(deltas[0], deltas[1]),
        )

    else:
        # coexist, distinguish, reconcile, dispute, gap, unresolved
        store.update_contradiction_disposition(
            contradiction_id, disposition,
            rule=rule, applies_when=applies_when,
            rationale_concept_id=rationale_concept_id,
        )

    return {
        "contradiction_id": contradiction_id,
        "disposition": disposition,
        "applied": True,
    }
