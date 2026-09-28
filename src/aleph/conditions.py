"""Claim conditions: links from claims to other claims that describe when
they apply (sample, scope, method, limitation, assumption).

The `explicit` flag records whether the condition is stated verbatim in the
source or inferred. Never silently mark inferred conditions as explicit —
this is the honesty contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .db import Store
from .llm import LLM
from .log import log


# ----- prompts (see 90-prompts.md) -----
EXTRACT_SCOPE_SYSTEM = """You extract SCOPE claims from a source document.

A scope claim describes WHEN a finding applies: sample, method, scope,
limitation, or assumption. Scope claims are themselves atomic claims with
verbatim spans — not a separate entity type.

Emit a JSON array. Each item has:
{
  "subject": "string, what the scope is about (e.g. 'the study', 'the sample', 'the method')",
  "predicate": "string, verb phrase",
  "object": "string, the specific scope value",
  "span": "string, verbatim substring of the source",
  "kind": "sample" | "method" | "scope" | "limitation" | "assumption",
  "confidence": number between 0 and 1
}

Rules:
- Only emit scope that is EXPLICITLY stated. If a condition is only hinted
  at, skip it — do not guess.
- span MUST be a verbatim substring of the source.
- Prefer specific subjects ("the mouse cohort", "the randomized trial")
  over vague ones ("it", "this work").
- Skip boilerplate (funding, acknowledgments) unless it encodes a condition."""

EXTRACT_SCOPE_USER_TEMPLATE = """Source document:
---
{text}
---

Extract scope claims as described. JSON array only."""

EXTRACT_CONDITIONS_SYSTEM = """You propose which EXISTING scope claims apply to a given atomic claim.

You are given one claim and a list of scope/method/sample/limitation
claims from the same source. For each scope claim, decide whether it
applies to the given claim.

Rules:
- Only say a scope applies when the source text makes it clear (same
  section, explicit reference, or the scope is document-wide like "all
  experiments used BALB/c mice").
- If you are unsure, do NOT include the scope. Missing conditions are
  honest; wrong conditions are noise.
- Do NOT invent new scope claims. Use only IDs from the provided list.

Reply in JSON:
{
  "applies": [
    {"condition_claim_id": int, "kind": "sample" | "method" | "scope" | "limitation" | "assumption", "explicit": true | false, "reason": "one short sentence"},
    ...
  ]
}"""

EXTRACT_CONDITIONS_USER_TEMPLATE = """Target claim: [claim:{claim_id}] {subj} — {pred} — {obj}
  source span: {span!r}

Available scope claims from the same source:
{scope_block}

Decide which apply (JSON)."""


VALID_KINDS = {"sample", "method", "scope", "limitation", "assumption"}


@dataclass
class ScopeClaim:
    subject: str
    predicate: str
    object_: str
    span: str
    kind: str            # one of VALID_KINDS
    confidence: float


def extract_scope_from_source(
    llm: LLM, source_text: str,
) -> list[ScopeClaim]:
    """Run EXTRACT_SCOPE on the source text. Returns atomic scope claims
    with a kind field. Caller is responsible for adding them via
    store.add_claim (inheriting the source_id) and then linking them as
    conditions for subsequent atomic claims extracted from the same
    source."""
    items = llm.complete_json(
        EXTRACT_SCOPE_SYSTEM,
        EXTRACT_SCOPE_USER_TEMPLATE.format(text=source_text),
    )
    if not isinstance(items, list):
        log("extract_scope_shape_invalid", level="warning",
            got_type=type(items).__name__)
        return []
    results: list[ScopeClaim] = []
    for item in items:
        try:
            subj = str(item["subject"]).strip()
            pred = str(item["predicate"]).strip()
            obj = str(item["object"]).strip()
            span = str(item.get("span", "")).strip()
            kind = str(item.get("kind", "")).strip()
            conf = float(item.get("confidence", 0.7))
        except (KeyError, TypeError, ValueError):
            continue
        if not (subj and pred and obj and span):
            continue
        if kind not in VALID_KINDS:
            continue
        results.append(ScopeClaim(
            subject=subj, predicate=pred, object_=obj,
            span=span, kind=kind, confidence=conf,
        ))
    return results


def link_condition(
    store: Store, claim_id: int, condition_claim_id: int,
    kind: str, *, explicit: bool = True, confidence: float = 0.9,
) -> None:
    """Upsert a claim_conditions row. Validates kind is in VALID_KINDS."""
    if kind not in VALID_KINDS:
        raise ValueError(f"invalid kind {kind!r}; expected one of {sorted(VALID_KINDS)}")
    store.add_claim_condition(claim_id, condition_claim_id, kind, explicit, confidence)


def unlink_condition(
    store: Store, claim_id: int, condition_claim_id: int,
) -> bool:
    """Remove a link. Returns True if a row was deleted."""
    with store.tx() as cx:
        cur = cx.execute(
            "DELETE FROM claim_conditions WHERE claim_id = ? AND condition_claim_id = ?",
            (claim_id, condition_claim_id),
        )
        return cur.rowcount > 0


def conditions_for_claim(store: Store, claim_id: int) -> list[dict]:
    """Return a list of {condition_claim_id, kind, explicit, confidence,
    condition_claim: {id, subject, predicate, object, span_text}}."""
    rows = store.get_claim_conditions(claim_id)
    results: list[dict] = []
    for r in rows:
        span_text = store.get_span_text(r["condition_claim_id"]) or ""
        results.append({
            "condition_claim_id": r["condition_claim_id"],
            "kind": r["kind"],
            "explicit": bool(r["explicit"]),
            "confidence": r["confidence"],
            "condition_claim": {
                "id": r["condition_claim_id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "span_text": span_text,
            },
        })
    return results


def overlap(store: Store, claim_a: int, claim_b: int) -> float:
    """Jaccard overlap between the two claims' condition-claim sets.
    Thin wrapper around store.conditions_overlap; exists so WS-B can
    import from conditions.py without reaching into db.py."""
    return store.conditions_overlap(claim_a, claim_b)


def infer_conditions(
    store: Store, llm: LLM, claim_id: int, *, same_source_only: bool = True,
) -> list[int]:
    """LLM-assisted inference: given a claim, propose which *already-existing*
    scope/method/limitation claims (from the same source by default) apply.
    Writes links with explicit=False. Returns the list of linked condition
    claim IDs.

    This is best-effort and opt-in. Never used by default ingestion."""
    claim = store.get_claim(claim_id)
    if not claim:
        return []

    # Fetch candidate scope claims (predicate = "applies-to" is the stable
    # label for scope claims inserted during ingest with extract_conditions)
    if same_source_only:
        candidates = store.conn.execute(
            "SELECT * FROM claims WHERE source_id = ? AND status = 'active' "
            "AND predicate = 'applies-to' AND id != ?",
            (claim["source_id"], claim_id),
        ).fetchall()
    else:
        candidates = store.conn.execute(
            "SELECT * FROM claims WHERE status = 'active' "
            "AND predicate = 'applies-to' AND id != ?",
            (claim_id,),
        ).fetchall()

    if not candidates:
        return []

    # Filter out candidates that are already linked to this claim
    existing = store.get_claim_conditions(claim_id)
    existing_ids = {r["condition_claim_id"] for r in existing}
    candidates = [c for c in candidates if c["id"] not in existing_ids]
    if not candidates:
        return []

    # Build scope_block
    scope_lines = []
    for c in candidates:
        span_text = store.get_span_text(c["id"]) or ""
        scope_lines.append(
            f"[claim:{c['id']}] kind=scope {c['subject']} — {c['predicate']} — {c['object']}\n"
            f"  span: {span_text!r}"
        )
    scope_block = "\n".join(scope_lines)

    claim_span = store.get_span_text(claim_id) or ""
    user_msg = EXTRACT_CONDITIONS_USER_TEMPLATE.format(
        claim_id=claim_id,
        subj=claim["subject"],
        pred=claim["predicate"],
        obj=claim["object"],
        span=claim_span,
        scope_block=scope_block,
    )

    result = llm.complete_json(EXTRACT_CONDITIONS_SYSTEM, user_msg)
    if not isinstance(result, dict):
        return []
    applies = result.get("applies", [])
    if not isinstance(applies, list):
        return []

    candidate_ids = {c["id"] for c in candidates}
    linked: list[int] = []
    for entry in applies:
        try:
            cond_id = int(entry["condition_claim_id"])
            kind = str(entry.get("kind", "scope"))
        except (KeyError, TypeError, ValueError):
            continue
        if cond_id not in candidate_ids:
            continue
        if kind not in VALID_KINDS:
            kind = "scope"
        link_condition(store, claim_id, cond_id, kind, explicit=False)
        linked.append(cond_id)

    return linked
