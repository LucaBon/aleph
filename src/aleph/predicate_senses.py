"""Predicate sense disambiguation (WS-E.2).

A *sense* is a ``(canonical, sense_tag, domain)`` triple naming one specific
meaning of a surface predicate. Claims optionally point at one sense via
``claim_predicate_senses``, with an attributed provenance trail
(``assigned_by``, ``explicit``, ``rationale``) mirroring
``claim_conditions``.

The verbatim ``claim.predicate`` is never modified. Sense lookup happens at
retrieval and contradiction-pre-filter time, and through the new agent
commands. LLM-assisted assignment writes ``assigned_by='llm', explicit=0``
and is never silently promoted.
"""
from __future__ import annotations

from typing import Optional

from .db import Store
from .llm import LLM
from .log import log


# ----- prompts (verbatim from docs/plan/90-prompts.md) -----

PREDICATE_SENSE_EXTRACT_SYSTEM = """\
You assign a sense to a claim's predicate from a closed list of
candidates.

You are given:
- a claim with subject, predicate, and object
- the claim's verbatim source span
- the source's domain
- a list of candidate predicate senses for that predicate in that
  domain (each with id, sense_tag, definition, optional parent and
  inverse)

Pick the sense that best matches the claim's actual relation, or
reply with no match if no candidate fits. Do not invent senses. Do
not paraphrase the claim or the span; judge the relation as stated.

Reply in JSON:
{"sense_id": int | null, "confidence": number 0..1, "rationale": "one short sentence"}"""

PREDICATE_SENSE_EXTRACT_USER_TEMPLATE = """\
Claim: [claim:{claim_id}] {subject} — {predicate} — {object}
Source span: {span!r}
Source domain: {domain}

Candidate senses for predicate {predicate!r} in domain {domain!r}:
{candidates_block}

Assign a sense (JSON)."""


def _format_candidate(row) -> str:
    parts = [f"[sense:{row['id']}] {row['sense_tag']}"]
    if row["definition"]:
        parts.append(f"— {row['definition']}")
    extras = []
    if row["parent_id"] is not None:
        extras.append(f"parent: sense:{row['parent_id']}")
    if row["inverse_id"] is not None:
        extras.append(f"inverse: sense:{row['inverse_id']}")
    line = " ".join(parts)
    if extras:
        line += " (" + "; ".join(extras) + ")"
    return line


def infer_sense(
    store: Store, llm: LLM, claim_id: int,
) -> Optional[tuple[Optional[int], float, str]]:
    """Ask the LLM which sense (if any) fits this claim.

    Builds the candidate list filtered by ``(claim's resolve_predicate,
    source.domain)``, formats the prompt, calls ``llm.complete_json``,
    validates the returned ``sense_id`` is in the candidate set, and
    returns ``(sense_id, confidence, rationale)`` or ``None`` if no
    candidate fits or the response is malformed.

    Always writes via ``set_claim_predicate_sense(..., assigned_by='llm',
    explicit=False)``. NEVER overwrites a row whose existing
    ``explicit=1`` — operator attestations beat LLM proposals.
    """
    claim = store.get_claim(claim_id)
    if not claim:
        return None
    meta = store.get_source_metadata(claim["source_id"])
    if not meta:
        return None
    domain = meta["domain"]
    canonical = store.resolve_predicate(claim["predicate"], domain=domain)
    candidates = store.list_predicate_senses(
        canonical=canonical, domain=domain,
    )
    if not candidates:
        return None

    candidate_ids = {row["id"] for row in candidates}
    block = "\n".join(_format_candidate(r) for r in candidates)
    span = store.get_span_text(claim_id) or ""
    user_msg = PREDICATE_SENSE_EXTRACT_USER_TEMPLATE.format(
        claim_id=claim_id,
        subject=claim["subject"],
        predicate=claim["predicate"],
        object=claim["object"],
        span=span[:400],
        domain=domain,
        candidates_block=block,
    )

    try:
        result = llm.complete_json(
            PREDICATE_SENSE_EXTRACT_SYSTEM, user_msg, max_tokens=256,
        )
    except Exception:
        log("predicate_sense_extract_failed", level="warning",
            claim_id=claim_id)
        return None
    if not isinstance(result, dict):
        return None
    raw_id = result.get("sense_id")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    rationale = str(result.get("rationale", "")).strip()
    sense_id: Optional[int]
    if raw_id is None:
        sense_id = None
    else:
        try:
            sense_id = int(raw_id)
        except (TypeError, ValueError):
            return None
        if sense_id not in candidate_ids:
            return None

    # Honesty contract: never overwrite an explicit assignment.
    existing = store.conn.execute(
        "SELECT explicit FROM claim_predicate_senses WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    if existing is not None and existing["explicit"]:
        return ("explicit_locked", confidence, rationale)

    if sense_id is not None:
        store.set_claim_predicate_sense(
            claim_id, sense_id,
            assigned_by="llm", confidence=confidence,
            explicit=False, rationale=rationale or None,
        )
    return (sense_id, confidence, rationale)
