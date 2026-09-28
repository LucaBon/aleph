"""Concepts: higher-order propositions grounded by sets of claims.

A concept is a statement about a subject that is supported by one or more
claims. Unlike a view (ephemeral prose), a concept is a persistent node that
other concepts and answers can cite. Concepts require a GROUNDED verdict
against their support set before transitioning from draft -> active.

See docs/plan/10-ws-concepts.md for the full spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .db import Store
from .llm import LLM
from .log import log


# ----- prompts (verbatim from docs/plan/90-prompts.md) -----

CONCEPT_DERIVE_SYSTEM = """\
You propose concepts from a set of atomic claims about one subject.

A concept is ONE proposition that summarizes, generalizes, or synthesizes
claims. It must be grounded: every factual element of the concept statement
must be supported by the span of at least one claim you cite.

Rules:
- Use only the claim IDs provided. Do not invent claim IDs.
- Emit ZERO or more concepts. If the claims do not cohere, emit an empty array.
- Each concept must list its supporting claim IDs with a role:
  premise | corroborating | qualifying | counterexample.
- Prefer specific statements over vague ones.
- If the claims disagree, do not paper over. Either skip the concept or
  emit it with the dissenting claim tagged role=counterexample and note the
  tension in the statement.
- Classify the concept by inference_type:
  * summary: compact restatement of what the claims collectively say
  * generalization: a broader rule inferred from multiple specific claims
  * synthesis: a new proposition combining claims in a non-obvious way

Emit a JSON array. Each item has:
{
  "statement": "string, one proposition",
  "inference_type": "summary" | "generalization" | "synthesis",
  "confidence": number between 0 and 1,
  "supports": [{"claim_id": int, "role": "premise" | "corroborating" | "qualifying" | "counterexample"}, ...]
}"""

CONCEPT_DERIVE_USER_TEMPLATE = """\
Subject: {subject}

Claims (ID, confidence, S -- P -- O):
{claims_block}

Each claim's source span (for grounding reference):
{spans_block}

Propose concepts now. JSON array only."""

CONCEPT_VALIDATE_SYSTEM = """\
You are a validator. Given a concept statement and the verbatim source spans
of its supporting claims, decide whether every factual element of the
statement is supported by the union of the spans.

Rules:
- GROUNDED: every factual element is stated or clearly implied by at least
  one span.
- PARTIAL: some factual elements are supported, but others are not.
- UNGROUNDED: the spans do not support the statement.

Do not judge whether the statement is true in the world. Judge only whether
it follows from the spans.

Reply in JSON:
{"verdict": "GROUNDED" | "PARTIAL" | "UNGROUNDED", "reason": "one short sentence"}"""

CONCEPT_VALIDATE_USER_TEMPLATE = """\
Concept statement: {statement}

Supporting spans (each bracketed block is one span):
{spans_block}

Verdict (JSON)."""


@dataclass
class ConceptProposal:
    statement: str
    inference_type: str  # summary | generalization | synthesis
    confidence: float
    support_claim_ids: list[tuple[int, str]] = field(default_factory=list)  # (claim_id, role)


def _build_claims_block(claims: list) -> str:
    """Format claims for the derivation prompt."""
    lines = []
    for c in claims:
        lines.append(
            f"[claim:{c['id']}] ({c['confidence']:.2f}) "
            f"{c['subject']} — {c['predicate']} — {c['object']}"
        )
    return "\n".join(lines)


def _build_spans_block_derive(store: Store, claims: list) -> str:
    """Format spans for the derivation prompt."""
    lines = []
    for c in claims:
        span = store.get_span_text(c["id"]) or ""
        lines.append(f"[claim:{c['id']}] {span!r}")
    return "\n".join(lines)


def _build_spans_block_validate(store: Store, support_rows: list) -> str:
    """Format spans for the validation prompt."""
    blocks = []
    for i, row in enumerate(support_rows, 1):
        span = store.get_span_text(row["claim_id"]) or ""
        blocks.append(f"--- span {i} ---\n{span}")
    return "\n".join(blocks)


def derive_concepts(
    store: Store, llm: LLM, subject: str, *,
    claim_ids: Optional[list[int]] = None, limit: int = 50,
) -> list[int]:
    """Run the derivation prompt for a subject. Writes each proposed concept
    as status='draft' with its support set, runs validation, transitions to
    'active' if GROUNDED. Returns the new concept IDs (both active and
    draft).

    If claim_ids is provided, restrict derivation to those claims; otherwise
    use all active claims on the alias-resolved subject (capped at limit).
    """
    canonical = store.resolve_subject(subject)
    if claim_ids:
        claims = []
        for cid in claim_ids:
            row = store.get_claim(cid)
            if row and row["status"] == "active":
                claims.append(row)
    else:
        claims = store.conn.execute(
            "SELECT * FROM claims WHERE subject = ? AND status = 'active' "
            "ORDER BY confidence DESC, extracted_at DESC LIMIT ?",
            (canonical, limit),
        ).fetchall()

    if not claims:
        return []

    claims_block = _build_claims_block(claims)
    spans_block = _build_spans_block_derive(store, claims)

    user_msg = CONCEPT_DERIVE_USER_TEMPLATE.format(
        subject=canonical,
        claims_block=claims_block,
        spans_block=spans_block,
    )

    proposals_raw = llm.complete_json(CONCEPT_DERIVE_SYSTEM, user_msg)
    if not isinstance(proposals_raw, list):
        proposals_raw = []

    # parse proposals
    proposals: list[ConceptProposal] = []
    valid_claim_ids = {c["id"] for c in claims}
    for item in proposals_raw:
        if not isinstance(item, dict):
            continue
        stmt = item.get("statement", "")
        itype = item.get("inference_type", "summary")
        if itype not in ("summary", "generalization", "synthesis"):
            itype = "summary"
        conf = float(item.get("confidence", 0.7))
        supports = []
        seen_cids: set[int] = set()
        for s in item.get("supports", []):
            cid = s.get("claim_id")
            role = s.get("role", "premise")
            if cid in valid_claim_ids and cid not in seen_cids:
                supports.append((cid, role))
                seen_cids.add(cid)
        if stmt and supports:
            proposals.append(ConceptProposal(
                statement=stmt,
                inference_type=itype,
                confidence=conf,
                support_claim_ids=supports,
            ))

    new_ids: list[int] = []
    for p in proposals:
        concept_id = store.add_concept(
            subject=canonical,
            statement=p.statement,
            inference_type=p.inference_type,
            confidence=p.confidence,
            support_claim_ids=p.support_claim_ids,
            status="draft",
        )
        # validate
        verdict, reason = validate_concept(store, llm, concept_id)
        new_ids.append(concept_id)

    return new_ids


def validate_concept(
    store: Store, llm: LLM, concept_id: int,
) -> tuple[str, str]:
    """Run the GROUNDED verdict prompt against the concept's support set.
    Returns (verdict, reason). Updates the concept row: verdict goes to
    validation_verdict; if GROUNDED, status becomes 'active'; otherwise
    concept stays in current status (draft or stale)."""
    concept = store.get_concept(concept_id)
    if not concept:
        return ("UNGROUNDED", "concept not found")

    supports = store.get_concept_supports(concept_id)
    if not supports:
        store.update_concept_status(
            concept_id, concept["status"],
            validation_verdict="UNGROUNDED",
            validation_reason="no supporting claims",
        )
        return ("UNGROUNDED", "no supporting claims")

    spans_block = _build_spans_block_validate(store, supports)
    user_msg = CONCEPT_VALIDATE_USER_TEMPLATE.format(
        statement=concept["statement"],
        spans_block=spans_block,
    )

    result = llm.complete_json(CONCEPT_VALIDATE_SYSTEM, user_msg)
    verdict = result.get("verdict", "UNGROUNDED") if isinstance(result, dict) else "UNGROUNDED"
    reason = result.get("reason", "") if isinstance(result, dict) else ""

    if verdict not in ("GROUNDED", "PARTIAL", "UNGROUNDED"):
        verdict = "UNGROUNDED"

    # A concept can't become citable while it rests on an inactive claim,
    # however well the spans read.
    inactive = [
        s["claim_id"] for s in supports
        if (c := store.get_claim(s["claim_id"])) is None or c["status"] != "active"
    ]
    if verdict == "GROUNDED" and inactive:
        verdict = "UNGROUNDED"
        reason = f"support claims not active: {inactive}"

    new_status = "active" if verdict == "GROUNDED" else concept["status"]
    store.update_concept_status(
        concept_id, new_status,
        validation_verdict=verdict,
        validation_reason=reason,
    )

    log("concept_validated", level="info",
        concept_id=concept_id, verdict=verdict, new_status=new_status)

    return (verdict, reason)


def rebuild_concept(
    store: Store, llm: LLM, concept_id: int,
) -> int:
    """For a stale/active concept, re-run derivation against the current
    active support set. Always creates a NEW concept row and supersedes
    the old one. Returns the new concept id."""
    old = store.get_concept(concept_id)
    if not old:
        raise ValueError(f"concept {concept_id} not found")

    supports = store.get_concept_supports(concept_id)
    # use only active supporting claims
    active_supports = [
        s for s in supports if s["claim_status"] == "active"
    ]

    if not active_supports:
        # no active claims remain; invalidate
        store.update_concept_status(
            concept_id, "invalidated",
            validation_verdict="UNGROUNDED",
            validation_reason="no active supporting claims remain",
        )
        return concept_id

    # gather claim data for derivation
    claim_ids = [s["claim_id"] for s in active_supports]
    claims = [store.get_claim(cid) for cid in claim_ids]
    claims = [c for c in claims if c is not None]

    claims_block = _build_claims_block(claims)
    spans_block = _build_spans_block_derive(store, claims)

    user_msg = CONCEPT_DERIVE_USER_TEMPLATE.format(
        subject=old["subject"],
        claims_block=claims_block,
        spans_block=spans_block,
    )

    proposals_raw = llm.complete_json(CONCEPT_DERIVE_SYSTEM, user_msg)
    if not isinstance(proposals_raw, list):
        proposals_raw = []

    valid_claim_ids = {c["id"] for c in claims}

    # pick the best proposal (or the first one)
    best = None
    for item in proposals_raw:
        if not isinstance(item, dict):
            continue
        stmt = item.get("statement", "")
        itype = item.get("inference_type", "summary")
        if itype not in ("summary", "generalization", "synthesis"):
            itype = "summary"
        conf = float(item.get("confidence", 0.7))
        sups = []
        seen_cids: set[int] = set()
        for s in item.get("supports", []):
            cid = s.get("claim_id")
            role = s.get("role", "premise")
            if cid in valid_claim_ids and cid not in seen_cids:
                sups.append((cid, role))
                seen_cids.add(cid)
        if stmt and sups:
            best = ConceptProposal(
                statement=stmt, inference_type=itype,
                confidence=conf, support_claim_ids=sups,
            )
            break

    if best is None:
        # derivation produced nothing; keep old concept, mark invalidated
        store.update_concept_status(
            concept_id, "invalidated",
            validation_verdict="UNGROUNDED",
            validation_reason="rebuild produced no proposals",
        )
        return concept_id

    # create new concept
    new_id = store.add_concept(
        subject=old["subject"],
        statement=best.statement,
        inference_type=best.inference_type,
        confidence=best.confidence,
        support_claim_ids=best.support_claim_ids,
        status="draft",
    )

    # validate the new concept
    verdict, reason = validate_concept(store, llm, new_id)

    # supersede the old concept
    store.update_concept_status(
        concept_id, "superseded", superseded_by=new_id,
    )

    log("concept_rebuilt", level="info",
        old_id=concept_id, new_id=new_id, verdict=verdict)

    return new_id


def invalidate_concept(
    store: Store, concept_id: int, reason: str,
) -> None:
    """Mark a concept status='invalidated' with the given reason.
    Cascades: any cached view citing this concept is dropped."""
    store.update_concept_status(
        concept_id, "invalidated",
        validation_verdict=None,
        validation_reason=reason,
    )
    log("concept_invalidated", level="info",
        concept_id=concept_id, reason=reason)
