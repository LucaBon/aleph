"""Ingest: source text -> atomic claims with exact source spans.

The rule: a claim always points back to a specific span of the source. The span
is the canonical form; the (subject, predicate, object) triple is just the index
card pointing at it. If the extracted triple and the span disagree, the span wins.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .db import Store
from .llm import LLM, cost_usd
from .log import log

EXTRACT_SYSTEM = """You extract atomic claims from source documents.

A claim is:
- ONE factual proposition in (subject, predicate, object) form
- Self-contained: understandable without reading the rest of the source
- Grounded: paraphrases exactly what the source says, never infers or adds
- Anchored to a specific span of source text that supports it

The span must be a VERBATIM substring of the provided source chunk — do not
paraphrase in the span field. Copy the exact characters.

Decompose aggressively. If a sentence contains five facts, emit five claims.
Prefer specific subjects ("lithium-ion batteries in Model S") over generic ones
("batteries"). Keep objects specific ("8 to 10 years") not vague ("a long time").

Emit a JSON array. Each item has:
{
  "proposition": "string, the claim as one self-contained sentence, keeping every number, date, unit, negation and name exactly as the source states them",
  "subject": "string, lowercased noun phrase naming the entity",
  "predicate": "string, lowercased verb phrase",
  "object": "string, the specific value or fact",
  "span": "string, verbatim substring of the source chunk",
  "confidence": number between 0 and 1 reflecting how clearly the source states this
}

Skip material that is speculative, rhetorical, or merely transitional.
Skip headings and metadata unless they assert a fact."""


EXTRACT_USER_TEMPLATE = """Source chunk:
---
{chunk}
---

Extract claims as described. JSON array only."""


def _chunk_text(text: str, target_chars: int = 4000, overlap: int = 200) -> list[tuple[int, str]]:
    """Chunk the text at paragraph boundaries. Returns list of (offset, chunk_text).

    Offset is the character position of the chunk in the original text — we need
    this to translate within-chunk spans back to global source offsets.
    """
    if len(text) <= target_chars:
        return [(0, text)]
    # safety: overlap must leave room for progress
    overlap = min(overlap, target_chars // 4)
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + target_chars, n)
        # try to extend to a paragraph break within the next ~500 chars
        if end < n:
            break_at = text.find("\n\n", end, min(end + 500, n))
            if break_at != -1:
                end = break_at
        chunks.append((i, text[i:end]))
        if end >= n:
            break
        next_i = end - overlap
        i = next_i if next_i > i else i + max(1, target_chars // 2)
    return chunks


def _locate_span(chunk: str, span: str, chunk_offset: int) -> Optional[tuple[int, int]]:
    """Find `span` inside `chunk`. Return (global_start, global_end) or None.

    First tries an exact substring match. If that fails, tries a whitespace-
    tolerant match: runs of whitespace in either side collapse to a single
    space, and we walk the original chunk to recover the true offsets.
    """
    if not span:
        return None
    idx = chunk.find(span)
    if idx >= 0:
        return chunk_offset + idx, chunk_offset + idx + len(span)
    norm_span = re.sub(r"\s+", " ", span).strip()
    if not norm_span:
        return None
    located = _locate_span_whitespace_tolerant(chunk, norm_span, chunk_offset)
    if located is not None:
        log(
            "span_loose_match",
            level="debug",
            span_preview=span[:80],
            recovered_start=located[0],
            recovered_end=located[1],
        )
    return located


def _locate_span_whitespace_tolerant(
    chunk: str, norm_span: str, chunk_offset: int
) -> Optional[tuple[int, int]]:
    """Scan `chunk` for `norm_span`, treating every run of whitespace in `chunk`
    as a single space. Returns the original (unnormalized) byte offsets.

    Two-pointer walk: at each candidate start position `i` in `chunk`, try to
    match `norm_span` character-by-character, collapsing whitespace runs on the
    chunk side. On match, the end pointer points at the last matched char + 1.
    """
    n = len(chunk)
    m = len(norm_span)
    for start in range(n):
        ci = start
        si = 0
        while si < m and ci < n:
            sc = norm_span[si]
            cc = chunk[ci]
            if sc == " " and cc.isspace():
                # consume a run of whitespace on the chunk side to match one space
                while ci < n and chunk[ci].isspace():
                    ci += 1
                si += 1
            elif sc == cc:
                ci += 1
                si += 1
            else:
                break
        if si == m:
            return chunk_offset + start, chunk_offset + ci
    return None


def ingest_file(store: Store, llm: LLM, path: Path, verbose: bool = False, extract_conditions: bool = False) -> dict:
    """Ingest a single file. Returns a summary dict."""
    text = path.read_text(encoding="utf-8", errors="replace")
    source_id = store.add_source(str(path), text)
    if source_id is None:
        return {"path": str(path), "status": "skipped", "reason": "already ingested"}

    # Snapshot before any LLM call, the conditions pre-pass included.
    usage_before = llm.usage.copy() if hasattr(llm, "usage") else None

    # WS-D: pre-pass to extract scope/method/sample/etc. claims
    scope_claim_ids: list[tuple[int, str]] = []  # (claim_id, kind)
    if extract_conditions:
        scope_claim_ids = _extract_scope_claims(store, llm, source_id, text)

    total_claims = 0
    flagged = 0
    dropped = 0
    dropped_details: list[dict] = []
    chunks = _chunk_text(text)
    for chunk_offset, chunk in chunks:
        if verbose:
            print(f"  extracting from chars {chunk_offset}-{chunk_offset+len(chunk)}...")
        try:
            items = llm.complete_json(EXTRACT_SYSTEM, EXTRACT_USER_TEMPLATE.format(chunk=chunk))
        except Exception as e:
            log(
                "extract_chunk_failed",
                level="warning",
                source_id=source_id,
                chunk_offset=chunk_offset,
                error=type(e).__name__,
                message=str(e),
            )
            if verbose:
                print(f"  [warn] extraction failed for chunk at {chunk_offset}: {e}")
            continue
        if not isinstance(items, list):
            log(
                "extract_shape_invalid",
                level="warning",
                source_id=source_id,
                chunk_offset=chunk_offset,
                got_type=type(items).__name__,
            )
            continue
        for item in items:
            try:
                subj = str(item["subject"]).strip()
                pred = str(item["predicate"]).strip()
                obj = str(item["object"]).strip()
                span_text = str(item.get("span", "")).strip()
                conf = float(item.get("confidence", 0.7))
                proposition = str(item.get("proposition") or "").strip() or None
            except (KeyError, TypeError, ValueError) as e:
                dropped += 1
                dropped_details.append({
                    "reason": "malformed_fields",
                    "chunk_offset": chunk_offset,
                    "error": type(e).__name__,
                })
                log(
                    "claim_dropped",
                    level="info",
                    reason="malformed_fields",
                    source_id=source_id,
                    error=type(e).__name__,
                )
                continue
            if not (subj and pred and obj and span_text):
                dropped += 1
                dropped_details.append({
                    "reason": "missing_fields",
                    "chunk_offset": chunk_offset,
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                })
                log(
                    "claim_dropped",
                    level="info",
                    reason="missing_fields",
                    source_id=source_id,
                    subject=subj,
                    predicate=pred,
                )
                continue
            located = _locate_span(chunk, span_text, chunk_offset)
            if located is None:
                # claim couldn't be grounded in the source — refuse to store it
                dropped += 1
                dropped_details.append({
                    "reason": "span_not_in_source",
                    "chunk_offset": chunk_offset,
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "span_preview": span_text[:160],
                })
                log(
                    "claim_dropped",
                    level="warning",
                    reason="span_not_in_source",
                    source_id=source_id,
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    span_preview=span_text[:120],
                )
                continue
            start, end = located
            atomic_claim_id = store.add_claim(source_id, subj, pred, obj, start, end, conf,
                                              proposition=proposition)
            total_claims += 1
            if store.open_claim_review(atomic_claim_id, "fidelity"):
                flagged += 1
            # WS-D: link every scope claim as a condition of this atomic claim
            if extract_conditions and scope_claim_ids:
                from . import conditions as _cond_mod
                for scope_id, scope_kind in scope_claim_ids:
                    _cond_mod.link_condition(store, atomic_claim_id, scope_id, scope_kind, explicit=True)
    # any cached views are now potentially stale
    store.clear_cache()
    result = {
        "path": str(path),
        "status": "ingested",
        "source_id": source_id,
        "chunks": len(chunks),
        "claims_added": total_claims,
        "claims_flagged_fidelity": flagged,
        "claims_dropped_ungrounded": dropped,
        "dropped": dropped_details,
    }
    result.update(_cost_report(llm, usage_before, text))
    # WS-D: include scope claim info when extract_conditions was used
    if extract_conditions:
        result["scope_claims_added"] = len(scope_claim_ids)
        result["scope_claim_ids"] = [sid for sid, _kind in scope_claim_ids]
    return result


def _cost_report(llm, usage_before, text: str) -> dict:
    """LLM usage and list-price cost of this ingest, per 1k source tokens.
    Source tokens are estimated as chars / 4 (no network token count)."""
    source_tokens = len(text) / 4
    report = {
        "llm_usage": None,
        "cost_usd": None,
        "source_chars": len(text),
        "source_tokens_estimate": source_tokens,
        "cost_per_1k_source_tokens": None,
    }
    if usage_before is None:
        return report
    usage = llm.usage.minus(usage_before)
    report["llm_usage"] = usage.to_dict()
    cost = cost_usd(getattr(llm, "model", ""), usage)
    report["cost_usd"] = cost
    if cost is not None and source_tokens:
        report["cost_per_1k_source_tokens"] = cost / source_tokens * 1000
    return report


# ----- WS-D scope extraction (opt-in via extract_conditions=True) -----

def _extract_scope_claims(store: Store, llm: LLM, source_id: int, text: str) -> list[tuple[int, str]]:
    """Pre-pass: extract scope/method/sample/limitation/assumption claims from the
    whole document. Returns list of (claim_id, kind) pairs for linking."""
    from . import conditions

    scope_items = conditions.extract_scope_from_source(llm, text)
    result: list[tuple[int, str]] = []
    for scope in scope_items:
        located = _locate_span(text, scope.span, 0)
        if located is None:
            log(
                "scope_claim_dropped",
                level="warning",
                reason="span_not_in_source",
                source_id=source_id,
                subject=scope.subject,
                span_preview=scope.span[:120],
            )
            continue
        start, end = located
        claim_id = store.add_claim(
            source_id, scope.subject, "applies-to", scope.object_,
            start, end, scope.confidence,
        )
        result.append((claim_id, scope.kind))
    return result


def ingest_paths(store: Store, llm: LLM, paths: list[Path], verbose: bool = False, extract_conditions: bool = False) -> list[dict]:
    """Ingest one or more paths. Recurses into directories."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in {".txt", ".md", ".markdown"}:
                    files.append(f)
        elif p.is_file():
            files.append(p)
    results = []
    for f in files:
        if verbose:
            print(f"ingesting {f}...")
        results.append(ingest_file(store, llm, f, verbose=verbose, extract_conditions=extract_conditions))
    return results
