"""Query pipeline: question -> claim retrieval -> view synthesis -> verifier pass.

Views are NOT stored as files. They're generated on demand from claims and
verified against their source spans before being returned. If a sentence can't
be grounded in the claim it cites, it gets flagged.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .db import Store
from .llm import LLM
from .log import log
from .retrieval import Retriever, default_retriever

# Verdict ordering (worst wins when aggregating a multi-citation sentence).
# ERROR means the verifier itself failed (couldn't decide). UNGROUNDED means
# the verifier decided the span does not support the sentence. Both surface
# as non-GROUNDED but have very different meanings for the caller.
_VERDICT_RANK = {"GROUNDED": 0, "PARTIAL": 1, "UNGROUNDED": 2, "ERROR": 3}

SYNTHESIZE_SYSTEM = """You answer a question using ONLY the provided claims.

Rules:
- Every factual sentence must end with one or more citations: [claim:ID] or [claim:ID,ID].
- Use only claim IDs present in the provided list. Never invent IDs.
- Do not add facts that aren't in the claims. If coverage is incomplete, say so explicitly.
- Keep it tight. Prefer short, specific sentences — each one anchored to a claim.
- Group related claims into coherent prose, but preserve the specificity of each claim.
- If claims disagree, surface the disagreement rather than picking one silently.

Format: plain markdown. No headings unless the answer genuinely has sections.
Start with a direct answer to the question."""


SYNTHESIZE_USER_TEMPLATE = """Question: {query}

Available claims (use only these — each has an ID you must cite):
{claims_block}

Write the answer now."""


VERIFY_SYSTEM = """You are a verifier. Given a source span and a sentence that
cites it, determine whether the sentence is supported by the span.

Reply in JSON: {"verdict": "GROUNDED" | "PARTIAL" | "UNGROUNDED", "reason": "one short sentence"}

- GROUNDED: every factual element of the sentence is stated or clearly implied by the span.
- PARTIAL: some of the sentence is supported, but it adds or changes something.
- UNGROUNDED: the span does not support the sentence."""


VERIFY_USER_TEMPLATE = """Source span:
---
{span}
---

Sentence: {sentence}

Verdict (JSON)."""


@dataclass
class Citation:
    sentence: str
    claim_ids: list[int]
    verdict: str  # aggregate: GROUNDED | PARTIAL | UNGROUNDED | ERROR
    reason: str   # aggregate reason (the worst claim's reason)
    # per-claim verdicts for multi-citation sentences. Empty when no cited claim
    # could be verified (e.g. all missing).
    per_claim: dict[int, tuple[str, str]] = field(default_factory=dict)


@dataclass
class QueryResult:
    query: str
    answer: str  # annotated markdown with flags
    citations: list[Citation]
    claim_ids_used: list[int]
    from_cache: bool


CITE_RE = re.compile(r"\[claim:([0-9,\s]+)\]")


def _extract_keywords(query: str) -> list[str]:
    stop = {
        "the", "and", "for", "with", "what", "when", "where", "how", "why",
        "are", "is", "was", "were", "does", "do", "did", "a", "an", "of",
        "in", "on", "to", "from", "that", "this", "those", "these", "about",
        "between", "compare", "relate", "relates", "tell", "me",
    }
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]+", query.lower())
    return [w for w in words if w not in stop and len(w) > 2]


def _format_claims_block(claims_with_spans: list[tuple]) -> str:
    lines = []
    for claim, span in claims_with_spans:
        lines.append(
            f"[claim:{claim['id']}] ({claim['confidence']:.2f}) "
            f"{claim['subject']} — {claim['predicate']} — {claim['object']}\n"
            f"    source span: {span[:300]!r}{'...' if len(span) > 300 else ''}"
        )
    return "\n\n".join(lines)


# Abbreviations that end in a period but don't close a sentence. When the naive
# splitter breaks on them, we re-merge post-hoc. Kept small on purpose.
_ABBREVIATIONS = (
    "Dr.", "Mr.", "Mrs.", "Ms.", "St.",
    "e.g.", "i.e.", "etc.", "vs.", "cf.",
    "No.", "Fig.", "a.m.", "p.m.",
)


def _split_sentences(text: str) -> list[str]:
    # rough sentence split that preserves markdown structure
    # split on sentence terminators followed by whitespace+capital, keeping the terminator
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", text.strip())
    merged: list[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        # if the previous fragment ended in a known abbreviation, the split was
        # spurious — glue this fragment back onto it
        if merged and any(merged[-1].endswith(abbr) for abbr in _ABBREVIATIONS):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def _parse_citations(sentence: str) -> list[int]:
    ids: list[int] = []
    for match in CITE_RE.finditer(sentence):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
    return ids


def query(
    store: Store,
    llm: LLM,
    question: str,
    retrieve_k: int = 30,
    verify: bool = True,
    use_cache: bool = True,
    retriever: Retriever | None = None,
) -> QueryResult:
    """Run the full question -> verified answer pipeline.

    `retriever` defaults to FTS5 if the store supports it, else keyword
    substring search. Pass an explicit Retriever to plug in a different
    backend (e.g. embeddings) without touching this function.
    """
    if use_cache:
        cached = store.get_cached_view(question)
        if cached:
            return QueryResult(
                query=question,
                answer=cached["response"],
                citations=[],
                claim_ids_used=json.loads(cached["claim_ids"]),
                from_cache=True,
            )

    # 1. retrieve candidate claims
    keywords = _extract_keywords(question)
    retr = retriever if retriever is not None else default_retriever(store)
    claim_rows = retr.search(keywords, limit=retrieve_k)
    if not claim_rows:
        return QueryResult(
            query=question,
            answer=(
                "_No relevant claims in the knowledge base. Ingest more sources, "
                "or check whether this topic is covered by what you've ingested._"
            ),
            citations=[],
            claim_ids_used=[],
            from_cache=False,
        )

    # span_text is joined inline by the retriever — no per-claim round-trip
    claims_with_spans = [(row, row["span_text"] or "") for row in claim_rows]

    # 2. synthesize view
    claims_block = _format_claims_block(claims_with_spans)
    answer = llm.complete(
        SYNTHESIZE_SYSTEM,
        SYNTHESIZE_USER_TEMPLATE.format(query=question, claims_block=claims_block),
        max_tokens=2048,
    )

    # split once: both verification and cache-key computation walk the sentences
    sentences = _split_sentences(answer)
    retrieved_ids = {r["id"] for r in claim_rows}

    # 3. verifier pass: for each cited claim in each sentence, check grounding
    citations: list[Citation] = []
    if verify:
        for sent in sentences:
            cited_ids = _parse_citations(sent)
            if not cited_ids:
                continue
            per_claim: dict[int, tuple[str, str]] = {}
            for cid in cited_ids:
                per_claim[cid] = _verify_one(llm, store, sent, cid)
            agg_verdict = _aggregate_verdict(v for v, _ in per_claim.values())
            # aggregate reason = the worst claim's reason (first match)
            agg_reason = next(
                (r for v, r in per_claim.values() if v == agg_verdict), ""
            )
            citations.append(Citation(sent, cited_ids, agg_verdict, agg_reason, per_claim))

    # 4. annotate answer with verdicts
    annotated = _annotate_answer(answer, citations) if verify else answer

    # cache-key claims = the ones the LLM actually cited (intersected with the
    # retrieved set, so hallucinated IDs don't end up in the index). This keeps
    # invalidation precise: supersede/remove only invalidates views that truly
    # depended on the affected claim.
    cited_across_answer: set[int] = set()
    for sent in sentences:
        for cid in _parse_citations(sent):
            if cid in retrieved_ids:
                cited_across_answer.add(cid)
    claim_ids_used = sorted(cited_across_answer)
    store.cache_view(question, annotated, claim_ids_used)
    return QueryResult(
        query=question,
        answer=annotated,
        citations=citations,
        claim_ids_used=claim_ids_used,
        from_cache=False,
    )


def _verify_one(llm: LLM, store: Store, sentence: str, claim_id: int) -> tuple[str, str]:
    """Verify a sentence against a single cited claim's span.

    Returns (verdict, reason). Verdict is one of GROUNDED | PARTIAL | UNGROUNDED
    | ERROR. ERROR means the verifier itself couldn't decide (parse/API failure),
    distinct from UNGROUNDED (the verifier decided the span does not support it).
    """
    span = store.get_span_text(claim_id)
    if span is None:
        return "UNGROUNDED", "cited claim not found"
    try:
        result = llm.complete_json(
            VERIFY_SYSTEM,
            VERIFY_USER_TEMPLATE.format(span=span, sentence=sentence),
            max_tokens=256,
        )
        verdict = str(result.get("verdict", "UNGROUNDED")).upper()
        if verdict not in {"GROUNDED", "PARTIAL", "UNGROUNDED"}:
            verdict = "UNGROUNDED"
        reason = str(result.get("reason", ""))
        return verdict, reason
    except Exception as e:
        log(
            "verifier_error",
            level="warning",
            claim_id=claim_id,
            error=type(e).__name__,
            message=str(e),
        )
        return "ERROR", f"verifier error: {e}"


def _aggregate_verdict(verdicts) -> str:
    """Worst verdict wins. Empty iterator aggregates to GROUNDED (caller
    already guarantees there's at least one cited claim when used)."""
    worst = "GROUNDED"
    for v in verdicts:
        if _VERDICT_RANK.get(v, 2) > _VERDICT_RANK.get(worst, 0):
            worst = v
    return worst


def _annotate_answer(answer: str, citations: list[Citation]) -> str:
    """Append a warning footer listing every non-GROUNDED per-claim verdict.

    For multi-citation sentences, each failing claim gets its own line so the
    caller can see which specific claim(s) the verifier rejected — not just
    that "the sentence had a problem".
    """
    flagged = [c for c in citations if c.verdict != "GROUNDED"]
    if not flagged:
        return answer
    lines = ["\n\n---", "**Verifier flags:**"]
    for c in flagged:
        # prefer per-claim detail when available
        if c.per_claim:
            for cid in c.claim_ids:
                v, r = c.per_claim.get(cid, (c.verdict, c.reason))
                if v == "GROUNDED":
                    continue
                lines.append(f"- [{v}] `[claim:{cid}]` — {r}")
        else:
            ids = ",".join(str(i) for i in c.claim_ids)
            lines.append(f"- [{c.verdict}] `[claim:{ids}]` — {c.reason}")
        lines.append(f"  > {c.sentence[:160]}{'...' if len(c.sentence) > 160 else ''}")
    return answer + "\n".join(lines)
