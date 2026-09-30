"""Query pipeline: question -> claim retrieval -> view synthesis -> verifier pass.

Views are NOT stored as files. They're generated on demand from claims and
verified against their source spans before being returned. If a sentence can't
be grounded in the claim it cites, it gets flagged.

Phase 2 extensions: concept retrieval, context filtering, disposition-aware
grouping, concept citation verification, SYNTHESIZE_V2 prompt.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .db import Store, _statuses_sql, view_cache_key
from .llm import LLM
from .log import log
from .retrieval import Retriever, default_retriever

# ---------------------------------------------------------------------------
# Verdict ordering (worst wins when aggregating a multi-citation sentence).
# ERROR means the verifier itself failed (couldn't decide). UNGROUNDED means
# the verifier decided the span does not support the sentence. Both surface
# as non-GROUNDED but have very different meanings for the caller.
# ---------------------------------------------------------------------------
# UNCERTAIN comes only from the opt-in layered verifier: no layer could decide.
_VERDICT_RANK = {"GROUNDED": 0, "PARTIAL": 1, "UNCERTAIN": 1.5, "UNGROUNDED": 2, "ERROR": 3}

# Layered-verifier verdicts on ask's scale.
_LAYERED_TO_ASK = {"SUPPORTED": "GROUNDED", "UNSUPPORTED": "UNGROUNDED",
                   "UNCERTAIN": "UNCERTAIN"}

# ---------------------------------------------------------------------------
# Prompts — v1 kept for reference; v2 is always used now.
# ---------------------------------------------------------------------------

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


# --- SYNTHESIZE_V2 (verbatim from 90-prompts.md) ---

SYNTHESIZE_V2_SYSTEM = """You answer a question using ONLY the provided claims and concepts.

Rules:
- Every factual sentence must end with one or more citations.
  Claim citation: [claim:ID] or [claim:ID,ID]
  Concept citation: [concept:ID]
  Sentences may mix both: [claim:7,concept:3]
- Use only claim IDs and concept IDs present in the provided lists.
- Do not add facts that aren't in the claims/concepts. If coverage is
  incomplete, say so explicitly.
- Keep it tight. Prefer short, specific sentences anchored to evidence.
- NEVER silently pick a side of a disposed contradiction. The
  'dispositions' block below tells you how to present each pair.

Dispositions handling (read carefully — this is the core of the answer's
epistemic shape):

- replicate: Present as ONE finding with all member IDs cited together.
  Example: "Finding X is supported by multiple studies [claim:12,47,88]."
  Do NOT list them as separate findings.

- reconcile: Present both findings WITH their conditions, then the
  reconciling rule. Cite the rule via its concept ID. Example:
  "Under sample S1, X is observed [claim:A]. Under sample S2, Y is observed
  instead [claim:B]. The apparent disagreement is explained by the
  reconciling mechanism [concept:C]."

- coexist: Present both findings WITH their scopes. Example:
  "Under scope_A, X [claim:A]. Under scope_B, Y [claim:B]."
  The scopes come from each claim's source metadata (jurisdiction, date,
  authority_level).

- distinguish: Present both as SEPARATE non-conflicting findings with a
  brief note that their scopes do not overlap. Example: "X applies to
  adults [claim:A]. Y applies to minors [claim:B]. The two do not
  overlap."

- dispute: Present both, flag the disagreement openly. Example: "Study A
  finds X [claim:A]; study B finds Y [claim:B]. The field has not
  resolved this."

- gap: Present both, surface as an open question. Use this exact phrase:
  "Findings conflict [claim:A, claim:B]. No scope difference was
  identified; an unstated premise may explain the divergence. Flagged
  for expert review."

- unresolved: A conflict that was detected but not yet disposed of.
  Present both claims, say plainly that they conflict and that the
  conflict is unresolved. Do not pick one. Example: "One source reports X
  [claim:A]; another reports Y [claim:B]. This conflict has not been
  resolved."

- supersede / retracted: Do not present the superseded or retracted claim
  as a live finding. If context requests include_retracted, you may
  mention it with an explicit annotation like "(retracted)".

If a claim or concept was retrieved but you do not cite it, that is fine
— not every retrieved item must be used. Some claims are included because
they conflict with, or state a condition of, a retrieved claim ("included
because: ..."); a condition tells you when the claim it scopes applies.

Format: plain markdown. No headings unless the answer genuinely has
sections. Start with a direct answer to the question."""


SYNTHESIZE_V2_USER_TEMPLATE = """Question: {query}

Context (filter constraints applied to retrieval):
{context_block}

Claims (ID, confidence, S — P — O, source span excerpt, source metadata):
{claims_block}

Concepts (ID, statement, inference_type, supporting claim IDs):
{concepts_block}

Dispositions (pre-computed from the retrieved claims):
{dispositions_block}

Write the answer now."""


# --- VERIFY (claim-level, unchanged from v1) ---

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


# --- VERIFY_CONCEPT (verbatim from 90-prompts.md) ---

VERIFY_CONCEPT_SYSTEM = """You are a verifier for concept citations. Given a concept's supporting
source spans (a set, not a single span) and a sentence that cites the
concept, determine whether the sentence is supported by the UNION of the
spans.

Rules:
- GROUNDED: every factual element of the sentence is supported by at
  least one of the spans.
- PARTIAL: some factual elements are supported but others are not.
- UNGROUNDED: the spans collectively do not support the sentence.

Do not judge whether the sentence is true in the world. Judge only whether
it follows from the spans taken together.

Reply in JSON:
{"verdict": "GROUNDED" | "PARTIAL" | "UNGROUNDED", "reason": "one short sentence"}"""


VERIFY_CONCEPT_USER_TEMPLATE = """Supporting spans (bracketed blocks):
{spans_block}

Sentence: {sentence}

Verdict (JSON)."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    sentence: str
    claim_ids: list[int]
    concept_ids: list[int] = field(default_factory=list)
    verdict: str = "GROUNDED"  # aggregate: GROUNDED | PARTIAL | UNGROUNDED | ERROR
    reason: str = ""           # aggregate reason (the worst ref's reason)
    # per-claim verdicts for multi-citation sentences. Empty when no cited claim
    # could be verified (e.g. all missing).
    per_claim: dict[int, tuple[str, str]] = field(default_factory=dict)
    # per-ref verdicts keyed by (kind, id). Superset of per_claim for backward compat.
    per_ref: dict[tuple[str, int], tuple[str, str]] = field(default_factory=dict)


@dataclass
class QueryResult:
    query: str
    answer: str  # annotated markdown with flags
    citations: list[Citation]
    claim_ids_used: list[int]
    concept_ids_used: list[int] = field(default_factory=list)
    from_cache: bool = False
    # Shown to the synthesizer but not cited ("evidence considered but not
    # used"). Recomputed on a cache hit, dropping claims no longer active.
    claim_ids_unused: list[int] = field(default_factory=list)
    # Uncited sides of conflicts involving cited claims (counter_evidence()).
    # Always computed at read time, so it reflects the current contradictions.
    counter_evidence: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Citation regex — now handles both [claim:ID] and [concept:ID]
# ---------------------------------------------------------------------------

CITE_RE = re.compile(r"\[(claim|concept):([0-9,\s]+)\]")


def _extract_keywords(query: str) -> list[str]:
    stop = {
        "the", "and", "for", "with", "what", "when", "where", "how", "why",
        "are", "is", "was", "were", "does", "do", "did", "a", "an", "of",
        "in", "on", "to", "from", "that", "this", "those", "these", "about",
        "between", "compare", "relate", "relates", "tell", "me",
    }
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]+", query.lower())
    return [w for w in words if w not in stop and len(w) > 2]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_claims_block_v2(
    claims_with_spans: list[tuple], store: Store,
    included_because: Optional[dict[int, str]] = None,
) -> str:
    """Format claims block for SYNTHESIZE_V2 user template. Each claim lists
    its conditions that are also in the block, and expanded claims say why
    they were included."""
    included_because = included_because or {}
    shown = {claim["id"] for claim, _ in claims_with_spans}
    lines = []
    for claim, span in claims_with_spans:
        # Get source metadata compact form
        meta = store.get_source_metadata(claim["source_id"])
        meta_compact = "none"
        if meta:
            meta_compact = json.dumps(
                {"domain": meta["domain"], **meta["metadata"]},
                default=str,
            )
        entry = (
            f"[claim:{claim['id']}] ({claim['confidence']:.2f}) "
            f"{claim['subject']} — {claim['predicate']} — {claim['object']}\n"
            f"    source span: {span[:300]!r}{'...' if len(span) > 300 else ''}\n"
            f"    source metadata: {meta_compact}"
        )
        conds = [
            f"[claim:{c['condition_claim_id']}] ({c['kind']}, "
            f"{'explicit' if c['explicit'] else 'inferred'})"
            for c in store.get_claim_conditions(claim["id"])
            if c["condition_claim_id"] in shown
        ]
        if conds:
            entry += "\n    conditions: " + ", ".join(conds)
        if claim["id"] in included_because:
            entry += f"\n    included because: {included_because[claim['id']]}"
        lines.append(entry)
    return "\n\n".join(lines) if lines else "none"


def _format_concepts_block(concept_rows: list) -> str:
    """Format concepts block for SYNTHESIZE_V2 user template."""
    if not concept_rows:
        return "none"
    lines = []
    for c in concept_rows:
        lines.append(
            f"[concept:{c['id']}] ({c['confidence']:.2f}, {c['inference_type']}) "
            f"{c['statement']}\n"
            f"    supporting claims: {c.get('_support_claim_ids_csv', 'unknown')}"
        )
    return "\n\n".join(lines) if lines else "none"


def _format_dispositions_block(groups: dict) -> str:
    """Format dispositions block for SYNTHESIZE_V2 user template."""
    lines = []
    for pair in groups.get("replications", []):
        ids_str = ",".join(str(i) for i in pair)
        lines.append(f"- replicate: [claim:{ids_str}]")
    for a, b, rule, cid in groups.get("reconciled", []):
        lines.append(
            f"- reconcile: [claim:{a}] vs [claim:{b}] — rule: {rule} "
            f"— rationale: [concept:{cid}]"
        )
    for a, b, rule, applies_when in groups.get("coexisting", []):
        lines.append(
            f"- coexist: [claim:{a}] vs [claim:{b}] — rule: {rule} "
            f"— applies_when: {json.dumps(applies_when, default=str)}"
        )
    for a, b, rule in groups.get("distinguished", []):
        lines.append(
            f"- distinguish: [claim:{a}] vs [claim:{b}] — rule: {rule}"
        )
    for a, b in groups.get("disputed", []):
        lines.append(f"- dispute: [claim:{a}] vs [claim:{b}]")
    for a, b in groups.get("gaps", []):
        lines.append(f"- gap: [claim:{a}] vs [claim:{b}]")
    for a, b in groups.get("unresolved", []):
        lines.append(f"- unresolved: [claim:{a}] vs [claim:{b}]")
    return "\n".join(lines) if lines else "none"


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


def _parse_citations(sentence: str) -> list[tuple[str, int]]:
    """Parse [claim:ID] and [concept:ID] tokens from a sentence.

    Returns list of (kind, id) pairs where kind is 'claim' or 'concept'.
    """
    refs: list[tuple[str, int]] = []
    for match in CITE_RE.finditer(sentence):
        kind = match.group(1)
        for part in match.group(2).split(","):
            part = part.strip()
            if part.isdigit():
                refs.append((kind, int(part)))
    return refs


def _parse_claim_ids(sentence: str) -> list[int]:
    """Backward-compat helper: extract just claim IDs from a sentence."""
    return [cid for kind, cid in _parse_citations(sentence) if kind == "claim"]


# ---------------------------------------------------------------------------
# Retrieval: concepts
# ---------------------------------------------------------------------------

def _retrieve_concepts(store: Store, keywords: list[str], limit: int = 10) -> list:
    """Concept retrieval: keyword match on subject OR statement, filter to
    status='active', ordered by confidence then last_validated_at."""
    if not keywords:
        return []
    kws = [k.lower() for k in keywords if len(k) > 2]
    if not kws:
        return []

    # Build LIKE clauses matching any keyword against subject or statement
    like_clauses = []
    params = []
    for k in kws:
        like_clauses.append(
            "(LOWER(subject) LIKE ? OR LOWER(statement) LIKE ?)"
        )
        params.extend([f"%{k}%", f"%{k}%"])

    where = " OR ".join(like_clauses)
    sql = (
        f"SELECT * FROM concepts WHERE status = 'active' AND ({where}) "
        f"ORDER BY confidence DESC, last_validated_at DESC LIMIT ?"
    )
    params.append(limit)
    rows = store.conn.execute(sql, params).fetchall()

    # Enrich each row with supporting claim IDs for the template
    enriched = []
    for row in rows:
        # Convert sqlite3.Row to a mutable dict
        d = dict(row)
        supports = store.get_concept_supports(row["id"])
        d["_support_claim_ids_csv"] = ",".join(
            str(s["claim_id"]) for s in supports
        )
        # Wrap in a simple namespace so d["key"] works
        enriched.append(_DictRow(d))
    return enriched


class _DictRow:
    """Minimal dict-like wrapper so enriched concept rows work with [] access."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _search(retr, keywords: list[str], limit: int,
            context: Optional[dict]) -> list:
    """Run the retriever, asking for retracted claims too when the context
    sets ``include_retracted`` (otherwise they never reach the filter)."""
    if context and context.get("include_retracted"):
        return retr.search(keywords, limit=limit, include_retracted=True)
    return retr.search(keywords, limit=limit)


def _filter_claims_by_context(
    store: Store, claim_rows: list, context: Optional[dict],
) -> list:
    """Drop retracted (unless context.include_retracted=True) and apply
    jurisdiction/date/domain filters. Log drop counts."""
    from . import authority

    if context is None:
        context = {}

    include_retracted = context.get("include_retracted", False)
    ctx_jurisdiction = context.get("jurisdiction")
    ctx_date = context.get("date")
    ctx_domain = context.get("domain")

    kept = []
    drop_count = 0

    for row in claim_rows:
        meta = store.get_source_metadata(row["source_id"])

        # --- retracted filter ---
        if meta and authority.is_retracted(meta) and not include_retracted:
            log("context_filter_drop", level="info",
                claim_id=row["id"], reason="retracted")
            drop_count += 1
            continue

        # No metadata → no further filtering; keep
        if meta is None:
            kept.append(row)
            continue

        # --- jurisdiction filter ---
        if ctx_jurisdiction:
            source_jurisdiction = meta.get("metadata", {}).get("jurisdiction")
            if source_jurisdiction is not None:
                # Keep if exact match or prefix match (e.g. US-CA keeps US-CA-LA)
                if not (
                    source_jurisdiction == ctx_jurisdiction
                    or source_jurisdiction.startswith(ctx_jurisdiction + "-")
                ):
                    log("context_filter_drop", level="info",
                        claim_id=row["id"], reason="jurisdiction_mismatch")
                    drop_count += 1
                    continue

        # --- date / effective_at filter ---
        if ctx_date is not None:
            if not authority.is_effective_at(meta, ctx_date):
                log("context_filter_drop", level="info",
                    claim_id=row["id"], reason="not_effective_at_date")
                drop_count += 1
                continue

        # --- domain filter ---
        if ctx_domain:
            if meta["domain"] != ctx_domain:
                log("context_filter_drop", level="info",
                    claim_id=row["id"], reason="domain_mismatch")
                drop_count += 1
                continue

        kept.append(row)

    if drop_count:
        log("context_filter_summary", level="info",
            total=len(claim_rows), kept=len(kept), dropped=drop_count)

    return kept


# ---------------------------------------------------------------------------
# Disposition-aware grouping
# ---------------------------------------------------------------------------

def _claims_with_spans(store: Store, ids: list[int], include_retracted: bool) -> list:
    """Claim rows shaped like a retriever's (with ``span_text``), by id."""
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    rows = store.conn.execute(
        "SELECT c.*, SUBSTR(s.content, c.span_start + 1, c.span_end - c.span_start) "
        "AS span_text FROM claims c JOIN sources s ON c.source_id = s.id "
        f"WHERE c.id IN ({marks}) AND c.status IN ({_statuses_sql(include_retracted)})",
        ids,
    ).fetchall()
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in ids if i in by_id]


# How an expanded claim relates to the retrieved one, by disposition, so a
# replication isn't pitched to the synthesizer as a conflict.
_RELATION_BY_DISPOSITION = {
    "replicate": "replicates",
    "coexist": "coexists with",
    "distinguish": "is distinguished from",
    "reconcile": "is reconciled with",
    "dispute": "disputes",
    "gap": "has an unexplained conflict with",
}


def _expand_evidence(
    store: Store, claim_rows: list, context: Optional[dict], limit: int,
) -> tuple[list, dict[int, str]]:
    """Add claims that conflict with, or are conditions of, retrieved claims.

    Retrieval matches words; the claim that contradicts or scopes a
    retrieved claim may share none of them. Added claims go through the
    same context filter, at most ``limit`` of them, in retrieval order.
    Returns (rows, {added claim id: why it was included}).
    """
    have = {r["id"] for r in claim_rows}
    candidates: dict[int, str] = {}
    for row in claim_rows:
        cid = row["id"]
        for x in store.conn.execute(
            "SELECT claim_a_id, claim_b_id, status, disposition FROM contradictions x "
            f"WHERE (claim_a_id = ? OR claim_b_id = ?) AND NOT {_KEEP_RESOLVED_SQL} "
            "ORDER BY id", (cid, cid),
        ):
            other = x["claim_b_id"] if x["claim_a_id"] == cid else x["claim_a_id"]
            if other in have or _senses_differ(store, cid, other):
                continue
            relation = "conflicts with" if x["status"] == "open" else \
                _RELATION_BY_DISPOSITION.get(x["disposition"], "conflicts with")
            candidates.setdefault(other, f"{relation} [claim:{cid}]")
        for cond in store.get_claim_conditions(cid):
            other = cond["condition_claim_id"]
            if other not in have:
                candidates.setdefault(other, f"{cond['kind']} condition of [claim:{cid}]")
    if not candidates or limit <= 0:
        return claim_rows, {}
    include_retracted = bool((context or {}).get("include_retracted"))
    extra = _claims_with_spans(store, list(candidates), include_retracted)
    extra = _filter_claims_by_context(store, extra, context)[:limit]
    return list(claim_rows) + extra, {r["id"]: candidates[r["id"]] for r in extra}


def _group_by_disposition(store: Store, claim_rows: list) -> dict:
    """Lookup contradictions where either claim is in the retrieved set and
    build groups by disposition type. Unresolved pairs are grouped only when
    both claims are in the set.

    WS-E.2: when both claims of a contradiction have predicate-sense
    assignments and the senses differ, skip the pair — different relations
    are not the same kind of contradiction even when the surface predicate
    matches.
    """
    claim_id_set = {row["id"] for row in claim_rows}
    if not claim_id_set:
        return {
            "replications": [], "reconciled": [], "coexisting": [],
            "distinguished": [], "disputed": [], "gaps": [], "unresolved": [],
        }

    # Every contradiction touching the set (not a store-wide top-N, which
    # would drop old conflicts of the claims actually shown).
    ids = sorted(claim_id_set)
    marks = ",".join("?" * len(ids))
    all_contradictions = store.conn.execute(
        "SELECT ct.*, cr.rule, cr.applies_when, cr.rationale_concept_id "
        "FROM contradictions ct "
        "LEFT JOIN contradiction_rules cr ON ct.id = cr.contradiction_id "
        f"WHERE ct.claim_a_id IN ({marks}) OR ct.claim_b_id IN ({marks}) "
        "ORDER BY ct.id",
        (*ids, *ids),
    ).fetchall()

    groups: dict = {
        "replications": [],
        "reconciled": [],
        "coexisting": [],
        "distinguished": [],
        "disputed": [],
        "gaps": [],
        "unresolved": [],
    }

    for ct in all_contradictions:
        a_id = ct["claim_a_id"]
        b_id = ct["claim_b_id"]
        disposition = ct["disposition"]

        # Skip if neither claim is in the retrieved set
        if a_id not in claim_id_set and b_id not in claim_id_set:
            continue
        # WS-E.2: if both claims have differing predicate senses, treat them
        # as different relations and skip the disposition group.
        a_sense = store.conn.execute(
            "SELECT sense_id FROM claim_predicate_senses WHERE claim_id = ?",
            (a_id,),
        ).fetchone()
        b_sense = store.conn.execute(
            "SELECT sense_id FROM claim_predicate_senses WHERE claim_id = ?",
            (b_id,),
        ).fetchone()
        if (
            a_sense is not None and b_sense is not None
            and a_sense["sense_id"] != b_sense["sense_id"]
        ):
            continue

        # Unresolved means open (never disposed, or reopened). A legacy
        # `contradiction-resolve --keep` leaves disposition at its default but
        # status resolved: settled, so not shown. Unresolved pairs are shown
        # only when both sides are in the set, so both can be presented.
        if ct["status"] == "open":
            if a_id in claim_id_set and b_id in claim_id_set:
                groups["unresolved"].append((a_id, b_id))
            continue
        if disposition is None or disposition == "unresolved":
            continue

        rule = ct["rule"] or ""
        applies_when_raw = ct["applies_when"]
        rationale_concept_id = ct["rationale_concept_id"]

        applies_when = None
        if applies_when_raw:
            try:
                applies_when = json.loads(applies_when_raw) if isinstance(applies_when_raw, str) else applies_when_raw
            except (json.JSONDecodeError, TypeError):
                applies_when = None

        if disposition == "replicate":
            # Check if either id is already in a replication group
            found = False
            for group in groups["replications"]:
                if a_id in group or b_id in group:
                    group.add(a_id)
                    group.add(b_id)
                    found = True
                    break
            if not found:
                groups["replications"].append({a_id, b_id})
        elif disposition == "reconcile":
            groups["reconciled"].append(
                (a_id, b_id, rule, rationale_concept_id)
            )
        elif disposition == "coexist":
            groups["coexisting"].append(
                (a_id, b_id, rule, applies_when or {})
            )
        elif disposition == "distinguish":
            groups["distinguished"].append((a_id, b_id, rule))
        elif disposition == "dispute":
            groups["disputed"].append((a_id, b_id))
        elif disposition == "gap":
            groups["gaps"].append((a_id, b_id))
        # supersede / retracted: already filtered out by _filter_claims_by_context

    # Convert replication sets to sorted lists
    groups["replications"] = [sorted(s) for s in groups["replications"]]

    return groups


# ---------------------------------------------------------------------------
# Cache helpers (bypass Store.get_cached_view / cache_view for new hash)
# ---------------------------------------------------------------------------

def _compute_cache_hash(question: str, context: Optional[dict]) -> str:
    return view_cache_key(question, context)


def _get_cached_view(store: Store, query_hash: str):
    """Look up view_cache by precomputed hash."""
    return store.conn.execute(
        "SELECT * FROM view_cache WHERE query_hash = ?", (query_hash,)
    ).fetchone()


def _cache_view(
    store: Store, query_hash: str, question: str, response: str,
    claim_ids: list[int], concept_ids: list[int],
    considered_claim_ids: Optional[list[int]] = None,
) -> None:
    """Write to view_cache with the new hash, including concept_ids.

    ``considered_claim_ids`` (everything shown to the synthesizer) is not an
    invalidation index: the prose doesn't depend on uncited claims, so it is
    filtered to active claims when read instead."""
    with store.tx():
        store.conn.execute(
            "INSERT OR REPLACE INTO view_cache "
            "(query_hash, query, response, claim_ids, concept_ids, generated_at, "
            " considered_claim_ids) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query_hash, question, response,
             json.dumps(claim_ids), json.dumps(concept_ids), time.time(),
             json.dumps(sorted(considered_claim_ids or []))),
        )


# A conflict is live while it is open (detected, never disposed, or reopened
# by the cascade) or disposed as dispute / gap. coexist / distinguish /
# reconcile carry a rule saying when each side applies; replicate is
# agreement; supersede / retracted and the legacy `contradiction-resolve
# --keep` (status resolved, disposition left at its default) settle it.
_COUNTER_DISPOSITIONS = ("dispute", "gap")
_LIVE_CONFLICT_SQL = (
    "(x.status = 'open' OR x.disposition IN ("
    + ", ".join(f"'{d}'" for d in _COUNTER_DISPOSITIONS) + "))"
)
# Settled with one side kept: the other side is no longer part of the picture.
_KEEP_RESOLVED_SQL = (
    "(x.status = 'resolved' AND COALESCE(x.disposition, 'unresolved') "
    "IN ('unresolved', 'supersede', 'retracted'))"
)


def _senses_differ(store: Store, a_id: int, b_id: int) -> bool:
    """WS-E.2: both claims have predicate senses and they differ, so the
    pair is two different relations, not a conflict."""
    rows = store.conn.execute(
        "SELECT claim_id, sense_id FROM claim_predicate_senses WHERE claim_id IN (?, ?)",
        (a_id, b_id),
    ).fetchall()
    senses = {r["claim_id"]: r["sense_id"] for r in rows}
    return len(senses) == 2 and senses[a_id] != senses[b_id]


def counter_evidence(
    store: Store, cited_claim_ids, context: Optional[dict] = None,
) -> list[dict]:
    """For each cited claim, the active, uncited claims it is in a live
    conflict with (an open contradiction, or one disposed as dispute or
    gap). An answer that cites one side of such a conflict and not the
    other is presenting a contested fact as settled. Counter claims go
    through the same context filter as retrieval, and pairs with differing
    predicate senses (WS-E.2) are not conflicts."""
    cited = set(cited_claim_ids)
    if not cited:
        return []
    marks = ",".join("?" * len(cited))
    rows = store.conn.execute(
        f"""
        SELECT x.id, x.claim_a_id, x.claim_b_id,
               CASE WHEN x.status = 'open' THEN 'unresolved' ELSE x.disposition END
                   AS disposition
        FROM contradictions x
        JOIN claims a ON a.id = x.claim_a_id
        JOIN claims b ON b.id = x.claim_b_id
        WHERE (x.claim_a_id IN ({marks}) OR x.claim_b_id IN ({marks}))
          AND a.status = 'active' AND b.status = 'active'
          AND {_LIVE_CONFLICT_SQL}
        ORDER BY x.id
        """,
        (*cited, *cited),
    ).fetchall()
    candidates = []
    for r in rows:
        if _senses_differ(store, r["claim_a_id"], r["claim_b_id"]):
            continue
        for mine, other in ((r["claim_a_id"], r["claim_b_id"]),
                            (r["claim_b_id"], r["claim_a_id"])):
            if mine in cited and other not in cited:
                candidates.append({"cited_claim_id": mine, "counter_claim_id": other,
                                   "contradiction_id": r["id"],
                                   "disposition": r["disposition"]})
    if not candidates or not context:
        return candidates
    allowed = {row["id"] for row in _filter_claims_by_context(
        store, _claims_with_spans(store, sorted({c["counter_claim_id"] for c in candidates}),
                                  bool(context.get("include_retracted"))),
        context)}
    return [c for c in candidates if c["counter_claim_id"] in allowed]


def _claim_ids_with_status(store: Store, ids, include_retracted: bool = False) -> set[int]:
    """The ids still retrievable: active, or retracted too when the context
    asks for retracted claims (as retrieval does)."""
    ids = list(ids)
    if not ids:
        return set()
    marks = ",".join("?" * len(ids))
    return {r[0] for r in store.conn.execute(
        f"SELECT id FROM claims WHERE status IN ({_statuses_sql(include_retracted)}) "
        f"AND id IN ({marks})", ids)}


# ---------------------------------------------------------------------------
# Main query pipeline
# ---------------------------------------------------------------------------

def query(
    store: Store,
    llm: LLM,
    question: str,
    retrieve_k: int = 30,
    verify: bool = True,
    use_cache: bool = True,
    context: Optional[dict] = None,
    retriever: Retriever | None = None,
    concept_k: Optional[int] = None,
    verifier=None,
    expand: bool = True,
) -> QueryResult:
    """Run the full question -> verified answer pipeline.

    `retriever` defaults to FTS5 if the store supports it, else keyword
    substring search. Pass an explicit Retriever to plug in a different
    backend (e.g. embeddings) without touching this function.

    `context` is a dict with optional keys: jurisdiction, date, domain,
    include_retracted.  When None, only retraction filtering applies.

    `concept_k` controls how many concepts to retrieve (default max(5, retrieve_k // 3)).

    `verifier` swaps the claim-citation check for a
    :class:`~aleph.verifier.LayeredVerifier` (opt-in until the verifier eval
    shows it beats the default). Its verdicts map SUPPORTED -> GROUNDED,
    UNSUPPORTED -> UNGROUNDED, UNCERTAIN -> UNCERTAIN; reasons are prefixed
    with the deciding layer. Concept citations keep the union-of-spans check.
    With a `verifier`, `expand=False` or `verify=False`, the view cache is
    bypassed (neither read nor written).

    `expand` (default on) adds claims that conflict with, or are conditions
    of, retrieved claims — up to ``retrieve_k // 2`` of them.
    """
    # --- cache check ---
    # Views are cached under ask's defaults (verified by the LLM verifier,
    # expansion on). A
    # view built any other way is neither served from nor written to the
    # cache (see step 8).
    default_view = verifier is None and expand and verify
    if not default_view:
        use_cache = False
    cache_hash = _compute_cache_hash(question, context)
    if use_cache:
        cached = _get_cached_view(store, cache_hash)
        if cached:
            concept_ids_cached = []
            try:
                concept_ids_cached = json.loads(cached["concept_ids"])
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
            used = json.loads(cached["claim_ids"])
            considered = json.loads(cached["considered_claim_ids"] or "[]")
            return QueryResult(
                query=question,
                answer=cached["response"],
                citations=[],
                claim_ids_used=used,
                concept_ids_used=concept_ids_cached,
                from_cache=True,
                claim_ids_unused=sorted(
                    _claim_ids_with_status(
                        store, considered, bool((context or {}).get("include_retracted")))
                    - set(used)),
                counter_evidence=counter_evidence(store, used, context),
            )

    # 1. retrieve candidate claims
    keywords = _extract_keywords(question)
    retr = retriever if retriever is not None else default_retriever(store)
    claim_rows = _search(retr, keywords, retrieve_k, context)
    if not claim_rows:
        return QueryResult(
            query=question,
            answer=(
                "_No relevant claims in the knowledge base. Ingest more sources, "
                "or check whether this topic is covered by what you've ingested._"
            ),
            citations=[],
            claim_ids_used=[],
            concept_ids_used=[],
            from_cache=False,
        )

    # 2. filter claims by context (retracted, jurisdiction, date, domain)
    claim_rows = _filter_claims_by_context(store, claim_rows, context)
    if not claim_rows:
        return QueryResult(
            query=question,
            answer=(
                "_All retrieved claims were filtered out by context constraints. "
                "Try broadening the context or ingesting more sources._"
            ),
            citations=[],
            claim_ids_used=[],
            concept_ids_used=[],
            from_cache=False,
        )

    # 2b. expand to conflicting and condition claims
    included_because: dict[int, str] = {}
    if expand:
        claim_rows, included_because = _expand_evidence(
            store, claim_rows, context, retrieve_k // 2)

    # 3. retrieve concepts
    if concept_k is None:
        concept_k = max(5, retrieve_k // 3)
    concept_rows = _retrieve_concepts(store, keywords, limit=concept_k)

    # 4. disposition-aware grouping
    disposition_groups = _group_by_disposition(store, claim_rows)

    # span_text is joined inline by the retriever — no per-claim round-trip
    claims_with_spans = [(row, row["span_text"] or "") for row in claim_rows]

    # 5. synthesize view (always v2)
    claims_block = _format_claims_block_v2(claims_with_spans, store, included_because)
    concepts_block = _format_concepts_block(concept_rows)
    dispositions_block = _format_dispositions_block(disposition_groups)
    context_block = json.dumps(context, default=str) if context else "none"

    answer = llm.complete(
        SYNTHESIZE_V2_SYSTEM,
        SYNTHESIZE_V2_USER_TEMPLATE.format(
            query=question,
            context_block=context_block,
            claims_block=claims_block,
            concepts_block=concepts_block,
            dispositions_block=dispositions_block,
        ),
        max_tokens=2048,
    )

    # split once: both verification and cache-key computation walk the sentences
    sentences = _split_sentences(answer)
    retrieved_ids = {r["id"] for r in claim_rows}
    retrieved_concept_ids = {c["id"] for c in concept_rows}

    # 6. verifier pass: for each cited ref in each sentence, check grounding
    citations: list[Citation] = []
    if verify:
        for sent in sentences:
            cited_refs = _parse_citations(sent)
            if not cited_refs:
                continue
            per_ref: dict[tuple[str, int], tuple[str, str]] = {}
            per_claim: dict[int, tuple[str, str]] = {}
            sent_claim_ids = []
            sent_concept_ids = []
            for kind, cid in cited_refs:
                if kind == "claim":
                    if verifier is not None:
                        others = [c for k, c in cited_refs if k == "claim" and c != cid]
                        verdict_reason = _verify_layered(verifier, store, sent, cid, others)
                    else:
                        verdict_reason = _verify_one(llm, store, sent, cid)
                    per_ref[(kind, cid)] = verdict_reason
                    per_claim[cid] = verdict_reason
                    sent_claim_ids.append(cid)
                elif kind == "concept":
                    verdict_reason = _verify_concept_citation(
                        llm, store, sent, cid
                    )
                    per_ref[(kind, cid)] = verdict_reason
                    sent_concept_ids.append(cid)

            all_verdicts = [v for v, _ in per_ref.values()]
            agg_verdict = _aggregate_verdict(all_verdicts)
            agg_reason = next(
                (r for v, r in per_ref.values() if v == agg_verdict), ""
            )
            citations.append(Citation(
                sentence=sent,
                claim_ids=sent_claim_ids,
                concept_ids=sent_concept_ids,
                verdict=agg_verdict,
                reason=agg_reason,
                per_claim=per_claim,
                per_ref=per_ref,
            ))

    # 7. annotate answer with verdicts
    annotated = _annotate_answer(answer, citations) if verify else answer

    # cache-key claims = the ones the LLM actually cited (intersected with the
    # retrieved set, so hallucinated IDs don't end up in the index). This keeps
    # invalidation precise.
    cited_claim_ids: set[int] = set()
    cited_concept_ids: set[int] = set()
    for sent in sentences:
        for kind, cid in _parse_citations(sent):
            if kind == "claim" and cid in retrieved_ids:
                cited_claim_ids.add(cid)
            elif kind == "concept" and cid in retrieved_concept_ids:
                cited_concept_ids.add(cid)

    claim_ids_used = sorted(cited_claim_ids)
    concept_ids_used = sorted(cited_concept_ids)

    # 8. cache (refreshed even under use_cache=False; only a non-default
    # view stays out of it)
    if default_view:
        _cache_view(
            store, cache_hash, question, annotated,
            claim_ids_used, concept_ids_used, sorted(retrieved_ids),
        )

    return QueryResult(
        query=question,
        answer=annotated,
        citations=citations,
        claim_ids_used=claim_ids_used,
        concept_ids_used=concept_ids_used,
        from_cache=False,
        claim_ids_unused=sorted(retrieved_ids - cited_claim_ids),
        counter_evidence=counter_evidence(store, claim_ids_used, context),
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_one(llm: LLM, store: Store, sentence: str, claim_id: int) -> tuple[str, str]:
    """Verify a sentence against a single cited claim's span.

    Returns (verdict, reason). Verdict is one of GROUNDED | PARTIAL | UNGROUNDED
    | ERROR. ERROR means the verifier itself couldn't decide (parse/API failure),
    distinct from UNGROUNDED (the verifier decided the span does not support it).
    """
    span = store.get_span_text(claim_id)
    if span is None:
        return "UNGROUNDED", "cited claim not found"
    return _verify_span(llm, span, sentence, claim_id=claim_id)


def _verify_layered(
    verifier, store: Store, sentence: str, claim_id: int, co_cited: list[int] = (),
) -> tuple[str, str]:
    """Verify with a LayeredVerifier, against the span and its context window.
    A sentence citing several claims ("A is 5 and B is 7 [claim:1][claim:2]")
    is checked against the other cited spans too, so a fact from claim 2
    doesn't make claim 1's check reject the sentence."""
    span = store.get_span_text(claim_id)
    if span is None:
        return "UNGROUNDED", "cited claim not found"
    context = [store.get_context_text(claim_id) or ""]
    context += [store.get_span_text(c) or "" for c in co_cited]
    r = verifier.verify(sentence, span, "\n".join(t for t in context if t))
    return _LAYERED_TO_ASK[r.verdict], f"[{r.layer}] {r.reason}"


def _verify_span(
    llm: LLM, span: str, sentence: str, *, claim_id: Optional[int] = None,
) -> tuple[str, str]:
    """The LLM span check behind :func:`_verify_one`, on a raw span. Also the
    baseline the layered verifier is measured against."""
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


def _verify_concept_citation(
    llm: LLM, store: Store, sentence: str, concept_id: int,
) -> tuple[str, str]:
    """Verify a sentence against a concept's supporting claim spans (union).

    Fetches all support claims, collects their spans, and asks the LLM
    whether the sentence is supported by the UNION of those spans.
    Returns (verdict, reason).
    """
    supports = store.get_concept_supports(concept_id)
    if not supports:
        return "UNGROUNDED", "concept has no supporting claims"

    spans_parts = []
    for sup in supports:
        span = store.get_span_text(sup["claim_id"])
        if span:
            spans_parts.append(
                f"--- span (claim_id={sup['claim_id']}) ---\n{span}"
            )

    if not spans_parts:
        return "UNGROUNDED", "no spans found for concept supports"

    spans_block = "\n".join(spans_parts)

    try:
        result = llm.complete_json(
            VERIFY_CONCEPT_SYSTEM,
            VERIFY_CONCEPT_USER_TEMPLATE.format(
                spans_block=spans_block, sentence=sentence,
            ),
            max_tokens=256,
        )
        verdict = str(result.get("verdict", "UNGROUNDED")).upper()
        if verdict not in {"GROUNDED", "PARTIAL", "UNGROUNDED"}:
            verdict = "UNGROUNDED"
        reason = str(result.get("reason", ""))
        return verdict, reason
    except Exception as e:
        log(
            "concept_verifier_error",
            level="warning",
            concept_id=concept_id,
            error=type(e).__name__,
            message=str(e),
        )
        return "ERROR", f"concept verifier error: {e}"


def _aggregate_verdict(verdicts) -> str:
    """Worst verdict wins. Empty iterator aggregates to GROUNDED (caller
    already guarantees there's at least one cited ref when used)."""
    worst = "GROUNDED"
    for v in verdicts:
        if _VERDICT_RANK.get(v, 2) > _VERDICT_RANK.get(worst, 0):
            worst = v
    return worst


def _annotate_answer(answer: str, citations: list[Citation]) -> str:
    """Append a warning footer listing every non-GROUNDED per-ref verdict.

    For multi-citation sentences, each failing ref gets its own line so the
    caller can see which specific claim(s)/concept(s) the verifier rejected.
    """
    flagged = [c for c in citations if c.verdict != "GROUNDED"]
    if not flagged:
        return answer
    lines = ["\n\n---", "**Verifier flags:**"]
    for c in flagged:
        # prefer per-ref detail when available
        if c.per_ref:
            for (kind, cid), (v, r) in c.per_ref.items():
                if v == "GROUNDED":
                    continue
                lines.append(f"- [{v}] `[{kind}:{cid}]` — {r}")
        elif c.per_claim:
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
