"""Source metadata: typed per-domain JSON blobs describing authority,
provenance, and status of each source.

Domains: legal, scientific, policy, corporate, generic.

The metadata is INPUT the user/agent provides. Aleph never fabricates
jurisdiction, authority level, citation counts, or peer-review status.
Helpers here only validate and rank, never invent.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from .db import Store, _invalidate_cache_for_claims, _revive_orphaned_supersessions_tx
from .log import log


# ----- domain schemas (documented; validated here) -----

LEGAL_FIELDS: dict[str, Any] = {
    "jurisdiction": str,         # e.g. "US-federal", "US-CA", "EU", "IT"
    "authority_type": str,       # constitution | statute | regulation | case | guidance
    "authority_level": int,      # higher wins; domain-defined scale
    "specificity": int,          # higher wins within a level (lex specialis)
    "issued_at": (int, float),   # unix seconds
    "effective_at": (int, float),
    "expires_at": (int, float),
    "superseded_by_source_id": int,
}

SCIENTIFIC_FIELDS: dict[str, Any] = {
    "authors": list,
    "venue": str,
    "doi": str,
    "published_at": (int, float),
    "peer_reviewed": bool,
    "retracted": bool,
    "retracted_at": (int, float),
    "retraction_reason": str,
    "replication_status": str,   # untested | replicated | failed-replication | disputed
    "citation_count": int,
}

POLICY_FIELDS: dict[str, Any] = {
    "org": str,
    "version": str,
    "effective_at": (int, float),
    "expires_at": (int, float),
    "supersedes_source_id": int,
}

CORPORATE_FIELDS: dict[str, Any] = {
    "team": str,
    "version": str,
    "doc_type": str,            # PRD | spec | runbook | decision-log | ...
    "owner": str,
    "reviewed_at": (int, float),
}

GENERIC_FIELDS: dict[str, Any] = {}  # accepts any keys

# Common provenance fields accepted on any domain. `fetch_method` records how
# the source content was obtained (the authoritative original, a distilled
# rendering, an OCR pass, etc.). `provenance_notes` is a free-text audit trail.
# Both are optional; their presence lets downstream verifiers flag citations
# whose source is not a first-party original.
FETCH_METHODS: set[str] = {"direct", "distilled", "ocr", "transcription", "pasted"}

COMMON_FIELDS: dict[str, Any] = {
    "fetch_method": str,
    "provenance_notes": str,
}

DOMAIN_SCHEMAS: dict[str, dict[str, Any]] = {
    "legal": LEGAL_FIELDS,
    "scientific": SCIENTIFIC_FIELDS,
    "policy": POLICY_FIELDS,
    "corporate": CORPORATE_FIELDS,
    "generic": GENERIC_FIELDS,
}


# ----- validation -----

@dataclass
class ValidationError:
    field: str
    reason: str


def _check_type(expected, value) -> Optional[str]:
    """Return an error reason if ``value`` does not match ``expected``, else
    None. ``expected`` may be a type or a tuple of types."""
    if isinstance(expected, tuple):
        if not isinstance(value, expected):
            names = " or ".join(t.__name__ for t in expected)
            return f"expected {names}, got {type(value).__name__}"
    else:
        if not isinstance(value, expected):
            return f"expected {expected.__name__}, got {type(value).__name__}"
    return None


def validate_metadata(domain: str, metadata: dict) -> list[ValidationError]:
    """Check each provided field against the schema for domain. Unknown
    fields are allowed (forward-compatible) but logged. Missing fields are
    allowed (all optional). Wrong-type fields are errors. Returns a list of
    errors; empty = valid.

    Common provenance fields (``fetch_method``, ``provenance_notes``) are
    accepted on every domain and validated uniformly: ``fetch_method`` must
    be one of :data:`FETCH_METHODS`.
    """
    if domain not in DOMAIN_SCHEMAS:
        return [ValidationError(field="domain", reason=f"unknown domain {domain!r}")]

    schema = DOMAIN_SCHEMAS[domain]
    errors: list[ValidationError] = []

    for key, value in metadata.items():
        # Common provenance fields: accepted on any domain.
        if key in COMMON_FIELDS:
            reason = _check_type(COMMON_FIELDS[key], value)
            if reason:
                errors.append(ValidationError(field=key, reason=reason))
                continue
            if key == "fetch_method" and value not in FETCH_METHODS:
                errors.append(ValidationError(
                    field="fetch_method",
                    reason=f"expected one of {sorted(FETCH_METHODS)}, got {value!r}",
                ))
            continue
        if key not in schema:
            # unknown field: log warning but allow
            if schema:  # don't warn for generic (empty schema = anything goes)
                log("unknown_metadata_field", level="warning",
                    domain=domain, field=key)
            continue
        reason = _check_type(schema[key], value)
        if reason:
            errors.append(ValidationError(field=key, reason=reason))
    return errors


def provenance_warnings(metadata: dict) -> list[str]:
    """Non-fatal advisories about provenance completeness.

    A source retrieved via OCR, distillation, or transcription is
    epistemically weaker than the original — flag any non-direct
    ``fetch_method`` that lacks accompanying ``provenance_notes`` so a
    downstream verifier can surface the gap.
    """
    warnings: list[str] = []
    fm = metadata.get("fetch_method")
    if fm is not None and fm != "direct" and not metadata.get("provenance_notes"):
        warnings.append(
            f"fetch_method={fm!r} but provenance_notes is empty; "
            "consider recording how the rendering was produced"
        )
    return warnings


# ----- ranking -----

def authority_rank(meta: dict) -> tuple:
    """Return a tuple suitable for sorting (higher ranks first): for legal
    domain, use (authority_level, specificity, issued_at). For scientific,
    use (peer_reviewed, citation_count, published_at). Missing fields sort
    as 'unknown' (lowest). For unknown domains, returns (0,)."""
    # Accept both the full row-style dict and just the inner metadata dict.
    domain = meta.get("domain")
    inner = meta.get("metadata", meta)

    if domain == "legal":
        return (
            inner.get("authority_level") or 0,
            inner.get("specificity") or 0,
            inner.get("issued_at") or 0,
        )
    if domain == "scientific":
        return (
            int(inner.get("peer_reviewed") or False),
            inner.get("citation_count") or 0,
            inner.get("published_at") or 0,
        )
    return (0,)


# When each domain says a source was issued, in priority order.
_DATE_FIELDS = {
    "legal": ("issued_at", "effective_at"),
    "scientific": ("published_at",),
    "policy": ("effective_at",),
    "corporate": ("reviewed_at",),
    "generic": ("published_at", "issued_at", "effective_at"),
}


def source_date(meta: Optional[dict]) -> Optional[float]:
    """The source's own date (unix seconds) from its metadata, or None.
    This is when the source was issued or published, never when it was
    ingested or its claims extracted."""
    if not meta:
        return None
    inner = meta.get("metadata", meta)
    for field in _DATE_FIELDS.get(meta.get("domain"), _DATE_FIELDS["generic"]):
        value = inner.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def is_retracted(meta: dict) -> bool:
    """Scientific-domain only: metadata.retracted is True. Other domains:
    always False (retraction is not a universal concept)."""
    # Accept either the full row-style {"domain": ..., "metadata": {...}} dict
    # OR just the inner metadata dict.
    domain = meta.get("domain")
    inner = meta.get("metadata", meta)

    if domain is not None:
        # Full row-style dict
        if domain != "scientific":
            return False
        return inner.get("retracted") is True

    # Just the inner dict — no domain info, default False
    return False


def is_effective_at(meta: dict, when: float) -> bool:
    """For domains with effective_at/expires_at, check whether `when` is
    within the window. If the fields are missing, default to True (we don't
    assume expiration)."""
    inner = meta.get("metadata", meta)

    effective = inner.get("effective_at")
    expires = inner.get("expires_at")

    if effective is not None and when < effective:
        return False
    if expires is not None and when > expires:
        return False
    return True


# ----- persistence helpers -----

_RETRACTION_FIELDS = ("retracted", "retracted_at", "retraction_reason")


def set_metadata(
    store: Store, source_id: int, domain: str, metadata: dict,
) -> dict:
    """Validate then persist via store.set_source_metadata. Returns a dict
    with {'source_id', 'domain', 'metadata', 'validation_errors': [...],
    'warnings': [...]}. If domain is unknown, returns validation error for
    domain without writing."""
    errors = validate_metadata(domain, metadata)

    # Retraction state changes only through retract_source / unretract_source,
    # which run the claim cascade. Omitting the retraction fields on a
    # retracted source carries them over; contradicting them is refused.
    existing = store.get_source_metadata(source_id)
    was_retracted = existing is not None and is_retracted(existing)
    if was_retracted and "retracted" not in metadata and domain == "scientific":
        metadata = dict(metadata)
        for key in _RETRACTION_FIELDS:
            if key in existing["metadata"] and key not in metadata:
                metadata[key] = existing["metadata"][key]
    if is_retracted({"domain": domain, "metadata": metadata}) != was_retracted:
        return {
            "source_id": source_id,
            "error": "retraction_state_change",
            "message": (
                "use source-retract / source-unretract to change whether a "
                "source is retracted"
            ),
        }

    # If the domain itself is unknown, refuse to write (Store would raise too)
    domain_error = any(e.field == "domain" for e in errors)
    if not domain_error:
        store.set_source_metadata(source_id, domain, metadata)

    return {
        "source_id": source_id,
        "domain": domain,
        "metadata": metadata,
        "validation_errors": [{"field": e.field, "reason": e.reason} for e in errors],
        "warnings": provenance_warnings(metadata),
    }


def retract_source(
    store: Store, source_id: int, reason: str, *, retracted_at: Optional[float] = None,
) -> dict:
    """Atomic retraction. Steps:
      1. Load existing metadata; domain must be 'scientific' (other domains
         use supersede, not retract).
      2. Update metadata with retracted=True, retracted_at, retraction_reason.
      3. Call store.retract_source(source_id) to cascade claim/concept
         status changes (installed by Phase 0).
    Returns a dict with the before/after state."""
    existing = store.get_source_metadata(source_id)
    if existing is None:
        return {"error": "no_metadata"}
    if existing["domain"] != "scientific":
        return {"error": "not_scientific_domain"}

    before = {
        "domain": existing["domain"],
        "metadata": dict(existing["metadata"]),
    }

    updated_meta = dict(existing["metadata"])
    updated_meta["retracted"] = True
    updated_meta["retracted_at"] = retracted_at or time.time()
    updated_meta["retraction_reason"] = reason

    # Metadata and claim cascade commit together, or not at all.
    with store.tx() as cx:
        cx.execute(
            "INSERT OR REPLACE INTO source_metadata "
            "(source_id, domain, metadata, updated_at) VALUES (?, ?, ?, ?)",
            (source_id, existing["domain"], json.dumps(updated_meta), time.time()),
        )
        claims_retracted = store._retract_source_claims_tx(cx, source_id)

    after = store.get_source_metadata(source_id)

    return {
        "before": before,
        "after": {"domain": after["domain"], "metadata": after["metadata"]} if after else None,
        "claims_retracted": claims_retracted,
    }


def unretract_source(
    store: Store, source_id: int, reason: str,
) -> dict:
    """Reverse a retraction: flip retracted=False, clear retracted_at, and
    transition claims from 'retracted' back to 'active'. Wraps in store.tx()
    for atomicity.

    Claims dropped by a ``retracted`` contradiction disposition stay
    retracted: that was a decision about the claim, not about its source.
    """
    existing = store.get_source_metadata(source_id)
    if existing is None:
        return {"error": "no_metadata"}
    if existing["domain"] != "scientific":
        return {"error": "not_scientific_domain"}
    if not is_retracted(existing):
        return {"error": "not_retracted"}

    before = {
        "domain": existing["domain"],
        "metadata": dict(existing["metadata"]),
    }

    updated_meta = dict(existing["metadata"])
    updated_meta["retracted"] = False
    updated_meta.pop("retracted_at", None)
    updated_meta["unretraction_reason"] = reason

    with store.tx() as cx:
        # Update metadata
        cx.execute(
            "INSERT OR REPLACE INTO source_metadata "
            "(source_id, domain, metadata, updated_at) VALUES (?, ?, ?, ?)",
            (source_id, existing["domain"], json.dumps(updated_meta), time.time()),
        )

        # Flip retracted claims back to active, except those a `retracted`
        # contradiction disposition dropped.
        claim_rows = cx.execute(
            "SELECT id FROM claims WHERE source_id = ? AND status = 'retracted' "
            "AND retracted_cause IS NOT 'disposition'",
            (source_id,),
        ).fetchall()
        claim_ids = [r["id"] for r in claim_rows]
        if claim_ids:
            placeholders = ",".join("?" for _ in claim_ids)
            cx.execute(
                f"UPDATE claims SET status = 'active', retracted_cause = NULL "
                f"WHERE id IN ({placeholders})",
                claim_ids,
            )
            # Invalidate cached views that reference these claims
            _invalidate_cache_for_claims(cx, claim_ids, cause="unretract_source")
        # Claims elsewhere that were superseded by a claim of this source, or
        # claims of this source whose successor went away while it was
        # retracted, are re-evaluated now that the source is back.
        _revive_orphaned_supersessions_tx(cx, cause="unretract_source")

    after = store.get_source_metadata(source_id)

    return {
        "before": before,
        "after": {"domain": after["domain"], "metadata": after["metadata"]} if after else None,
        "claims_unretracted": len(claim_ids),
        "reason": reason,
    }
