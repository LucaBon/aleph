"""Agent-facing CLI: JSON in, JSON out, no LLM calls.

These commands are meant to be invoked by Claude Code (or any agent). The agent
does the reading, reasoning, and writing; these commands just persist and
retrieve. No ANTHROPIC_API_KEY needed.

Every command emits a single JSON object on stdout with the envelope:
  success: {"ok": true,  "data":  {...}}
  error:   {"ok": false, "error": {"code": "...", "message": "...", "details": {...}}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .db import Store


def _ok(data) -> None:
    print(json.dumps({"ok": True, "data": data}, ensure_ascii=False, default=str))


def _err(code: str, message: str, **details) -> None:
    err: dict = {"code": code, "message": message}
    if details:
        err["details"] = details
    print(json.dumps({"ok": False, "error": err}, ensure_ascii=False, default=str))


def _parse_id_tag_pairs(
    raw: str, *, default_tag: Optional[str] = None,
) -> list[tuple[int, str]]:
    """Parse a list of (claim_id, tag) pairs from a CLI argument.

    Accepts three forms (auto-detected):
      1. Legacy colon syntax: ``"26:premise,174:corroborating"``
         (bare ``"26"`` is accepted iff ``default_tag`` is set)
      2. JSON array of pairs:   ``"[[26,\"premise\"],[174,\"corroborating\"]]"``
      3. JSON object:           ``"{\"26\":\"premise\",\"174\":\"corroborating\"}"``
         (also accepts a list of ``{"claim_id": 26, "role": "premise"}`` or
         ``{"claim_id": 12, "kind": "sample"}`` objects).

    Raises ``ValueError`` with a human-readable message on any parse failure.
    The caller is responsible for semantic validation (role/kind vocabulary,
    claim existence).
    """
    if raw is None:
        return []
    s = raw.strip()
    if not s:
        return []
    if s[0] in "[{":
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from None
        pairs: list[tuple[int, str]] = []
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                try:
                    cid = int(k)
                except (TypeError, ValueError):
                    raise ValueError(f"JSON key {k!r} is not an integer claim id")
                if not isinstance(v, str):
                    raise ValueError(f"JSON value for {k!r} must be a string")
                pairs.append((cid, v))
            return pairs
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, list) and len(item) == 2:
                    cid_raw, tag = item
                    try:
                        cid = int(cid_raw)
                    except (TypeError, ValueError):
                        raise ValueError(f"JSON list entry {item!r}: first element is not an integer")
                    if not isinstance(tag, str):
                        raise ValueError(f"JSON list entry {item!r}: second element must be a string")
                    pairs.append((cid, tag))
                elif isinstance(item, dict):
                    cid_raw = item.get("claim_id")
                    tag = item.get("role") or item.get("kind")
                    if cid_raw is None or not isinstance(tag, str):
                        raise ValueError(
                            f"JSON object entry {item!r}: expected keys "
                            "'claim_id' and 'role'/'kind'"
                        )
                    try:
                        cid = int(cid_raw)
                    except (TypeError, ValueError):
                        raise ValueError(f"JSON object entry {item!r}: 'claim_id' must be an integer")
                    pairs.append((cid, tag))
                else:
                    raise ValueError(
                        f"JSON list entry {item!r} is not [id, tag] or "
                        "{claim_id, role/kind}"
                    )
            return pairs
        raise ValueError(f"JSON must be an object or array, got {type(parsed).__name__}")
    # Legacy: comma-separated id:tag tokens
    pairs = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":", 1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f"malformed token: {token!r} (expected 'id:tag')")
        try:
            cid = int(parts[0])
        except ValueError:
            raise ValueError(f"bad id in token: {token!r}")
        pairs.append((cid, parts[1].strip()))
    return pairs


# ---------- sources ----------

def cmd_source_add(args, store: Store) -> int:
    path = Path(args.path)
    if not path.is_file():
        _err("not_a_file", f"not a file: {path}", path=str(path))
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    source_id = store.add_source(str(path), text)
    if source_id is None:
        # already ingested — find existing
        existing = [r for r in store.list_sources() if r["path"] == str(path)]
        eid = existing[0]["id"] if existing else None
        _ok({"status": "already_ingested", "source_id": eid, "length": len(text)})
        return 0
    _ok({"status": "ingested", "source_id": source_id, "length": len(text)})
    return 0


def cmd_source_replace(args, store: Store) -> int:
    """Atomically remove any existing source at PATH and re-add from disk.

    Use when a source file has changed on disk: `source-add` would create a
    duplicate entry with a different sha256 and leave old claims orphaned.
    This primitive makes the replacement atomic so cached views citing the
    old claims are invalidated in the same transaction.
    """
    path = Path(args.path)
    if not path.is_file():
        _err("not_a_file", f"not a file: {path}", path=str(path))
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        result = store.replace_source(str(path), text)
    except ValueError as e:
        other = int(str(e).split(":", 1)[1])
        _err("duplicate_content",
             f"content is identical to source {other}; nothing was replaced",
             path=str(path), existing_source_id=other)
        return 1
    _ok({**result, "length": len(text)})
    return 0


def cmd_source_get(args, store: Store) -> int:
    row = store.get_source(args.source_id)
    if not row:
        _err("source_not_found", f"no source with id {args.source_id}",
             source_id=args.source_id)
        return 1
    out = {
        "source_id": row["id"],
        "path": row["path"],
        "sha256": row["sha256"],
        "length": len(row["content"]),
        "ingested_at": row["ingested_at"],
    }
    if not args.no_content:
        out["content"] = row["content"]
    _ok(out)
    return 0


def cmd_source_list(args, store: Store) -> int:
    rows = store.list_sources()
    out = []
    for r in rows:
        n = store.conn.execute(
            "SELECT COUNT(*) FROM claims WHERE source_id = ? AND status = 'active'",
            (r["id"],),
        ).fetchone()[0]
        out.append({"source_id": r["id"], "path": r["path"], "active_claims": n,
                    "ingested_at": r["ingested_at"]})
    _ok({"sources": out})
    return 0


def cmd_source_remove(args, store: Store) -> int:
    n = store.remove_source(args.source_id)
    if n == 0:
        _err("source_not_found", f"no source with id {args.source_id}",
             source_id=args.source_id)
        return 1
    _ok({"removed": n})
    return 0


def cmd_source_yield(args, store: Store) -> int:
    """Per-source claim density diagnostic: how many active claims per KB.

    Useful for triaging before phases that depend on claim density
    (concept derivation, contradiction scan): sources with unusually low
    yield often indicate a thin extraction pass, not a thin source.
    """
    thin_threshold = args.thin_threshold
    rows = store.conn.execute(
        "SELECT s.id AS source_id, s.path, LENGTH(s.content) AS size_bytes, "
        "SUM(CASE WHEN c.status = 'active' THEN 1 ELSE 0 END) AS active_claims, "
        "SUM(CASE WHEN c.status = 'superseded' THEN 1 ELSE 0 END) AS superseded_claims, "
        "SUM(CASE WHEN c.status = 'retracted' THEN 1 ELSE 0 END) AS retracted_claims "
        "FROM sources s LEFT JOIN claims c ON c.source_id = s.id "
        "GROUP BY s.id, s.path, s.content "
        "ORDER BY s.id"
    ).fetchall()

    sources_out: list[dict] = []
    yields: list[float] = []
    for r in rows:
        size_bytes = r["size_bytes"] or 0
        active = int(r["active_claims"] or 0)
        superseded = int(r["superseded_claims"] or 0)
        retracted = int(r["retracted_claims"] or 0)
        per_kb = (active * 1024.0 / size_bytes) if size_bytes > 0 else 0.0
        sources_out.append({
            "source_id": r["source_id"],
            "path": r["path"],
            "size_bytes": size_bytes,
            "active_claims": active,
            "superseded_claims": superseded,
            "retracted_claims": retracted,
            "claims_per_kb": round(per_kb, 3),
            "flagged_thin": size_bytes > 0 and per_kb < thin_threshold,
        })
        if size_bytes > 0:
            yields.append(per_kb)

    median = 0.0
    if yields:
        ys = sorted(yields)
        mid = len(ys) // 2
        median = ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2
    thin_count = sum(1 for s in sources_out if s["flagged_thin"])

    _ok({
        "sources": sources_out,
        "summary": {
            "source_count": len(sources_out),
            "median_claims_per_kb": round(median, 3),
            "thin_threshold": thin_threshold,
            "thin_sources_count": thin_count,
        },
    })
    return 0


# ---------- claims ----------

def cmd_claim_add(args, store: Store) -> int:
    """Add a claim. Span must be a verbatim substring of the source; we locate it
    and store the character offsets. If the span is not found, the claim is
    refused — ungrounded claims don't enter the store."""
    src = store.get_source(args.source_id)
    if not src:
        _err("source_not_found", f"no source with id {args.source_id}",
             source_id=args.source_id)
        return 1
    content = src["content"]
    span = args.span
    # exact match first
    idx = content.find(span)
    if idx < 0:
        _err(
            "span_not_in_source",
            "span not found in source (must be verbatim substring)",
            source_id=args.source_id,
            span_preview=span[:120],
        )
        return 1
    start, end = idx, idx + len(span)
    claim_id = store.add_claim(
        source_id=args.source_id,
        subject=args.subject,
        predicate=args.predicate,
        object_=args.object,
        span_start=start,
        span_end=end,
        confidence=args.confidence,
        proposition=getattr(args, "proposition", None),
    )
    # WS-D: optional --conditions flag (legacy colon form or JSON, same parser as --support)
    conditions_arg = getattr(args, "conditions", None)
    if conditions_arg:
        from . import conditions as _cond_mod

        def _rollback(claim_id):
            with store.tx() as cx:
                cx.execute("DELETE FROM claims WHERE id = ?", (claim_id,))

        try:
            pairs = _parse_id_tag_pairs(conditions_arg)
        except ValueError as e:
            _rollback(claim_id)
            _err("invalid_condition", str(e))
            return 1
        for cond_id, kind in pairs:
            if kind not in _cond_mod.VALID_KINDS:
                _rollback(claim_id)
                _err("invalid_condition", f"invalid kind {kind!r}",
                     valid_kinds=sorted(_cond_mod.VALID_KINDS))
                return 1
            if cond_id == claim_id:
                _rollback(claim_id)
                _err("invalid_condition", "condition cannot reference the claim itself")
                return 1
            if not store.get_claim(cond_id):
                _rollback(claim_id)
                _err("invalid_condition", f"condition claim {cond_id} does not exist")
                return 1
        # All validated — link them
        for cond_id, kind in pairs:
            _cond_mod.link_condition(store, claim_id, cond_id, kind, explicit=True)

    row = store.get_claim(claim_id)
    store.clear_cache()
    # Fidelity issues flag the claim for review; they never refuse it.
    flag = store.open_claim_review(claim_id, "fidelity")
    _ok({
        "claim_id": claim_id,
        "subject": row["subject"],   # normalized/canonical form
        "predicate": row["predicate"],
        "object": row["object"],
        "proposition": row["proposition"],
        "span_start": start,
        "span_end": end,
        "fidelity_issues": json.loads(flag["details"])["issues"] if flag else [],
        "review_id": flag["id"] if flag else None,
    })
    return 0


def cmd_claim_get(args, store: Store) -> int:
    row = store.get_claim(args.claim_id)
    if not row:
        _err("claim_not_found", f"no claim with id {args.claim_id}",
             claim_id=args.claim_id)
        return 1
    span = store.get_span_text(args.claim_id) or ""
    src = store.get_source(row["source_id"])
    _ok({
        "claim_id": row["id"],
        "source_id": row["source_id"],
        "source_path": src["path"] if src else None,
        "subject": row["subject"],
        "predicate": row["predicate"],
        "object": row["object"],
        "confidence": row["confidence"],
        "status": row["status"],
        "superseded_by": row["superseded_by"],
        "span_start": row["span_start"],
        "span_end": row["span_end"],
        "span_text": span,
        "proposition": row["proposition"],
        "context_start": row["context_start"],
        "context_end": row["context_end"],
        "context_text": store.get_context_text(args.claim_id),
    })
    return 0


def cmd_claim_fidelity_check(args, store: Store) -> int:
    """Re-run the deterministic fidelity checker on one claim or on every
    active claim; ``--enqueue`` queues flagged claims for review."""
    if args.all:
        ids = [r["id"] for r in store.all_active_claims()]
    elif args.claim_id is not None:
        if not store.get_claim(args.claim_id):
            _err("claim_not_found", f"no claim with id {args.claim_id}",
                 claim_id=args.claim_id)
            return 1
        ids = [args.claim_id]
    else:
        _err("missing_target", "pass a CLAIM_ID or --all")
        return 1
    flagged = []
    for cid in ids:
        issues = [i.to_dict() for i in store.check_claim_fidelity(cid)]
        if not issues:
            continue
        entry = {"claim_id": cid, "issues": issues}
        if args.enqueue:
            entry["review_id"] = store.enqueue_review(
                "claim", cid, reason="fidelity", details={"issues": issues})
        flagged.append(entry)
    _ok({"checked": len(ids), "flagged": flagged})
    return 0


def cmd_counter_evidence(args, store: Store) -> int:
    """The agent-mode counterpart of ask's counter-evidence check: given the
    claims an answer cites, the uncited sides of their live conflicts."""
    from .query import counter_evidence

    try:
        ids = [int(x) for x in args.claim_ids.split(",") if x.strip()]
    except ValueError:
        _err("invalid_claim_ids", "--claim-ids must be comma-separated integers",
             claim_ids=args.claim_ids)
        return 1
    _ok({"claim_ids": ids, "counter_evidence": counter_evidence(store, ids)})
    return 0


def _review_dict(row) -> dict:
    target = {
        "claim": {"claim_id": row["claim_id"]},
        "contradiction": {"contradiction_id": row["contradiction_id"]},
        "concept": {"concept_id": row["concept_id"]},
        "alias": {"alias_from": row["alias_from"]},
    }[row["item_type"]]
    return {
        "review_id": row["id"],
        "item_type": row["item_type"],
        "target": target,
        "reason": row["reason"],
        "details": json.loads(row["details"] or "{}"),
        "status": row["status"],
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
        "resolved_by": row["resolved_by"],
        "resolution_note": row["resolution_note"],
    }


# What to do after rejecting an item: recording the decision fixes nothing.
_REJECT_NEXT_STEP = {
    "claim": "supersede the claim with a corrected one (claim-add, then claim-supersede)",
    "contradiction": "re-dispose it (contradiction-dispose)",
    "concept": "invalidate or rebuild it (concept-invalidate / concept-rebuild)",
    "alias": "undo the merge (alias-undo)",
}


def cmd_review_list(args, store: Store) -> int:
    rows = store.list_reviews(status=args.status, item_type=args.type, limit=args.limit)
    _ok({"items": [_review_dict(r) for r in rows]})
    return 0


def cmd_review_add(args, store: Store) -> int:
    wants_alias = args.type == "alias"
    if wants_alias != (args.alias is not None) or wants_alias == (args.id is not None):
        _err("invalid_review_target",
             "use --alias FROM for alias items and --id N for the others",
             item_type=args.type)
        return 1
    target = args.alias if args.type == "alias" else args.id
    details = {"note": args.note} if args.note else {}
    try:
        rid = store.enqueue_review(args.type, target, reason=args.reason, details=details)
    except LookupError as e:
        _err("review_target_not_found", str(e), item_type=args.type, target=target)
        return 1
    _ok(_review_dict(store.get_review(rid)))
    return 0


def cmd_review_resolve(args, store: Store) -> int:
    row = store.get_review(args.review_id)
    if row is None:
        _err("review_not_found", f"no review item {args.review_id}",
             review_id=args.review_id)
        return 1
    if args.decision not in ("accepted", "rejected"):
        _err("invalid_decision", "decision must be accepted or rejected",
             decision=args.decision)
        return 1
    if row["status"] != "open":
        _err("review_not_open", f"review item {args.review_id} is already {row['status']}",
             review_id=args.review_id, status=row["status"])
        return 1
    store.resolve_review(args.review_id, args.decision, resolved_by=args.by, note=args.note)
    out = _review_dict(store.get_review(args.review_id))
    if args.decision == "rejected":
        out["next_step"] = _REJECT_NEXT_STEP[row["item_type"]]
    _ok(out)
    return 0


_CLAIM_COMPACT_FIELDS = ("claim_id", "predicate", "object", "confidence")


def _project_claim_dict(item: dict, fields: Optional[list[str]], compact: bool) -> dict:
    """Project a claim dict to a subset of fields.

    ``--compact`` emits {claim_id, predicate, object (<= 60 chars), confidence}.
    ``--fields a,b,c`` emits only those keys that exist on the item. The two
    flags do not combine — if both are set, ``fields`` wins.
    """
    if fields:
        return {k: item[k] for k in fields if k in item}
    if compact:
        out = {k: item[k] for k in _CLAIM_COMPACT_FIELDS if k in item}
        obj = out.get("object")
        if isinstance(obj, str) and len(obj) > 60:
            out["object"] = obj[:60] + "…"
        return out
    return item


def _parse_fields_arg(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    return [f.strip() for f in raw.split(",") if f.strip()]


def cmd_claim_search(args, store: Store) -> int:
    keywords = args.query.split() if args.query else []
    rows = store.search_claims(keywords, limit=args.k)
    fields = _parse_fields_arg(getattr(args, "fields", None))
    compact = bool(getattr(args, "compact", False))
    out = []
    for r in rows:
        item = {
            "claim_id": r["id"],
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "confidence": r["confidence"],
            "hits": r["hits"] if "hits" in r.keys() else None,
        }
        if args.with_spans:
            item["span_text"] = store.get_span_text(r["id"]) or ""
        out.append(_project_claim_dict(item, fields, compact))
    _ok({"query": args.query, "results": out})
    return 0


def cmd_claim_by_subject(args, store: Store) -> int:
    """Return all active claims whose subject matches (after normalization+alias)."""
    canonical = store.resolve_subject(args.subject)
    rows = store.conn.execute(
        "SELECT * FROM claims WHERE subject = ? AND status = 'active' ORDER BY predicate",
        (canonical,),
    ).fetchall()
    fields = _parse_fields_arg(getattr(args, "fields", None))
    compact = bool(getattr(args, "compact", False))
    out = []
    for r in rows:
        item = {
            "claim_id": r["id"],
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "confidence": r["confidence"],
        }
        out.append(_project_claim_dict(item, fields, compact))
    _ok({"subject_input": args.subject, "subject_canonical": canonical, "claims": out})
    return 0


def cmd_claim_supersede(args, store: Store) -> int:
    if not store.get_claim(args.old_id):
        _err("claim_not_found", f"no claim with id {args.old_id}",
             claim_id=args.old_id)
        return 1
    if not store.get_claim(args.new_id):
        _err("claim_not_found", f"no claim with id {args.new_id}",
             claim_id=args.new_id)
        return 1
    store.supersede_claim(args.old_id, args.new_id)
    _ok({"superseded": args.old_id, "by": args.new_id})
    return 0


def cmd_subjects(args, store: Store) -> int:
    """List distinct subjects with claim counts (useful for finding synonyms)."""
    rows = store.conn.execute(
        "SELECT subject, COUNT(*) AS n FROM claims WHERE status = 'active' "
        "GROUP BY subject ORDER BY n DESC, subject"
    ).fetchall()
    _ok({"subjects": [{"subject": r["subject"], "count": r["n"]} for r in rows]})
    return 0


# ---------- aliases ----------

def cmd_alias_add(args, store: Store) -> int:
    result = store.add_alias(args.from_subject, args.to_subject,
                             force=getattr(args, "force", False))
    if result.get("note") == "alias-exists":
        _err(
            "alias_exists",
            "FROM is already an alias for a different subject; pass --force to "
            "overwrite (the overwrite is logged and can be undone with alias-undo)",
            from_canonical=store.normalize_subject(args.from_subject),
            existing_to=result["existing_to"],
            requested_to=store.normalize_subject(args.to_subject),
        )
        return 1
    if result.get("note") == "would-create-cycle":
        _err(
            "alias_would_create_cycle",
            "adding this alias would create a resolution cycle",
            from_canonical=store.normalize_subject(args.from_subject),
            to_canonical=store.normalize_subject(args.to_subject),
        )
        return 1
    result["from_canonical"] = store.normalize_subject(args.from_subject)
    result["to_canonical"] = store.normalize_subject(args.to_subject)
    _ok(result)
    return 0


def cmd_alias_undo(args, store: Store) -> int:
    result = store.undo_alias(args.from_subject)
    if not result["undone"] and result["note"] == "would-create-cycle":
        _err("alias_would_cycle",
             f"restoring {result['alias_from']} -> {result['restore_to']} would create "
             "an alias cycle; undo the alias that leads back first",
             from_canonical=result["alias_from"], restore_to=result["restore_to"])
        return 1
    if not result["undone"]:
        _err("alias_not_found", "no live alias merge for this subject",
             from_canonical=result["alias_from"])
        return 1
    _ok(result)
    return 0


def cmd_alias_log(args, store: Store) -> int:
    rows = store.list_alias_events(limit=args.limit)
    _ok({"events": [
        {"event_id": r["id"], "from": r["alias_from"], "to": r["canonical_to"],
         "previous_to": r["previous_to"], "created_at": r["created_at"],
         "undone_at": r["undone_at"], "claims_rewritten": r["claims_rewritten"]}
        for r in rows
    ]})
    return 0


def cmd_invariant_check(args, store: Store) -> int:
    violations = store.check_invariants()
    if violations:
        _err("invariant_violated",
             f"{len(violations)} invariant violation(s)",
             violations=violations)
        return 1
    _ok({"violations": []})
    return 0


def cmd_alias_list(args, store: Store) -> int:
    rows = store.list_aliases()
    _ok({
        "aliases": [
            {"from": r["alias_from"], "to": r["canonical_to"], "created_at": r["created_at"]}
            for r in rows
        ]
    })
    return 0


# ---------- contradictions ----------

# P1.6: small vocabulary for the cross-subject escape valve. Keeping this
# tight prevents the flag from becoming a dumping ground for "well it's
# related" pairs.
_CROSS_SUBJECT_RELATION_KINDS = {
    "regime-supersedes",     # one legal/policy regime strikes down / replaces another
    "rule-limits-rule",      # one rule constrains the scope of another
    "doctrinal-cross-ref",   # different subjects linked by an explicit doctrinal pointer
}


def cmd_contradiction_add(args, store: Store) -> int:
    """Record a contradiction between two claims.

    Validates that both claims exist and that they share the same (resolved)
    subject — a contradiction that spans subjects is almost certainly a bug
    in the caller's reasoning. Predicates may differ (lint groups by subject
    only, so "last on average" vs "last" is a legitimate pair), but identical
    objects are rejected (nothing to contradict).

    P1.6 escape valve: ``--cross-subject --relation-kind X --justification Y``
    records a legitimate doctrinal tension across subjects (e.g. a
    Constitutional ruling about one subject striking down a rule about
    another). All three flags must be provided together; the resulting row
    has ``cross_subject=1`` and is otherwise kept out of same-subject
    disposition logic.
    """
    # Accept either positional args (legacy: `contradiction-add A B`) or the
    # flag form (`--claim-a A --claim-b B`). Flags win if given.
    claim_a = args.claim_a_flag if args.claim_a_flag is not None else args.claim_a
    claim_b = args.claim_b_flag if args.claim_b_flag is not None else args.claim_b
    if claim_a is None or claim_b is None:
        _err("missing_arg",
             "contradiction-add requires two claim ids "
             "(positional A B or --claim-a/--claim-b)")
        return 1
    a = store.get_claim(claim_a)
    if not a:
        _err("claim_not_found", f"no claim with id {claim_a}",
             claim_id=claim_a)
        return 1
    b = store.get_claim(claim_b)
    if not b:
        _err("claim_not_found", f"no claim with id {claim_b}",
             claim_id=claim_b)
        return 1
    if claim_a == claim_b:
        _err("contradiction_invalid",
             "cannot contradict a claim with itself",
             claim_id=claim_a)
        return 1
    if a["object"].strip().lower() == b["object"].strip().lower():
        _err(
            "contradiction_invalid",
            "claims have identical objects — nothing to contradict",
            object=a["object"],
        )
        return 1

    cross_subject = bool(getattr(args, "cross_subject", False))
    relation_kind = getattr(args, "relation_kind", None)
    justification = getattr(args, "justification", None)

    if cross_subject:
        if not relation_kind or not justification:
            _err(
                "cross_subject_missing_args",
                "--cross-subject requires --relation-kind and --justification",
                relation_kind=relation_kind,
                justification=justification,
                allowed_relation_kinds=sorted(_CROSS_SUBJECT_RELATION_KINDS),
            )
            return 1
        if relation_kind not in _CROSS_SUBJECT_RELATION_KINDS:
            _err(
                "invalid_relation_kind",
                f"relation_kind {relation_kind!r} not in allowed set",
                allowed=sorted(_CROSS_SUBJECT_RELATION_KINDS),
            )
            return 1
        cid = store.add_contradiction_cross_subject(
            claim_a, claim_b,
            relation_kind=relation_kind, justification=justification,
        )
        if cid is None:
            _ok({"contradiction_id": None, "created": False,
                 "note": "contradiction already exists"})
            return 0
        _ok({
            "contradiction_id": cid, "created": True,
            "cross_subject": True,
            "relation_kind": relation_kind,
            "justification": justification,
        })
        return 0

    # Same-subject default path.
    if relation_kind or justification:
        _err(
            "cross_subject_flag_missing",
            "--relation-kind and --justification are only valid with --cross-subject",
        )
        return 1
    if a["subject"] != b["subject"]:
        _err(
            "contradiction_invalid",
            "claims have different subjects — not a contradiction "
            "(use --cross-subject --relation-kind ... --justification ... "
            "for legitimate doctrinal cross-references)",
            subject_a=a["subject"], subject_b=b["subject"],
            allowed_relation_kinds=sorted(_CROSS_SUBJECT_RELATION_KINDS),
        )
        return 1
    cid = store.add_contradiction(claim_a, claim_b)
    if cid is None:
        _ok({"contradiction_id": None, "created": False,
             "note": "contradiction already exists"})
        return 0
    _ok({"contradiction_id": cid, "created": True, "cross_subject": False})
    return 0


def cmd_contradiction_list(args, store: Store) -> int:
    rows = store.list_contradictions(only_open=not args.all)
    out = []
    for r in rows:
        a = store.get_claim(r["claim_a_id"])
        b = store.get_claim(r["claim_b_id"])
        cross_flag = (r["cross_subject"] if "cross_subject" in r.keys() else 0) or 0
        entry = {
            "contradiction_id": r["id"],
            "status": r["status"],
            "cross_subject": bool(cross_flag),
            "relation_kind": r["relation_kind"] if "relation_kind" in r.keys() else None,
            "claim_a": {"id": a["id"], "subject": a["subject"],
                        "predicate": a["predicate"], "object": a["object"]} if a else None,
            "claim_b": {"id": b["id"], "subject": b["subject"],
                        "predicate": b["predicate"], "object": b["object"]} if b else None,
        }
        # WS-B: extended with --disposition / --kind filters
        if hasattr(args, "disposition") and args.disposition:
            disp = r["disposition"] if "disposition" in r.keys() else None
            if disp != args.disposition:
                continue
        if hasattr(args, "kind") and args.kind:
            k = r["kind"] if "kind" in r.keys() else None
            if k != args.kind:
                continue
        # P1.6: optional filters on the cross-subject escape valve.
        cross_filter = getattr(args, "cross_subject_filter", "any")
        if cross_filter == "only" and not cross_flag:
            continue
        if cross_filter == "exclude" and cross_flag:
            continue
        out.append(entry)
    _ok({"contradictions": out})
    return 0


def cmd_contradiction_resolve(args, store: Store) -> int:
    """Mark a specific contradiction as resolved; optionally supersede the losing claim.

    Validates that `--keep` (and `--drop`, if given) are the two claim IDs of
    the given contradiction — the most common caller bug is passing unrelated
    claim IDs by mistake.
    """
    crow = store.conn.execute(
        "SELECT * FROM contradictions WHERE id = ?", (args.contradiction_id,)
    ).fetchone()
    if not crow:
        _err("contradiction_not_found",
             f"no contradiction with id {args.contradiction_id}",
             contradiction_id=args.contradiction_id)
        return 1
    pair = {crow["claim_a_id"], crow["claim_b_id"]}
    if args.keep not in pair:
        _err(
            "contradiction_invalid",
            "--keep must be one of the two claim ids in the contradiction",
            contradiction_id=args.contradiction_id,
            contradiction_claims=sorted(pair),
            keep=args.keep,
        )
        return 1
    if args.drop is not None and args.drop not in pair:
        _err(
            "contradiction_invalid",
            "--drop must be one of the two claim ids in the contradiction",
            contradiction_id=args.contradiction_id,
            contradiction_claims=sorted(pair),
            drop=args.drop,
        )
        return 1
    if args.drop is not None and args.drop == args.keep:
        _err(
            "contradiction_invalid",
            "--keep and --drop must be different claim ids",
            keep=args.keep, drop=args.drop,
        )
        return 1
    with store.tx() as cx:
        cx.execute(
            "UPDATE contradictions SET status = 'resolved', resolved_to = ? WHERE id = ?",
            (args.keep, args.contradiction_id),
        )
        from .db import _invalidate_cache_for_contradiction_tx
        _invalidate_cache_for_contradiction_tx(cx, args.contradiction_id, cause="resolve")
    if args.drop is not None:
        store.supersede_claim(args.drop, args.keep)
    _ok({"resolved": args.contradiction_id, "kept": args.keep, "superseded": args.drop})
    return 0


# ---------- view cache ----------

def cmd_view_get(args, store: Store) -> int:
    row = store.get_cached_view(args.query)
    if not row:
        _ok({"cached": False})
        return 0
    _ok({
        "cached": True,
        "query": row["query"],
        "response": row["response"],
        "claim_ids": json.loads(row["claim_ids"]),
        "generated_at": row["generated_at"],
    })
    return 0


def cmd_view_cache(args, store: Store) -> int:
    claim_ids = [int(x) for x in args.claim_ids.split(",") if x.strip()]
    try:
        store.cache_view(args.query, args.response, claim_ids)
    except ValueError as e:
        inactive = str(e).split(":", 1)[1]
        _err("view_cites_inactive_claim",
             "a cached view may only cite active claims",
             inactive_claim_ids=[int(x) for x in inactive.split(",")])
        return 1
    _ok({"cached": True, "claim_ids": claim_ids})
    return 0


def cmd_cache_clear(args, store: Store) -> int:
    n = store.clear_cache()
    _ok({"cleared": n})
    return 0


# ---------- store config ----------

_CONFIG_WHITELIST = {"locale"}
# Readable via config-get but never writable: migrations own schema_version.
_CONFIG_READONLY = {"schema_version"}


def cmd_config_get(args, store: Store) -> int:
    """Return the value of a single config key, or a dump of everything.

    No key -> return all config rows (``{"config": {...}, "locale": "..."}``).
    Key supplied -> return just that key's value or ``None``.
    """
    if args.key is None:
        cfg = store.list_config()
        _ok({"config": cfg, "locale": store.locale})
        return 0
    if args.key not in _CONFIG_WHITELIST | _CONFIG_READONLY:
        _err("unknown_config_key", f"unknown config key {args.key!r}",
             allowed=sorted(_CONFIG_WHITELIST | _CONFIG_READONLY))
        return 1
    _ok({"key": args.key, "value": store.get_config(args.key)})
    return 0


def cmd_config_set(args, store: Store) -> int:
    """Set a config key. Only whitelisted keys are writable; the ``locale``
    key additionally goes through :meth:`Store.set_locale` so the active
    normalizer reloads."""
    if args.key in _CONFIG_READONLY:
        _err("readonly_config_key",
             f"config key {args.key!r} is managed by migrations")
        return 1
    if args.key not in _CONFIG_WHITELIST:
        _err("unknown_config_key", f"unknown config key {args.key!r}",
             allowed=sorted(_CONFIG_WHITELIST))
        return 1
    if args.key == "locale":
        try:
            result = store.set_locale(args.value)
        except ValueError as e:
            _err("invalid_locale", str(e))
            return 1
        _ok({"key": "locale", "value": args.value,
             "old": result["old"], "new": result["new"],
             "note": (
                 "existing aliases were not rewritten; subject normalization "
                 "under the new locale may produce different canonical forms"
             )})
        return 0
    store.set_config(args.key, args.value)
    _ok({"key": args.key, "value": args.value})
    return 0


# ---------- stats ----------

def cmd_stats(args, store: Store) -> int:
    _ok(store.stats())
    return 0


# ------------- WS-A CONCEPTS (from docs/plan/10-ws-concepts.md) -------------
# (cmd_concept_add, cmd_concept_derive, cmd_concept_get, cmd_concept_list,
#  cmd_concept_validate, cmd_concept_rebuild, cmd_concept_supersede,
#  cmd_concept_invalidate go here)


def _get_llm_for_agent(args):
    """Construct an LLM instance for agent commands that need one.

    Returns ``(llm, error_code)`` — if ``error_code`` is not None, emit
    ``_err`` and return. Honours ``--mock-llm`` and ``ALEPH_LLM_FIXTURES``
    so agent-mode LLM commands (concept-derive, concept-validate,
    concept-rebuild, contradiction-scan, claim-condition-extract) can run
    offline in CI against canned fixtures (P2.1).
    """
    import os
    fixtures = getattr(args, "mock_llm", None) or os.environ.get(
        "ALEPH_LLM_FIXTURES"
    )
    if fixtures:
        from .llm import MockLLM
        return MockLLM(fixtures), None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, 2
    from .llm import LLM, DEFAULT_MODEL
    model = getattr(args, "model", None) or DEFAULT_MODEL
    return LLM(model=model), None


def cmd_concept_add(args, store: Store) -> int:
    """Add a concept with its support set. Optionally validate immediately.

    Accepts --support in either legacy colon form (``26:premise,174:premise``)
    or JSON (array of pairs/objects, or object). See :func:`_parse_id_tag_pairs`.
    """
    try:
        support_pairs = _parse_id_tag_pairs(args.support or "")
    except ValueError as e:
        _err("invalid_support", str(e))
        return 1

    if not support_pairs:
        _err("no_support", "at least one supporting claim is required")
        return 1

    # Default role to "premise" if caller used ``"26"`` or ``"26:"`` in legacy
    # form (the parser already rejects ``26:`` with empty tag, but a bare id
    # from JSON lists without a role is possible if someone passes `[26]`).
    _VALID_ROLES = {"premise", "corroborating", "qualifying", "counterexample"}
    for cid, role in support_pairs:
        if role not in _VALID_ROLES:
            _err(
                "invalid_support",
                f"invalid role {role!r} (expected one of {sorted(_VALID_ROLES)})",
                support_id=cid,
                role=role,
            )
            return 1

    # V1 constraint: concepts cannot cite other concepts. Each support id
    # must be a CLAIM (not a concept). We check by verifying get_claim returns
    # a row.
    for cid, _role in support_pairs:
        if not store.get_claim(cid):
            _err("support_must_be_claim",
                 f"id {cid} is not a claim (concepts cannot support concepts in v1)",
                 support_id=cid)
            return 1

    concept_id = store.add_concept(
        subject=store.resolve_subject(args.subject),
        statement=args.statement,
        inference_type=args.inference_type,
        confidence=args.confidence,
        support_claim_ids=support_pairs,
        status="draft",
    )

    verdict = None
    reason = None
    status = "draft"

    if not args.skip_validation:
        llm, err = _get_llm_for_agent(args)
        if err is not None:
            _err("missing_api_key", "ANTHROPIC_API_KEY not set")
            return 2
        from .concepts import validate_concept
        verdict, reason = validate_concept(store, llm, concept_id)
        row = store.get_concept(concept_id)
        status = row["status"]

    out = {
        "concept_id": concept_id,
        "status": status,
        "validation_verdict": verdict,
        "validation_reason": reason,
    }
    # Honesty contract: a concept stuck at draft cannot be used as a
    # rationale_concept in contradiction-dispose. Surface this explicitly so
    # agents (especially in fully agent-mode runs without an API key) know
    # that concept-validate is the documented promotion path.
    if status == "draft":
        out["note"] = (
            "draft concepts cannot be cited as rationale_concept in "
            "contradiction-dispose; promotion to 'active' requires "
            "concept-validate (LLM)"
        )
    _ok(out)
    return 0


def cmd_concept_derive(args, store: Store) -> int:
    """Derive concepts from claims about a subject (LLM-calling)."""
    llm, err = _get_llm_for_agent(args)
    if err is not None:
        _err("missing_api_key", "ANTHROPIC_API_KEY not set")
        return 2

    claim_ids = None
    if args.claims:
        claim_ids = [int(x) for x in args.claims.split(",") if x.strip()]

    from .concepts import derive_concepts
    new_ids = derive_concepts(
        store, llm, args.subject,
        claim_ids=claim_ids, limit=args.limit,
    )

    concepts_out = []
    for cid in new_ids:
        row = store.get_concept(cid)
        if row:
            concepts_out.append({
                "concept_id": cid,
                "status": row["status"],
                "statement": row["statement"],
                "validation_verdict": row["validation_verdict"],
            })

    _ok({"subject": args.subject, "concepts": concepts_out})
    return 0


def cmd_concept_get(args, store: Store) -> int:
    """Get a concept with its support set."""
    row = store.get_concept(args.concept_id)
    if not row:
        _err("concept_not_found", f"no concept with id {args.concept_id}",
             concept_id=args.concept_id)
        return 1

    supports = store.get_concept_supports(args.concept_id)
    support_out = []
    for s in supports:
        item = {
            "claim_id": s["claim_id"],
            "role": s["role"],
            "subject": s["subject"],
            "predicate": s["predicate"],
            "object": s["object"],
            "claim_confidence": s["claim_confidence"],
            "claim_status": s["claim_status"],
        }
        if args.with_spans:
            item["span_text"] = store.get_span_text(s["claim_id"]) or ""
        support_out.append(item)

    out = {
        "concept_id": row["id"],
        "subject": row["subject"],
        "statement": row["statement"],
        "inference_type": row["inference_type"],
        "confidence": row["confidence"],
        "status": row["status"],
        "superseded_by": row["superseded_by"],
        "validation_verdict": row["validation_verdict"],
        "validation_reason": row["validation_reason"],
        "supports": support_out,
    }
    # P1.2: surface attestation trail when present. Reading an attested
    # concept should always show the attestor and rationale — the trail is
    # the honesty signal that replaces the LLM GROUNDED verdict.
    if "attested_by" in row.keys() and row["attested_by"] is not None:
        out["attested_by"] = row["attested_by"]
        out["attested_at"] = row["attested_at"] if "attested_at" in row.keys() else None
        out["attestation_rationale"] = (
            row["attestation_rationale"] if "attestation_rationale" in row.keys() else None
        )
    _ok(out)
    return 0


def cmd_concept_list(args, store: Store) -> int:
    """List concepts, optionally filtered by subject and/or status."""
    subject = store.resolve_subject(args.subject) if args.subject else None
    concepts = store.list_concepts(
        subject=subject, status=args.status, limit=args.limit,
    )
    out = []
    for row in concepts:
        entry = {
            "concept_id": row["id"],
            "subject": row["subject"],
            "statement": row["statement"],
            "inference_type": row["inference_type"],
            "confidence": row["confidence"],
            "status": row["status"],
            "validation_verdict": row["validation_verdict"],
        }
        # P1.2: attestation trail travels with the concept in listings too,
        # so agents can filter/inspect without a follow-up concept-get.
        if "attested_by" in row.keys() and row["attested_by"] is not None:
            entry["attested_by"] = row["attested_by"]
        out.append(entry)
    _ok({"concepts": out})
    return 0


def cmd_concept_attest(args, store: Store) -> int:
    """Promote a draft concept to 'attested' (P1.2).

    Attestation is the agent-mode counterpart to ``concept-validate``: in a
    run without an LLM there is no GROUNDED verdict to gate on, so the
    operator records their own attestation trail (``attested_by`` plus a
    free-text rationale). ``attested`` concepts are accepted as
    ``rationale_concept_id`` on ``contradiction-dispose`` alongside
    ``active``. Only ``draft`` concepts can be attested; other statuses
    must go through rebuild/validate.
    """
    row = store.get_concept(args.concept_id)
    if not row:
        _err("concept_not_found", f"no concept with id {args.concept_id}",
             concept_id=args.concept_id)
        return 1
    if row["status"] != "draft":
        _err(
            "concept_not_draft",
            f"only draft concepts can be attested; current status is {row['status']!r}",
            concept_id=args.concept_id,
            current_status=row["status"],
            required_status="draft",
        )
        return 1
    try:
        store.attest_concept(
            args.concept_id,
            attested_by=args.attested_by,
            rationale=args.rationale,
        )
    except ValueError as e:
        _err("concept_attest_failed", str(e), concept_id=args.concept_id)
        return 1
    updated = store.get_concept(args.concept_id)
    _ok({
        "concept_id": args.concept_id,
        "status": updated["status"],
        "attested_by": updated["attested_by"],
        "attested_at": updated["attested_at"],
        "attestation_rationale": updated["attestation_rationale"],
    })
    return 0


def cmd_concept_validate(args, store: Store) -> int:
    """Re-run validation on an existing concept (LLM-calling)."""
    row = store.get_concept(args.concept_id)
    if not row:
        _err("concept_not_found", f"no concept with id {args.concept_id}",
             concept_id=args.concept_id)
        return 1

    llm, err = _get_llm_for_agent(args)
    if err is not None:
        _err("missing_api_key", "ANTHROPIC_API_KEY not set")
        return 2

    from .concepts import validate_concept
    verdict, reason = validate_concept(store, llm, args.concept_id)
    updated = store.get_concept(args.concept_id)

    _ok({
        "concept_id": args.concept_id,
        "status": updated["status"],
        "validation_verdict": verdict,
        "validation_reason": reason,
    })
    return 0


def cmd_concept_rebuild(args, store: Store) -> int:
    """Rebuild a stale/active concept (LLM-calling)."""
    row = store.get_concept(args.concept_id)
    if not row:
        _err("concept_not_found", f"no concept with id {args.concept_id}",
             concept_id=args.concept_id)
        return 1

    llm, err = _get_llm_for_agent(args)
    if err is not None:
        _err("missing_api_key", "ANTHROPIC_API_KEY not set")
        return 2

    from .concepts import rebuild_concept
    new_id = rebuild_concept(store, llm, args.concept_id)
    new_row = store.get_concept(new_id)

    _ok({
        "old_concept_id": args.concept_id,
        "new_concept_id": new_id,
        "status": new_row["status"] if new_row else None,
        "statement": new_row["statement"] if new_row else None,
        "validation_verdict": new_row["validation_verdict"] if new_row else None,
    })
    return 0


def cmd_concept_supersede(args, store: Store) -> int:
    """Manually supersede one concept by another."""
    old = store.get_concept(args.old_id)
    if not old:
        _err("concept_not_found", f"no concept with id {args.old_id}",
             concept_id=args.old_id)
        return 1
    new = store.get_concept(args.new_id)
    if not new:
        _err("concept_not_found", f"no concept with id {args.new_id}",
             concept_id=args.new_id)
        return 1
    store.update_concept_status(args.old_id, "superseded", superseded_by=args.new_id)
    _ok({"superseded": args.old_id, "by": args.new_id})
    return 0


def cmd_concept_invalidate(args, store: Store) -> int:
    """Manually invalidate a concept with a reason."""
    row = store.get_concept(args.concept_id)
    if not row:
        _err("concept_not_found", f"no concept with id {args.concept_id}",
             concept_id=args.concept_id)
        return 1
    from .concepts import invalidate_concept
    invalidate_concept(store, args.concept_id, args.reason)
    _ok({"concept_id": args.concept_id, "status": "invalidated", "reason": args.reason})
    return 0


# ------------- WS-B CONTRADICTIONS (from docs/plan/20-ws-contradictions.md) -


def cmd_contradiction_scan(args, store: Store) -> int:
    """Scan for contradictions (LLM-calling)."""
    llm, err = _get_llm_for_agent(args)
    if err is not None:
        _err("missing_api_key", "ANTHROPIC_API_KEY not set")
        return 2

    from .contradictions import detect_all

    since = None
    if args.since:
        try:
            since = float(args.since)
        except ValueError:
            # try ISO parse
            import datetime
            try:
                dt = datetime.datetime.fromisoformat(args.since)
                since = dt.timestamp()
            except (ValueError, TypeError):
                _err("invalid_since", f"cannot parse --since: {args.since}")
                return 1

    summary = detect_all(
        store, llm,
        subject=args.subject,
        kind=args.kind,
        since=since,
        verbose=args.verbose,
    )
    _ok(summary)
    return 0


def cmd_contradiction_dispose(args, store: Store) -> int:
    """Apply a disposition to a contradiction."""
    from .contradictions import dispose, _ALLOWED_DISPOSITIONS

    # validate contradiction exists
    crow = store.conn.execute(
        "SELECT * FROM contradictions WHERE id = ?", (args.contradiction_id,)
    ).fetchone()
    if not crow:
        _err("contradiction_not_found",
             f"no contradiction with id {args.contradiction_id}",
             contradiction_id=args.contradiction_id)
        return 1

    if args.disposition not in _ALLOWED_DISPOSITIONS:
        _err("invalid_disposition",
             f"disposition {args.disposition!r} not in allowed set",
             allowed=sorted(_ALLOWED_DISPOSITIONS))
        return 1

    applies_when = None
    if args.applies_when:
        try:
            applies_when = json.loads(args.applies_when)
        except json.JSONDecodeError:
            _err("invalid_applies_when", "cannot parse --applies-when as JSON")
            return 1

    try:
        result = dispose(
            store, args.contradiction_id, args.disposition,
            rule=args.rule,
            applies_when=applies_when,
            rationale_concept_id=args.rationale_concept,
            keep=args.keep,
            drop=args.drop,
        )
    except ValueError as e:
        code = str(e)
        # strip any suffix after colon for clean error codes
        if ":" in code:
            code = code.split(":")[0]
        details: dict = {}
        # Surface structured context on rationale-concept errors so an agent
        # does not need a separate concept-get to understand the failure.
        if code in ("rationale_concept_not_active",
                    "rationale_concept_not_found"
                    ) and args.rationale_concept is not None:
            concept = store.get_concept(args.rationale_concept)
            if concept is None:
                details = {
                    "concept_id": args.rationale_concept,
                    "current_status": None,
                    "required_status": "active",
                }
            else:
                details = {
                    "concept_id": args.rationale_concept,
                    "current_status": concept["status"],
                    "required_status": "active",
                }
        _err(code, str(e), **details)
        return 1

    _ok(result)
    return 0


def cmd_contradiction_get(args, store: Store) -> int:
    """Get a contradiction with both claims and disposition."""
    crow = store.conn.execute(
        "SELECT * FROM contradictions WHERE id = ?", (args.contradiction_id,)
    ).fetchone()
    if not crow:
        _err("contradiction_not_found",
             f"no contradiction with id {args.contradiction_id}",
             contradiction_id=args.contradiction_id)
        return 1

    a = store.get_claim(crow["claim_a_id"])
    b = store.get_claim(crow["claim_b_id"])

    # get rule if any
    rule_row = store.conn.execute(
        "SELECT * FROM contradiction_rules WHERE contradiction_id = ?",
        (args.contradiction_id,),
    ).fetchone()

    out = {
        "contradiction_id": crow["id"],
        "status": crow["status"],
        "kind": crow["kind"] if "kind" in crow.keys() else None,
        "disposition": crow["disposition"] if "disposition" in crow.keys() else None,
        "candidate_disposition": crow["candidate_disposition"] if "candidate_disposition" in crow.keys() else None,
        "cross_subject": bool(crow["cross_subject"]) if "cross_subject" in crow.keys() else False,
        "relation_kind": crow["relation_kind"] if "relation_kind" in crow.keys() else None,
        "justification": crow["justification"] if "justification" in crow.keys() else None,
        "claim_a": {
            "id": a["id"], "subject": a["subject"],
            "predicate": a["predicate"], "object": a["object"],
            "span_text": store.get_span_text(a["id"]) or "",
        } if a else None,
        "claim_b": {
            "id": b["id"], "subject": b["subject"],
            "predicate": b["predicate"], "object": b["object"],
            "span_text": store.get_span_text(b["id"]) or "",
        } if b else None,
    }
    if rule_row:
        out["rule"] = {
            "rule": rule_row["rule"],
            "applies_when": json.loads(rule_row["applies_when"]) if rule_row["applies_when"] else None,
            "rationale_concept_id": rule_row["rationale_concept_id"],
            "decided_at": rule_row["decided_at"],
        }
    _ok(out)
    return 0


def cmd_contradiction_rule_get(args, store: Store) -> int:
    """Get the contradiction_rules row for a contradiction."""
    crow = store.conn.execute(
        "SELECT * FROM contradictions WHERE id = ?", (args.contradiction_id,)
    ).fetchone()
    if not crow:
        _err("contradiction_not_found",
             f"no contradiction with id {args.contradiction_id}",
             contradiction_id=args.contradiction_id)
        return 1

    rule_row = store.conn.execute(
        "SELECT * FROM contradiction_rules WHERE contradiction_id = ?",
        (args.contradiction_id,),
    ).fetchone()
    if not rule_row:
        _ok({"contradiction_id": args.contradiction_id, "rule": None})
        return 0

    _ok({
        "contradiction_id": args.contradiction_id,
        "rule": rule_row["rule"],
        "applies_when": json.loads(rule_row["applies_when"]) if rule_row["applies_when"] else None,
        "rationale_concept_id": rule_row["rationale_concept_id"],
        "decided_at": rule_row["decided_at"],
    })
    return 0


# ------------- WS-C SOURCE METADATA (from docs/plan/30-ws-authority.md) ----


def cmd_source_authority_set(args, store: Store) -> int:
    """Set domain metadata for a source. Validates fields, persists unless
    --strict and validation errors exist."""
    from . import authority

    try:
        metadata = json.loads(args.metadata)
    except json.JSONDecodeError as e:
        _err("invalid_json", f"cannot parse --metadata as JSON: {e}")
        return 1

    if not isinstance(metadata, dict):
        _err("invalid_metadata", "metadata must be a JSON object")
        return 1

    src = store.get_source(args.source_id)
    if not src:
        _err("source_not_found", f"no source with id {args.source_id}",
             source_id=args.source_id)
        return 1

    errors = authority.validate_metadata(args.domain, metadata)

    # Domain validation failure is always fatal (Store would raise)
    domain_error = any(e.field == "domain" for e in errors)
    if domain_error:
        _err("validation_failed", "unknown domain",
             errors=[{"field": e.field, "reason": e.reason} for e in errors])
        return 1

    if errors and args.strict:
        _err("validation_failed", "metadata validation failed with --strict",
             errors=[{"field": e.field, "reason": e.reason} for e in errors])
        return 1

    result = authority.set_metadata(store, args.source_id, args.domain, metadata)
    if "error" in result:
        _err(result["error"], result["message"], source_id=args.source_id)
        return 1
    _ok(result)
    return 0


def cmd_source_authority_get(args, store: Store) -> int:
    """Get source metadata with computed authority_rank."""
    from . import authority

    src = store.get_source(args.source_id)
    if not src:
        _err("source_not_found", f"no source with id {args.source_id}",
             source_id=args.source_id)
        return 1

    meta = store.get_source_metadata(args.source_id)
    if meta is None:
        _ok({
            "source_id": args.source_id,
            "path": src["path"],
            "ingested_at": src["ingested_at"],
            "metadata": None,
        })
        return 0

    rank = authority.authority_rank(meta)
    _ok({
        "source_id": args.source_id,
        "path": src["path"],
        "ingested_at": src["ingested_at"],
        "domain": meta["domain"],
        "metadata": meta["metadata"],
        "authority_rank": list(rank),
        "warnings": authority.provenance_warnings(meta["metadata"]),
    })
    return 0


def cmd_source_list_by_domain(args, store: Store) -> int:
    """List sources in a domain."""
    rows = store.list_sources_by_domain(args.domain)
    out = []
    for r in rows:
        out.append({
            "source_id": r["source_id"],
            "domain": r["domain"],
            "path": r["path"],
            "ingested_at": r["ingested_at"],
        })
    _ok({"domain": args.domain, "sources": out})
    return 0


def cmd_source_retract(args, store: Store) -> int:
    """Retract a scientific source. Cascades to claims, concepts, views."""
    from . import authority

    src = store.get_source(args.source_id)
    if not src:
        _err("source_not_found", f"no source with id {args.source_id}",
             source_id=args.source_id)
        return 1

    result = authority.retract_source(store, args.source_id, args.reason)
    if "error" in result:
        _err(result["error"],
             f"cannot retract source {args.source_id}: {result['error']}",
             source_id=args.source_id)
        return 1
    _ok(result)
    return 0


def cmd_source_unretract(args, store: Store) -> int:
    """Reverse a retraction on a scientific source."""
    from . import authority

    src = store.get_source(args.source_id)
    if not src:
        _err("source_not_found", f"no source with id {args.source_id}",
             source_id=args.source_id)
        return 1

    result = authority.unretract_source(store, args.source_id, args.reason)
    if "error" in result:
        _err(result["error"],
             f"cannot unretract source {args.source_id}: {result['error']}",
             source_id=args.source_id)
        return 1
    _ok(result)
    return 0


# ------------- WS-D CONDITIONS (from docs/plan/40-ws-conditions.md) --------


def cmd_claim_condition_add(args, store: Store) -> int:
    """Link a condition claim to a claim."""
    from . import conditions

    if args.claim == args.condition:
        _err("invalid_condition",
             "condition cannot reference the claim itself")
        return 1
    if args.kind not in conditions.VALID_KINDS:
        _err("invalid_condition",
             f"invalid kind {args.kind!r}",
             valid_kinds=sorted(conditions.VALID_KINDS))
        return 1
    if not store.get_claim(args.claim):
        _err("claim_not_found",
             f"no claim with id {args.claim}",
             claim_id=args.claim)
        return 1
    if not store.get_claim(args.condition):
        _err("claim_not_found",
             f"no claim with id {args.condition}",
             claim_id=args.condition)
        return 1
    conditions.link_condition(
        store, args.claim, args.condition, args.kind,
        explicit=args.explicit, confidence=args.confidence,
    )
    _ok({
        "claim_id": args.claim,
        "condition_claim_id": args.condition,
        "kind": args.kind,
        "explicit": args.explicit,
        "confidence": args.confidence,
    })
    return 0


def cmd_claim_condition_remove(args, store: Store) -> int:
    """Remove a condition link."""
    from . import conditions
    removed = conditions.unlink_condition(
        store, args.claim, args.condition,
    )
    _ok({"removed": removed})
    return 0


def cmd_claim_conditions_list(args, store: Store) -> int:
    """List conditions for a claim."""
    from . import conditions
    conds = conditions.conditions_for_claim(store, args.claim_id)
    _ok({"claim_id": args.claim_id, "conditions": conds})
    return 0


def cmd_claim_condition_extract(args, store: Store) -> int:
    """Infer conditions for a claim (LLM-calling)."""
    llm, err = _get_llm_for_agent(args)
    if err is not None:
        _err("missing_api_key", "ANTHROPIC_API_KEY not set")
        return 2
    if not store.get_claim(args.claim_id):
        _err("claim_not_found",
             f"no claim with id {args.claim_id}",
             claim_id=args.claim_id)
        return 1
    from . import conditions
    linked = conditions.infer_conditions(
        store, llm, args.claim_id,
        same_source_only=args.same_source_only,
    )
    _ok({
        "claim_id": args.claim_id,
        "linked_condition_ids": linked,
    })
    return 0


# ------------- P2.4 PROVENANCE CHAIN ---------------------------------------


def cmd_provenance(args, store: Store) -> int:
    """Walk the provenance chain for a claim in one call (P2.4).

    ``claim -> source -> authority -> fetch_method/provenance_notes`` plus
    conditions, concepts citing this claim, and contradictions involving
    it. Useful when a reviewer wants to audit a specific citation without
    stitching together multiple other commands.
    """
    claim = store.get_claim(args.claim_id)
    if not claim:
        _err("claim_not_found", f"no claim with id {args.claim_id}",
             claim_id=args.claim_id)
        return 1

    src = store.get_source(claim["source_id"])
    span_text = store.get_span_text(args.claim_id) or ""

    # Source block with authority metadata + provenance warnings
    from . import authority as _authority
    meta = store.get_source_metadata(claim["source_id"])
    source_block: dict = {
        "source_id": claim["source_id"],
        "path": src["path"] if src else None,
        "sha256": src["sha256"] if src else None,
        "ingested_at": src["ingested_at"] if src else None,
    }
    if meta is not None:
        inner = meta["metadata"]
        source_block["authority"] = {
            "domain": meta["domain"],
            "authority_rank": list(_authority.authority_rank(meta)),
            **inner,
        }
        source_block["fetch_method"] = inner.get("fetch_method")
        source_block["provenance_notes"] = inner.get("provenance_notes")
        source_block["provenance_warnings"] = _authority.provenance_warnings(inner)
    else:
        source_block["authority"] = None
        source_block["fetch_method"] = None
        source_block["provenance_notes"] = None
        source_block["provenance_warnings"] = []

    # Conditions (claim_conditions where this claim is the target)
    cond_rows = store.get_claim_conditions(args.claim_id)
    conditions_out = [
        {
            "condition_claim_id": r["condition_claim_id"],
            "kind": r["kind"],
            "explicit": bool(r["explicit"]),
            "confidence": r["confidence"],
        }
        for r in cond_rows
    ]

    # Concepts citing this claim
    concept_rows = store.conn.execute(
        "SELECT concept_id, role FROM concept_supports WHERE claim_id = ?",
        (args.claim_id,),
    ).fetchall()
    concepts_out = [
        {"concept_id": r["concept_id"], "role": r["role"]}
        for r in concept_rows
    ]

    # Contradictions involving this claim
    contra_rows = store.conn.execute(
        "SELECT id, claim_a_id, claim_b_id, status, disposition, "
        "  kind, cross_subject, relation_kind "
        "FROM contradictions WHERE claim_a_id = ? OR claim_b_id = ? "
        "ORDER BY detected_at DESC",
        (args.claim_id, args.claim_id),
    ).fetchall()
    contras_out = [
        {
            "contradiction_id": r["id"],
            "other_claim_id": (
                r["claim_b_id"] if r["claim_a_id"] == args.claim_id
                else r["claim_a_id"]
            ),
            "status": r["status"],
            "disposition": r["disposition"],
            "kind": r["kind"],
            "cross_subject": bool(r["cross_subject"] or 0),
            "relation_kind": r["relation_kind"],
        }
        for r in contra_rows
    ]

    # WS-E.2: surface a predicate_sense block when the claim has a sense.
    sense_row = store.get_claim_predicate_sense(args.claim_id)
    sense_block = None
    if sense_row is not None:
        sense_block = {
            "sense_id": sense_row["sense_id"],
            "canonical": sense_row["canonical"],
            "sense_tag": sense_row["sense_tag"],
            "domain": sense_row["domain"],
            "assigned_by": sense_row["assigned_by"],
            "explicit": bool(sense_row["explicit"]),
            "confidence": sense_row["confidence"],
            "rationale": sense_row["rationale"],
        }

    payload = {
        "claim_id": claim["id"],
        "subject": claim["subject"],
        "predicate": claim["predicate"],
        "object": claim["object"],
        "confidence": claim["confidence"],
        "status": claim["status"],
        "span_start": claim["span_start"],
        "span_end": claim["span_end"],
        "span_text": span_text,
        "source": source_block,
        "conditions": conditions_out,
        "concepts_citing_this_claim": concepts_out,
        "contradictions_involving_this_claim": contras_out,
    }
    if sense_block is not None:
        payload["predicate_sense"] = sense_block
    _ok(payload)
    return 0


# ------------- P2.5 COMPOSE (agent-mode retrieval brief) -------------------


def cmd_compose(args, store: Store) -> int:
    """Retrieve the structured evidence brief an agent needs to write an
    answer, without calling the LLM (P2.5).

    This is the agent-mode counterpart to ``aleph ask``. It runs the same
    deterministic prefix of the pipeline — keyword extraction, retrieval,
    context filter, concept retrieval, disposition grouping — and emits
    the result as JSON. The agent composes the prose and cites IDs
    itself; no synthesis LLM, no per-sentence verifier.

    The output includes "gaps": subjects that appear in the retrieved
    claims but have no active (or attested) concept, so the agent knows
    where concept derivation would help.
    """
    from .query import (
        _expand_evidence,
        _extract_keywords,
        _filter_claims_by_context,
        _group_by_disposition,
        _search,
        _retrieve_concepts,
    )
    from .retrieval import default_retriever

    ctx: Optional[dict] = None
    if args.context:
        try:
            ctx = json.loads(args.context)
        except json.JSONDecodeError as e:
            _err("invalid_context", f"--context is not valid JSON: {e}")
            return 1
        # Allow ISO date strings too, just like the API-mode `ask` command.
        if isinstance(ctx.get("date"), str):
            import datetime
            try:
                ctx["date"] = datetime.datetime.fromisoformat(
                    ctx["date"]
                ).timestamp()
            except ValueError as e:
                _err("invalid_context_date",
                     f"context.date is not ISO parseable: {e}")
                return 1

    keywords = _extract_keywords(args.query)
    retriever = default_retriever(store)
    claim_rows = _search(retriever, keywords, args.k, ctx)
    claim_rows = _filter_claims_by_context(store, claim_rows, ctx)
    # Same expansion as ask: claims conflicting with, or conditioning,
    # retrieved claims, each tagged with why it was included.
    included_because: dict[int, str] = {}
    if not args.no_expand:
        claim_rows, included_because = _expand_evidence(store, claim_rows, ctx, args.k // 2)

    # Concept retrieval: active + attested both count as citeable.
    concept_k = args.concept_k if args.concept_k is not None else max(5, args.k // 3)
    active_concepts = _retrieve_concepts(store, keywords, limit=concept_k)
    # _retrieve_concepts only returns active concepts; surface attested ones
    # separately so the agent can see the two tiers.
    attested_rows: list = []
    if keywords:
        like_parts = []
        params: list = []
        for kw in (k.lower() for k in keywords if len(k) > 2):
            like_parts.append("(LOWER(subject) LIKE ? OR LOWER(statement) LIKE ?)")
            params.extend([f"%{kw}%", f"%{kw}%"])
        if like_parts:
            where = " OR ".join(like_parts)
            params.append(concept_k)
            attested_rows = store.conn.execute(
                f"SELECT * FROM concepts WHERE status = 'attested' AND ({where}) "
                f"ORDER BY confidence DESC, attested_at DESC LIMIT ?",
                params,
            ).fetchall()

    groups = _group_by_disposition(store, claim_rows)

    # Gaps: subjects present in retrieved claims but with no citeable concept
    retrieved_subjects = {r["subject"] for r in claim_rows}
    concept_subjects = {c["subject"] for c in active_concepts}
    concept_subjects.update(r["subject"] for r in attested_rows)
    gaps = sorted(retrieved_subjects - concept_subjects)

    # WS-E.2: bucket retrieved claims by their predicate sense (or null).
    grouped_by_sense: dict = {}
    null_bucket: list[int] = []
    for r in claim_rows:
        srow = store.get_claim_predicate_sense(r["id"])
        if srow is None:
            null_bucket.append(r["id"])
            continue
        key = srow["sense_id"]
        bucket = grouped_by_sense.setdefault(key, {
            "sense_id": srow["sense_id"],
            "canonical": srow["canonical"],
            "sense_tag": srow["sense_tag"],
            "domain": srow["domain"],
            "claim_ids": [],
        })
        bucket["claim_ids"].append(r["id"])
    grouped_by_sense_out = list(grouped_by_sense.values())
    if null_bucket:
        grouped_by_sense_out.append({
            "sense_id": None, "canonical": None, "sense_tag": None,
            "domain": None, "claim_ids": null_bucket,
        })

    _ok({
        "query": args.query,
        "keywords": keywords,
        "context": ctx,
        "retrieved_claims": [
            {
                "claim_id": r["id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "confidence": r["confidence"],
                "source_id": r["source_id"],
                "span_text": (r["span_text"] if "span_text" in r.keys() else None) or "",
                "included_because": included_because.get(r["id"]),
            }
            for r in claim_rows
        ],
        "active_concepts": [
            {
                "concept_id": c["id"],
                "subject": c["subject"],
                "statement": c["statement"],
                "inference_type": c["inference_type"],
                "confidence": c["confidence"],
            }
            for c in active_concepts
        ],
        "attested_concepts": [
            {
                "concept_id": r["id"],
                "subject": r["subject"],
                "statement": r["statement"],
                "attested_by": r["attested_by"] if "attested_by" in r.keys() else None,
                "attestation_rationale": (
                    r["attestation_rationale"] if "attestation_rationale" in r.keys() else None
                ),
            }
            for r in attested_rows
        ],
        "dispositions": {
            "replicate": [sorted(s) if isinstance(s, (list, set)) else s for s in groups["replications"]],
            "reconcile": [
                {"claim_a": a, "claim_b": b, "rule": rule, "rationale_concept_id": cid}
                for a, b, rule, cid in groups["reconciled"]
            ],
            "coexist": [
                {"claim_a": a, "claim_b": b, "rule": rule, "applies_when": aw}
                for a, b, rule, aw in groups["coexisting"]
            ],
            "distinguish": [
                {"claim_a": a, "claim_b": b, "rule": rule}
                for a, b, rule in groups["distinguished"]
            ],
            "dispute": [
                {"claim_a": a, "claim_b": b} for a, b in groups["disputed"]
            ],
            "gap": [
                {"claim_a": a, "claim_b": b} for a, b in groups["gaps"]
            ],
            "unresolved": [
                {"claim_a": a, "claim_b": b} for a, b in groups["unresolved"]
            ],
        },
        "gaps_without_concept": gaps,
        "grouped_by_sense": grouped_by_sense_out,
    })
    return 0


# ------------- P2.2 REPORT (one-shot diagnostics snapshot) -----------------


# ------------- WS-E.1 PREDICATE ALIASES (from docs/plan/70-ws-predicates.md) -


def cmd_predicate_alias_add(args, store: Store) -> int:
    domain = args.domain
    valid = store._VALID_DOMAINS | {"*"}
    if domain not in valid:
        _err("invalid_domain",
             f"--domain must be one of {sorted(valid)}", domain=domain)
        return 1
    try:
        result = store.add_predicate_alias(
            domain, args.from_predicate, args.to_predicate,
        )
    except ValueError as e:
        _err("invalid_predicate_alias", str(e))
        return 1
    if result.get("note") == "would-create-cycle":
        _err("alias_would_create_cycle",
             "alias would create a resolution cycle in this domain",
             **{k: v for k, v in result.items() if k != "note"})
        return 1
    _ok(result)
    return 0


def cmd_predicate_alias_list(args, store: Store) -> int:
    rows = store.list_predicate_aliases(domain=args.domain)
    _ok({
        "aliases": [
            {
                "domain": r["domain"],
                "from": r["alias_from"],
                "to": r["canonical_to"],
                "created_at": r["created_at"],
            }
            for r in rows
        ],
    })
    return 0


def cmd_predicate_alias_remove(args, store: Store) -> int:
    result = store.remove_predicate_alias(args.domain, args.from_predicate)
    _ok(result)
    return 0


# ------------- WS-E.2 PREDICATE SENSES (from docs/plan/71-ws-predicate-senses.md) -


def cmd_predicate_sense_add(args, store: Store) -> int:
    try:
        sense_id = store.add_predicate_sense(
            canonical=args.canonical,
            sense_tag=args.sense_tag,
            domain=args.domain,
            definition=args.definition,
            parent_id=args.parent_id,
            inverse_id=args.inverse_id,
            is_symmetric=bool(args.is_symmetric),
            is_transitive=bool(args.is_transitive),
        )
    except ValueError as e:
        msg = str(e)
        if msg.startswith("duplicate_sense"):
            _err("duplicate_sense",
                 "(canonical, sense_tag, domain) triple already exists",
                 canonical=args.canonical,
                 sense_tag=args.sense_tag, domain=args.domain)
            return 1
        _err("invalid_predicate_sense", msg)
        return 1
    row = store.get_predicate_sense(sense_id)
    _ok({
        "sense_id": sense_id,
        "canonical": row["canonical"],
        "sense_tag": row["sense_tag"],
        "domain": row["domain"],
        "definition": row["definition"],
        "parent_id": row["parent_id"],
        "inverse_id": row["inverse_id"],
        "is_symmetric": bool(row["is_symmetric"]),
        "is_transitive": bool(row["is_transitive"]),
    })
    return 0


def cmd_predicate_sense_list(args, store: Store) -> int:
    rows = store.list_predicate_senses(
        canonical=args.canonical, domain=args.domain,
    )
    _ok({
        "senses": [
            {
                "sense_id": r["id"],
                "canonical": r["canonical"],
                "sense_tag": r["sense_tag"],
                "domain": r["domain"],
                "definition": r["definition"],
                "parent_id": r["parent_id"],
                "inverse_id": r["inverse_id"],
                "is_symmetric": bool(r["is_symmetric"]),
                "is_transitive": bool(r["is_transitive"]),
            }
            for r in rows
        ],
    })
    return 0


def cmd_predicate_sense_get(args, store: Store) -> int:
    row = store.get_predicate_sense(args.sense_id)
    if not row:
        _err("sense_not_found",
             f"no sense with id {args.sense_id}",
             sense_id=args.sense_id)
        return 1
    claim_count = store.conn.execute(
        "SELECT COUNT(*) AS n FROM claim_predicate_senses WHERE sense_id = ?",
        (args.sense_id,),
    ).fetchone()["n"]
    _ok({
        "sense_id": row["id"],
        "canonical": row["canonical"],
        "sense_tag": row["sense_tag"],
        "domain": row["domain"],
        "definition": row["definition"],
        "parent_id": row["parent_id"],
        "inverse_id": row["inverse_id"],
        "is_symmetric": bool(row["is_symmetric"]),
        "is_transitive": bool(row["is_transitive"]),
        "claim_count": claim_count,
    })
    return 0


def cmd_claim_predicate_sense_set(args, store: Store) -> int:
    try:
        store.set_claim_predicate_sense(
            args.claim_id, args.sense,
            assigned_by=args.assigned_by,
            confidence=args.confidence,
            explicit=bool(args.explicit),
            rationale=args.rationale,
        )
    except ValueError as e:
        msg = str(e)
        if msg == "would_silently_demote":
            _err("would_silently_demote",
                 "existing row is explicit=1; unset first",
                 claim_id=args.claim_id)
            return 1
        _err("invalid_predicate_sense", msg)
        return 1
    row = store.get_claim_predicate_sense(args.claim_id)
    _ok({
        "claim_id": args.claim_id,
        "sense_id": row["sense_id"],
        "assigned_by": row["assigned_by"],
        "explicit": bool(row["explicit"]),
        "confidence": row["confidence"],
        "rationale": row["rationale"],
    })
    return 0


def cmd_claim_predicate_sense_unset(args, store: Store) -> int:
    removed = store.unset_claim_predicate_sense(args.claim_id)
    _ok({"unset": removed})
    return 0


def cmd_predicate_sense_extract(args, store: Store) -> int:
    from . import predicate_senses as _ps
    llm, err = _get_llm_for_agent(args)
    if err is not None:
        _err("missing_api_key",
             "ANTHROPIC_API_KEY not set; use --mock-llm or ALEPH_LLM_FIXTURES")
        return 1
    if llm is None:
        return 1
    # check explicit-locked first to honour the contract before calling LLM
    existing = store.conn.execute(
        "SELECT explicit FROM claim_predicate_senses WHERE claim_id = ?",
        (args.claim_id,),
    ).fetchone()
    if existing is not None and existing["explicit"]:
        _ok({
            "claim_id": args.claim_id,
            "skipped": True,
            "reason": "existing explicit assignment",
        })
        return 0
    result = _ps.infer_sense(store, llm, args.claim_id)
    if result is None:
        _ok({
            "claim_id": args.claim_id,
            "sense_id": None,
            "confidence": 0.0,
            "rationale": "",
            "explicit": False,
            "assigned_by": "llm",
            "skipped": True,
            "reason": "no candidate or unparseable response",
        })
        return 0
    sense_id, confidence, rationale = result
    if sense_id == "explicit_locked":
        _ok({
            "claim_id": args.claim_id,
            "skipped": True,
            "reason": "existing explicit assignment",
        })
        return 0
    _ok({
        "claim_id": args.claim_id,
        "sense_id": sense_id,
        "confidence": confidence,
        "rationale": rationale,
        "explicit": False,
        "assigned_by": "llm",
    })
    return 0


def cmd_report_agent(args, store: Store) -> int:
    """JSON dashboard of the store's state (P2.2).

    Composes counts/top-lists from existing read paths so an operator can
    eyeball the store without running half a dozen separate commands. The
    ``aleph report`` top-level command (API-mode) reuses this and also
    offers text/markdown formatting.
    """
    cx = store.conn

    # Source counts by authority_type + authority_level bucket (legal-style)
    domain_counts = dict(cx.execute(
        "SELECT domain, COUNT(*) FROM source_metadata GROUP BY domain"
    ).fetchall())
    authority_types: dict = {}
    for r in cx.execute(
        "SELECT metadata FROM source_metadata WHERE domain = 'legal'"
    ).fetchall():
        try:
            m = json.loads(r["metadata"])
        except (json.JSONDecodeError, TypeError):
            continue
        at = m.get("authority_type")
        if at:
            authority_types[at] = authority_types.get(at, 0) + 1

    # Claim counts by subject (top 20), by status
    top_subjects = [
        {"subject": r["subject"], "count": r["n"]}
        for r in cx.execute(
            "SELECT subject, COUNT(*) AS n FROM claims "
            "WHERE status = 'active' GROUP BY subject ORDER BY n DESC LIMIT 20"
        ).fetchall()
    ]
    claims_by_status = dict(cx.execute(
        "SELECT status, COUNT(*) FROM claims GROUP BY status"
    ).fetchall())

    # Concepts by status / by subject
    concepts_by_status = dict(cx.execute(
        "SELECT status, COUNT(*) FROM concepts GROUP BY status"
    ).fetchall())
    concepts_by_subject = [
        {"subject": r["subject"], "count": r["n"]}
        for r in cx.execute(
            "SELECT subject, COUNT(*) AS n FROM concepts "
            "WHERE status IN ('active', 'attested') GROUP BY subject "
            "ORDER BY n DESC LIMIT 20"
        ).fetchall()
    ]

    # Contradictions by disposition and open/resolved
    by_disposition = dict(cx.execute(
        "SELECT disposition, COUNT(*) FROM contradictions GROUP BY disposition"
    ).fetchall())
    open_count = cx.execute(
        "SELECT COUNT(*) FROM contradictions WHERE status = 'open'"
    ).fetchone()[0]
    resolved_count = cx.execute(
        "SELECT COUNT(*) FROM contradictions WHERE status = 'resolved'"
    ).fetchone()[0]
    cross_count = cx.execute(
        "SELECT COUNT(*) FROM contradictions WHERE cross_subject = 1"
    ).fetchone()[0]

    # Aliases and top-5 rewriters (approximation: canonical_to counts)
    alias_total = cx.execute(
        "SELECT COUNT(*) FROM subject_aliases"
    ).fetchone()[0]
    top_alias_targets = [
        {"canonical_to": r["canonical_to"], "alias_count": r["n"]}
        for r in cx.execute(
            "SELECT canonical_to, COUNT(*) AS n FROM subject_aliases "
            "GROUP BY canonical_to ORDER BY n DESC LIMIT 5"
        ).fetchall()
    ]

    # Cache state
    cache_count = cx.execute("SELECT COUNT(*) FROM view_cache").fetchone()[0]

    # Gateway hints: subjects without supporting concepts; pairs with same
    # subject + differing predicates (contradiction-scan candidates)
    no_concept_subjects = [
        r["subject"] for r in cx.execute(
            "SELECT subject, COUNT(*) AS n FROM claims "
            "WHERE status = 'active' "
            "AND subject NOT IN (SELECT subject FROM concepts "
            "                    WHERE status IN ('active', 'attested')) "
            "GROUP BY subject ORDER BY n DESC LIMIT 10"
        ).fetchall()
    ]
    scan_candidates_raw = cx.execute(
        "SELECT a.id AS a_id, b.id AS b_id, a.subject AS subject, "
        "  a.predicate AS a_pred, b.predicate AS b_pred, "
        "  a.object AS a_obj, b.object AS b_obj "
        "FROM claims a JOIN claims b ON a.subject = b.subject AND a.id < b.id "
        "WHERE a.status = 'active' AND b.status = 'active' "
        "  AND a.predicate != b.predicate "
        "LIMIT 10"
    ).fetchall()
    scan_candidates = [
        {
            "subject": r["subject"],
            "claim_a": {"id": r["a_id"], "predicate": r["a_pred"], "object": r["a_obj"]},
            "claim_b": {"id": r["b_id"], "predicate": r["b_pred"], "object": r["b_obj"]},
        }
        for r in scan_candidates_raw
    ]

    # WS-E.2: list (canonical, domain) pairs with >= 3 active claims but zero
    # sense assignments — triage signal for which predicates need a sense
    # catalog populated.
    predicates_without_senses = [
        {
            "predicate": r["predicate"],
            "domain": r["domain"],
            "claim_count": r["n"],
        }
        for r in cx.execute(
            "SELECT c.predicate AS predicate, sm.domain AS domain, "
            "  COUNT(*) AS n "
            "FROM claims c "
            "JOIN source_metadata sm ON sm.source_id = c.source_id "
            "WHERE c.status = 'active' "
            "  AND NOT EXISTS ("
            "    SELECT 1 FROM claim_predicate_senses cps "
            "    WHERE cps.claim_id = c.id"
            "  ) "
            "GROUP BY c.predicate, sm.domain "
            "HAVING n >= 3 "
            "ORDER BY n DESC LIMIT 10"
        ).fetchall()
    ]

    _ok({
        "stats": store.stats(),
        "sources": {
            "by_domain": domain_counts,
            "legal_by_authority_type": authority_types,
        },
        "claims": {
            "by_status": claims_by_status,
            "top_subjects": top_subjects,
        },
        "concepts": {
            "by_status": concepts_by_status,
            "by_subject": concepts_by_subject,
        },
        "contradictions": {
            "open": open_count,
            "resolved": resolved_count,
            "cross_subject": cross_count,
            "by_disposition": by_disposition,
        },
        "aliases": {
            "total": alias_total,
            "top_targets": top_alias_targets,
        },
        "cache": {"cached_views": cache_count},
        "hints": {
            "subjects_without_concepts": no_concept_subjects,
            "contradiction_scan_candidates": scan_candidates,
            "predicates_without_senses": predicates_without_senses,
        },
    })
    return 0


def register_agent_commands(subparsers, common) -> set[str]:
    """Register all agent-facing commands on a subparser namespace.

    Returns the set of registered command names so the top-level dispatcher
    can route agent-mode commands without a hand-maintained list.
    """
    registered: set[str] = set()

    def _add(name: str, **kwargs):
        registered.add(name)
        return subparsers.add_parser(name, parents=[common], **kwargs)

    p = _add("source-add", help="register a file as a source")
    p.add_argument("path")
    p.set_defaults(func=cmd_source_add)

    p = _add("source-get", help="get a source (with content)")
    p.add_argument("source_id", type=int)
    p.add_argument("--no-content", action="store_true")
    p.set_defaults(func=cmd_source_get)

    p = _add("source-list", help="list sources")
    p.set_defaults(func=cmd_source_list)

    p = _add("source-remove", help="remove a source")
    p.add_argument("source_id", type=int)
    p.set_defaults(func=cmd_source_remove)

    p = _add("source-replace",
             help="atomically remove and re-add a source at the given path")
    p.add_argument("path")
    p.set_defaults(func=cmd_source_replace)

    p = _add("source-yield",
             help="per-source claim density (claims/KB) for triage")
    p.add_argument("--thin-threshold", type=float, default=1.0,
                   help="flag sources with claims_per_kb below this value (default 1.0)")
    p.set_defaults(func=cmd_source_yield)

    p = _add("claim-add",
             help="add a claim (span must be verbatim substring of source)")
    p.add_argument("--source-id", type=int, required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--predicate", required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--span", required=True, help="verbatim substring of the source")
    p.add_argument("--confidence", type=float, default=0.8)
    # WS-D: optional --conditions flag
    p.add_argument("--conditions", default=None,
                   help="comma-separated condition_claim_id:kind pairs")
    p.add_argument("--proposition", default=None,
                   help="the claim as one self-contained sentence; the triple "
                        "stays as its index. Checked for fidelity with the triple's "
                        "fallback when omitted")
    p.set_defaults(func=cmd_claim_add)

    p = _add("claim-fidelity-check",
             help="check claims' numbers/dates/units/negations/entities against "
                  "their span and context")
    p.add_argument("claim_id", type=int, nargs="?", default=None)
    p.add_argument("--all", action="store_true", help="check every active claim")
    p.add_argument("--enqueue", action="store_true",
                   help="queue flagged claims for review")
    p.set_defaults(func=cmd_claim_fidelity_check)

    p = _add("counter-evidence",
             help="uncited sides of live conflicts (open, dispute, gap) with the given claims")
    p.add_argument("--claim-ids", required=True, help="comma-separated claim ids an answer cites")
    p.set_defaults(func=cmd_counter_evidence)

    p = _add("review-list", help="list review-queue items (default: open)")
    p.add_argument("--status", default="open",
                   choices=["open", "accepted", "rejected", "obsolete", "all"])
    p.add_argument("--type", default=None,
                   choices=["claim", "contradiction", "concept", "alias"])
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(func=cmd_review_list)

    p = _add("review-add", help="queue a claim, contradiction, concept or alias for review")
    p.add_argument("--type", required=True,
                   choices=["claim", "contradiction", "concept", "alias"])
    p.add_argument("--id", type=int, default=None,
                   help="claim / contradiction / concept id")
    p.add_argument("--alias", default=None, help="the alias's FROM subject")
    p.add_argument("--reason", default="manual")
    p.add_argument("--note", default=None)
    p.set_defaults(func=cmd_review_add)

    p = _add("review-resolve",
             help="record accepted/rejected on an open review item (changes nothing else)")
    p.add_argument("review_id", type=int)
    p.add_argument("--decision", required=True, help="accepted | rejected")
    p.add_argument("--by", required=True, help="who decided (attestor identifier)")
    p.add_argument("--note", default=None)
    p.set_defaults(func=cmd_review_resolve)

    p = _add("claim-get", help="get a claim with its span text")
    p.add_argument("claim_id", type=int)
    p.set_defaults(func=cmd_claim_get)

    p = _add("claim-search", help="keyword search over claims")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=30)
    p.add_argument("--with-spans", action="store_true")
    p.add_argument("--compact", action="store_true",
                   help="emit id + predicate + truncated object + confidence only")
    p.add_argument("--fields", default=None,
                   help="comma-separated output fields (e.g. 'claim_id,predicate,confidence')")
    p.set_defaults(func=cmd_claim_search)

    p = _add("claim-by-subject", help="list claims for a subject (alias-resolved)")
    p.add_argument("subject")
    p.add_argument("--compact", action="store_true",
                   help="emit id + predicate + truncated object + confidence only")
    p.add_argument("--fields", default=None,
                   help="comma-separated output fields (e.g. 'claim_id,predicate,confidence')")
    p.set_defaults(func=cmd_claim_by_subject)

    p = _add("claim-supersede", help="mark claim as superseded by another")
    p.add_argument("old_id", type=int)
    p.add_argument("new_id", type=int)
    p.set_defaults(func=cmd_claim_supersede)

    p = _add("subjects", help="list distinct subjects by claim count")
    p.set_defaults(func=cmd_subjects)

    p = _add("alias-add",
             help="declare FROM subject is the same as TO; rewrites existing claims")
    p.add_argument("from_subject", metavar="FROM")
    p.add_argument("to_subject", metavar="TO")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing alias with a different target (logged)")
    p.set_defaults(func=cmd_alias_add)

    p = _add("alias-undo",
             help="undo the latest merge of FROM: restore rewritten claims' subjects")
    p.add_argument("from_subject", metavar="FROM")
    p.set_defaults(func=cmd_alias_undo)

    p = _add("alias-log", help="list alias merge events (including undone ones)")
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(func=cmd_alias_log)

    p = _add("invariant-check",
             help="verify no current view/concept/resolution rests on inactive claims")
    p.set_defaults(func=cmd_invariant_check)

    p = _add("alias-list", help="list subject aliases")
    p.set_defaults(func=cmd_alias_list)

    p = _add("contradiction-add", help="record a contradiction between two claims")
    # Accept either positional (legacy: `contradiction-add A B`) or flag
    # form (`--claim-a A --claim-b B`). Both are optional; handler validates
    # that one form was used.
    p.add_argument("claim_a", type=int, nargs="?", default=None)
    p.add_argument("claim_b", type=int, nargs="?", default=None)
    p.add_argument("--claim-a", dest="claim_a_flag", type=int, default=None,
                   help="claim id A (alternative to first positional)")
    p.add_argument("--claim-b", dest="claim_b_flag", type=int, default=None,
                   help="claim id B (alternative to second positional)")
    # P1.6: escape valve for cross-subject doctrinal tensions. All three
    # flags must be provided together.
    p.add_argument("--cross-subject", dest="cross_subject", action="store_true",
                   default=False,
                   help="bypass the same-subject check; requires --relation-kind and --justification")
    p.add_argument("--relation-kind", dest="relation_kind", default=None,
                   help="kind of cross-subject relation: "
                        "regime-supersedes | rule-limits-rule | doctrinal-cross-ref")
    p.add_argument("--justification", dest="justification", default=None,
                   help="required free-text explanation for the cross-subject pairing")
    p.set_defaults(func=cmd_contradiction_add)

    p = _add("contradiction-list", help="list contradictions")
    p.add_argument("--all", action="store_true", help="include resolved")
    # WS-B: extended with --disposition / --kind filters
    p.add_argument("--disposition", default=None, help="filter by disposition")
    p.add_argument("--kind", default=None, help="filter by kind")
    # P1.6: filter on the cross-subject flag
    p.add_argument("--cross-subject-filter", dest="cross_subject_filter",
                   choices=["any", "only", "exclude"], default="any",
                   help="include cross-subject contradictions: any (default), only, or exclude")
    p.set_defaults(func=cmd_contradiction_list)

    p = _add("contradiction-resolve",
             help="resolve a contradiction (optionally supersede the losing claim)")
    p.add_argument("contradiction_id", type=int)
    p.add_argument("--keep", type=int, required=True, help="claim_id that wins")
    p.add_argument("--drop", type=int, help="claim_id to supersede (optional)")
    p.set_defaults(func=cmd_contradiction_resolve)

    p = _add("view-get", help="fetch cached view for a query")
    p.add_argument("query")
    p.set_defaults(func=cmd_view_get)

    p = _add("view-cache", help="store a generated view")
    p.add_argument("query")
    p.add_argument("response")
    p.add_argument("--claim-ids", required=True, help="comma-separated claim ids used")
    p.set_defaults(func=cmd_view_cache)

    p = _add("cache-clear", help="drop all cached views")
    p.set_defaults(func=cmd_cache_clear)

    p = _add("stats-json", help="stats as JSON")
    p.set_defaults(func=cmd_stats)

    p = _add("config-get", help="get store config value(s); omit KEY for full dump")
    p.add_argument("key", nargs="?", default=None,
                   help="config key (currently only 'locale'); omit for all")
    p.set_defaults(func=cmd_config_get)

    p = _add("config-set",
             help="set a store config value (currently only 'locale')")
    p.add_argument("key")
    p.add_argument("value")
    p.set_defaults(func=cmd_config_set)

    # ----- WS-A concepts -----

    p = _add("concept-add", help="add a concept with support claims")
    p.add_argument("--subject", required=True)
    p.add_argument("--statement", required=True)
    p.add_argument("--inference-type", default="summary",
                   choices=["summary", "generalization", "synthesis"])
    p.add_argument("--support", required=True,
                   help="comma-separated claim_id:role pairs, OR a JSON array "
                        "(e.g. '[[26,\"premise\"]]') or object "
                        "(e.g. '{\"26\":\"premise\"}')")
    p.add_argument("--confidence", type=float, default=0.7)
    p.add_argument("--skip-validation", action="store_true",
                   help="leave concept in draft without running LLM validation")
    p.set_defaults(func=cmd_concept_add)

    p = _add("concept-derive",
             help="derive concepts from claims about a subject (LLM)")
    p.add_argument("--subject", required=True)
    p.add_argument("--claims", default=None,
                   help="comma-separated claim ids (default: all active for subject)")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_concept_derive)

    p = _add("concept-get", help="get a concept with its support set")
    p.add_argument("concept_id", type=int)
    p.add_argument("--with-spans", action="store_true",
                   help="include span text for each supporting claim")
    p.set_defaults(func=cmd_concept_get)

    p = _add("concept-list", help="list concepts")
    p.add_argument("--subject", default=None)
    p.add_argument("--status", default=None,
                   choices=["draft", "attested", "active", "stale",
                            "superseded", "invalidated"])
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(func=cmd_concept_list)

    p = _add("concept-attest",
             help="promote a draft concept to 'attested' with an attestor trail (P1.2)")
    p.add_argument("concept_id", type=int)
    p.add_argument("--attested-by", required=True,
                   help="identifier for the attestor, e.g. 'agent-claude-2026-04' or a user handle")
    p.add_argument("--rationale", required=True,
                   help="short explanation of why the concept is grounded in its support spans")
    p.set_defaults(func=cmd_concept_attest)

    p = _add("concept-validate",
             help="re-run validation on a concept (LLM)")
    p.add_argument("concept_id", type=int)
    p.set_defaults(func=cmd_concept_validate)

    p = _add("concept-rebuild",
             help="rebuild a stale/active concept (LLM)")
    p.add_argument("concept_id", type=int)
    p.set_defaults(func=cmd_concept_rebuild)

    p = _add("concept-supersede",
             help="mark old concept as superseded by new concept")
    p.add_argument("old_id", type=int)
    p.add_argument("new_id", type=int)
    p.set_defaults(func=cmd_concept_supersede)

    p = _add("concept-invalidate",
             help="manually invalidate a concept")
    p.add_argument("concept_id", type=int)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_concept_invalidate)

    # ----- WS-B contradictions -----

    p = _add("contradiction-scan",
             help="scan for contradictions (LLM-calling)")
    p.add_argument("--subject", default=None)
    p.add_argument("--kind", default=None)
    p.add_argument("--since", default=None, help="ISO timestamp or epoch float")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_contradiction_scan)

    p = _add("contradiction-dispose",
             help="apply a disposition to a contradiction")
    p.add_argument("contradiction_id", type=int)
    p.add_argument("--disposition", required=True)
    p.add_argument("--rule", default=None)
    p.add_argument("--applies-when", default=None, help="JSON predicate")
    p.add_argument("--rationale-concept", type=int, default=None)
    p.add_argument("--keep", type=int, default=None)
    p.add_argument("--drop", type=int, default=None)
    p.set_defaults(func=cmd_contradiction_dispose)

    p = _add("contradiction-get",
             help="get a contradiction with both claims and disposition")
    p.add_argument("contradiction_id", type=int)
    p.set_defaults(func=cmd_contradiction_get)

    p = _add("contradiction-rule-get",
             help="get the contradiction_rules row for a contradiction")
    p.add_argument("contradiction_id", type=int)
    p.set_defaults(func=cmd_contradiction_rule_get)

    # ----- WS-C authority -----

    p = _add("source-authority-set",
             help="set domain metadata for a source")
    p.add_argument("source_id", type=int)
    p.add_argument("--domain", required=True)
    p.add_argument("--metadata", required=True, help="JSON string")
    p.add_argument("--strict", action="store_true",
                   help="refuse to write if validation errors exist")
    p.set_defaults(func=cmd_source_authority_set)

    p = _add("source-authority-get",
             help="get source metadata with authority rank")
    p.add_argument("source_id", type=int)
    p.set_defaults(func=cmd_source_authority_get)

    p = _add("source-list-by-domain",
             help="list sources in a domain")
    p.add_argument("domain")
    p.set_defaults(func=cmd_source_list_by_domain)

    p = _add("source-retract",
             help="retract a scientific source (cascade claims/concepts/views)")
    p.add_argument("source_id", type=int)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_source_retract)

    p = _add("source-unretract",
             help="reverse a retraction on a scientific source")
    p.add_argument("source_id", type=int)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_source_unretract)

    # ----- WS-D conditions -----

    p = _add("claim-condition-add",
             help="link a condition claim to a claim")
    p.add_argument("--claim", type=int, required=True)
    p.add_argument("--condition", type=int, required=True)
    p.add_argument("--kind", required=True)
    explicit_grp = p.add_mutually_exclusive_group()
    explicit_grp.add_argument("--explicit", dest="explicit",
                              action="store_true", default=True)
    explicit_grp.add_argument("--no-explicit", dest="explicit",
                              action="store_false")
    p.add_argument("--confidence", type=float, default=0.9)
    p.set_defaults(func=cmd_claim_condition_add)

    p = _add("claim-condition-remove",
             help="remove a condition link from a claim")
    p.add_argument("--claim", type=int, required=True)
    p.add_argument("--condition", type=int, required=True)
    p.set_defaults(func=cmd_claim_condition_remove)

    p = _add("claim-conditions-list",
             help="list conditions for a claim")
    p.add_argument("claim_id", type=int)
    p.set_defaults(func=cmd_claim_conditions_list)

    p = _add("claim-condition-extract",
             help="infer conditions for a claim (LLM)")
    p.add_argument("claim_id", type=int)
    p.add_argument("--same-source-only", action="store_true",
                   default=True)
    p.add_argument("--no-same-source-only", dest="same_source_only",
                   action="store_false")
    p.set_defaults(func=cmd_claim_condition_extract)

    # ----- P2.4 provenance -----

    p = _add("provenance",
             help="walk the provenance chain for a single claim (P2.4)")
    p.add_argument("claim_id", type=int)
    p.set_defaults(func=cmd_provenance)

    # ----- P2.5 compose (agent-mode synthesis brief, no LLM) -----

    p = _add("compose",
             help="emit a retrieval+disposition brief for the agent to compose from (P2.5)")
    p.add_argument("--query", required=True,
                   help="question to retrieve evidence for")
    p.add_argument("-k", type=int, default=30,
                   help="claims to retrieve")
    p.add_argument("--concept-k", type=int, default=None,
                   help="concepts to retrieve (default max(5, k // 3))")
    p.add_argument("--context", default=None,
                   help='JSON, e.g. \'{"jurisdiction":"US-CA","date":"2026-04-22"}\'')
    p.add_argument("--no-expand", action="store_true",
                   help="don't add claims that conflict with or condition retrieved ones")
    p.set_defaults(func=cmd_compose)

    # ----- P2.2 report-json (agent-mode; top-level `report` wraps it with formatting) -----

    p = _add("report-json",
             help="one-shot JSON snapshot of the store (P2.2)")
    p.set_defaults(func=cmd_report_agent)

    # ----- WS-E.1 predicate aliases (from docs/plan/70-ws-predicates.md) -----

    p = _add("predicate-alias-add",
             help="declare FROM predicate is the same as TO; "
                  "rewrites existing claims in the given domain")
    p.add_argument("--domain", required=True,
                   help="legal | scientific | policy | corporate | generic | * "
                        "(* = cross-domain wildcard, last-resort)")
    p.add_argument("from_predicate", metavar="FROM")
    p.add_argument("to_predicate", metavar="TO")
    p.set_defaults(func=cmd_predicate_alias_add)

    p = _add("predicate-alias-list", help="list predicate aliases")
    p.add_argument("--domain", default=None,
                   help="filter by domain (omit for all)")
    p.set_defaults(func=cmd_predicate_alias_list)

    p = _add("predicate-alias-remove",
             help="remove a predicate alias (does NOT un-rewrite "
                  "previously rewritten claims)")
    p.add_argument("--domain", required=True)
    p.add_argument("from_predicate", metavar="FROM")
    p.set_defaults(func=cmd_predicate_alias_remove)

    # ----- WS-E.2 predicate senses (from docs/plan/71-ws-predicate-senses.md) -----

    p = _add("predicate-sense-add",
             help="add a predicate sense to the catalog")
    p.add_argument("--canonical", required=True)
    p.add_argument("--sense-tag", dest="sense_tag", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--definition", default=None)
    p.add_argument("--parent-id", dest="parent_id", type=int, default=None)
    p.add_argument("--inverse-id", dest="inverse_id", type=int, default=None)
    p.add_argument("--is-symmetric", dest="is_symmetric",
                   action="store_true", default=False)
    p.add_argument("--is-transitive", dest="is_transitive",
                   action="store_true", default=False)
    p.set_defaults(func=cmd_predicate_sense_add)

    p = _add("predicate-sense-list", help="list predicate senses")
    p.add_argument("--canonical", default=None)
    p.add_argument("--domain", default=None)
    p.set_defaults(func=cmd_predicate_sense_list)

    p = _add("predicate-sense-get",
             help="get a sense row plus its claim_count")
    p.add_argument("sense_id", type=int)
    p.set_defaults(func=cmd_predicate_sense_get)

    p = _add("claim-predicate-sense-set",
             help="assign a sense to a claim (mirrors claim-condition-add "
                  "honesty contract)")
    p.add_argument("claim_id", type=int)
    p.add_argument("--sense", type=int, required=True,
                   help="sense_id from predicate-sense-add")
    p.add_argument("--assigned-by", dest="assigned_by", required=True,
                   choices=["agent", "llm", "human"])
    p.add_argument("--confidence", type=float, default=0.9)
    p.add_argument("--rationale", default=None)
    explicit_grp = p.add_mutually_exclusive_group()
    explicit_grp.add_argument("--explicit", dest="explicit",
                              action="store_true", default=False)
    explicit_grp.add_argument("--no-explicit", dest="explicit",
                              action="store_false")
    p.set_defaults(func=cmd_claim_predicate_sense_set)

    p = _add("claim-predicate-sense-unset",
             help="unset the sense assignment on a claim")
    p.add_argument("claim_id", type=int)
    p.set_defaults(func=cmd_claim_predicate_sense_unset)

    p = _add("predicate-sense-extract",
             help="LLM-assisted sense assignment for a claim "
                  "(writes assigned_by='llm', explicit=0)")
    p.add_argument("claim_id", type=int)
    p.set_defaults(func=cmd_predicate_sense_extract)

    return registered
