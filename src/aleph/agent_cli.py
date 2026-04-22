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

from .db import Store, normalize_subject


def _ok(data) -> None:
    print(json.dumps({"ok": True, "data": data}, ensure_ascii=False, default=str))


def _err(code: str, message: str, **details) -> None:
    err: dict = {"code": code, "message": message}
    if details:
        err["details"] = details
    print(json.dumps({"ok": False, "error": err}, ensure_ascii=False, default=str))


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
    old_rows = store.conn.execute(
        "SELECT id FROM sources WHERE path = ?", (str(path),)
    ).fetchall()
    old_ids = [r["id"] for r in old_rows]
    removed_claims = 0
    for sid in old_ids:
        removed_claims += store.conn.execute(
            "SELECT COUNT(*) FROM claims WHERE source_id = ?", (sid,)
        ).fetchone()[0]
        store.remove_source(sid)
    new_id = store.add_source(str(path), text)
    _ok({
        "status": "replaced" if old_ids else "ingested",
        "old_source_ids": old_ids,
        "old_claims_removed": removed_claims,
        "source_id": new_id,
        "length": len(text),
    })
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
    )
    row = store.get_claim(claim_id)
    store.clear_cache()
    _ok({
        "claim_id": claim_id,
        "subject": row["subject"],   # normalized/canonical form
        "predicate": row["predicate"],
        "object": row["object"],
        "span_start": start,
        "span_end": end,
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
    })
    return 0


def cmd_claim_search(args, store: Store) -> int:
    keywords = args.query.split() if args.query else []
    rows = store.search_claims(keywords, limit=args.k)
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
        out.append(item)
    _ok({"query": args.query, "results": out})
    return 0


def cmd_claim_by_subject(args, store: Store) -> int:
    """Return all active claims whose subject matches (after normalization+alias)."""
    canonical = store.resolve_subject(args.subject)
    rows = store.conn.execute(
        "SELECT * FROM claims WHERE subject = ? AND status = 'active' ORDER BY predicate",
        (canonical,),
    ).fetchall()
    out = [
        {
            "claim_id": r["id"],
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "confidence": r["confidence"],
        }
        for r in rows
    ]
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
    result = store.add_alias(args.from_subject, args.to_subject)
    if result.get("note") == "would-create-cycle":
        _err(
            "alias_would_create_cycle",
            "adding this alias would create a resolution cycle",
            from_canonical=normalize_subject(args.from_subject),
            to_canonical=normalize_subject(args.to_subject),
        )
        return 1
    result["from_canonical"] = normalize_subject(args.from_subject)
    result["to_canonical"] = normalize_subject(args.to_subject)
    _ok(result)
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

def cmd_contradiction_add(args, store: Store) -> int:
    """Record a contradiction between two claims.

    Validates that both claims exist and that they share the same (resolved)
    subject — a contradiction that spans subjects is almost certainly a bug
    in the caller's reasoning. Predicates may differ (lint groups by subject
    only, so "last on average" vs "last" is a legitimate pair), but identical
    objects are rejected (nothing to contradict).
    """
    a = store.get_claim(args.claim_a)
    if not a:
        _err("claim_not_found", f"no claim with id {args.claim_a}",
             claim_id=args.claim_a)
        return 1
    b = store.get_claim(args.claim_b)
    if not b:
        _err("claim_not_found", f"no claim with id {args.claim_b}",
             claim_id=args.claim_b)
        return 1
    if args.claim_a == args.claim_b:
        _err("contradiction_invalid",
             "cannot contradict a claim with itself",
             claim_id=args.claim_a)
        return 1
    # same subject check (after alias resolution — subjects were already
    # resolved at insertion time, but we re-check for robustness)
    if a["subject"] != b["subject"]:
        _err(
            "contradiction_invalid",
            "claims have different subjects — not a contradiction",
            subject_a=a["subject"], subject_b=b["subject"],
        )
        return 1
    if a["object"].strip().lower() == b["object"].strip().lower():
        _err(
            "contradiction_invalid",
            "claims have identical objects — nothing to contradict",
            object=a["object"],
        )
        return 1
    cid = store.add_contradiction(args.claim_a, args.claim_b)
    if cid is None:
        _ok({"contradiction_id": None, "created": False,
             "note": "contradiction already exists"})
        return 0
    _ok({"contradiction_id": cid, "created": True})
    return 0


def cmd_contradiction_list(args, store: Store) -> int:
    rows = store.list_contradictions(only_open=not args.all)
    out = []
    for r in rows:
        a = store.get_claim(r["claim_a_id"])
        b = store.get_claim(r["claim_b_id"])
        out.append({
            "contradiction_id": r["id"],
            "status": r["status"],
            "claim_a": {"id": a["id"], "subject": a["subject"],
                        "predicate": a["predicate"], "object": a["object"]} if a else None,
            "claim_b": {"id": b["id"], "subject": b["subject"],
                        "predicate": b["predicate"], "object": b["object"]} if b else None,
        })
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
    store.cache_view(args.query, args.response, claim_ids)
    _ok({"cached": True, "claim_ids": claim_ids})
    return 0


def cmd_cache_clear(args, store: Store) -> int:
    n = store.clear_cache()
    _ok({"cleared": n})
    return 0


# ---------- stats ----------

def cmd_stats(args, store: Store) -> int:
    _ok(store.stats())
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

    p = _add("claim-add",
             help="add a claim (span must be verbatim substring of source)")
    p.add_argument("--source-id", type=int, required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--predicate", required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--span", required=True, help="verbatim substring of the source")
    p.add_argument("--confidence", type=float, default=0.8)
    p.set_defaults(func=cmd_claim_add)

    p = _add("claim-get", help="get a claim with its span text")
    p.add_argument("claim_id", type=int)
    p.set_defaults(func=cmd_claim_get)

    p = _add("claim-search", help="keyword search over claims")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=30)
    p.add_argument("--with-spans", action="store_true")
    p.set_defaults(func=cmd_claim_search)

    p = _add("claim-by-subject", help="list claims for a subject (alias-resolved)")
    p.add_argument("subject")
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
    p.set_defaults(func=cmd_alias_add)

    p = _add("alias-list", help="list subject aliases")
    p.set_defaults(func=cmd_alias_list)

    p = _add("contradiction-add", help="record a contradiction between two claims")
    p.add_argument("claim_a", type=int)
    p.add_argument("claim_b", type=int)
    p.set_defaults(func=cmd_contradiction_add)

    p = _add("contradiction-list", help="list contradictions")
    p.add_argument("--all", action="store_true", help="include resolved")
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

    return registered
