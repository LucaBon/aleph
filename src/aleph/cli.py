"""Command line interface for aleph."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent_cli import cmd_report_agent, register_agent_commands
from .db import SchemaVersionError, Store
from .ingest import ingest_paths
from .lint import lint as lint_cmd, resolve_by_recency
from .llm import LLM, MockLLM, DEFAULT_MODEL
from .query import query as query_cmd


# subcommands that need an LLM (API mode). Everything else is agent-mode and
# needs no API key. When the user supplies --mock-llm or sets
# ALEPH_LLM_FIXTURES, these still route through the same paths but use the
# deterministic MockLLM adapter instead, so an ANTHROPIC_API_KEY is not
# required.
LLM_COMMANDS = {"ingest", "ask", "lint", "concept-derive", "concept-rebuild", "concept-validate", "contradiction-scan", "claim-condition-extract", "predicate-sense-extract"}


def _default_db_path() -> Path:
    # per-project db: ./aleph.db if you're in a project, else ~/.aleph/aleph.db
    local = Path.cwd() / "aleph.db"
    if local.exists():
        return local
    home = Path.home() / ".aleph" / "aleph.db"
    return home


def _store(args) -> Store:
    db = getattr(args, "db", None)
    locale = getattr(args, "locale", None)
    return Store(Path(db) if db else _default_db_path(), locale=locale)


def _llm(args):
    """Construct an LLM adapter. ``--mock-llm path`` (or the
    ``ALEPH_LLM_FIXTURES`` env var) substitutes a :class:`MockLLM` that
    reads canned responses from the given fixtures file — used for CI,
    smoke tests, and developer iteration without an API key (P2.1)."""
    fixtures = getattr(args, "mock_llm", None) or os.environ.get(
        "ALEPH_LLM_FIXTURES"
    )
    if fixtures:
        return MockLLM(fixtures)
    return LLM(model=getattr(args, "model", None) or DEFAULT_MODEL)


def cmd_ingest(args) -> int:
    store = _store(args)
    llm = _llm(args)
    paths = [Path(p) for p in args.paths]
    results = ingest_paths(store, llm, paths, verbose=args.verbose,
                           extract_conditions=getattr(args, 'extract_conditions', False))
    for r in results:
        if r["status"] == "ingested":
            print(f"  + {r['path']}  ({r['claims_added']} claims, "
                  f"{r['claims_dropped_ungrounded']} dropped, "
                  f"{r['claims_flagged_fidelity']} flagged for review)")
        else:
            print(f"  . {r['path']}  [{r['status']}: {r.get('reason', '')}]")
    ingested = [r for r in results if r["status"] == "ingested"]
    if ingested:
        tokens = sum(r["source_tokens_estimate"] for r in ingested)
        costs = [r["cost_usd"] for r in ingested]
        if any(r["llm_usage"] is None for r in ingested):
            print("LLM cost: unknown (this LLM adapter doesn't report usage)")
        elif any(c is None for c in costs):
            print(f"LLM cost: unknown (no list price for model "
                  f"{getattr(llm, 'model', '?')!r})")
        elif not tokens:
            print(f"LLM cost: ${sum(costs):.4f} (no source text)")
        else:
            total = sum(costs)
            print(f"LLM cost: ${total:.4f} for ~{tokens:.0f} source tokens "
                  f"(${total / tokens * 1000:.4f} per 1k; list price, "
                  f"source tokens estimated as chars/4)")
    print()
    print(_fmt_stats(store.stats()))
    return 0


def _verifier(args, llm):
    """``--verifier layered`` opts into the layered verifier; the default
    (``llm``) keeps ask's original span check."""
    if getattr(args, "verifier", "llm") != "layered":
        return None
    from .verifier import LayeredVerifier, default_entailer
    return LayeredVerifier(llm, entailer=default_entailer())


def cmd_ask(args) -> int:
    store = _store(args)
    llm = _llm(args)
    ctx = None
    if args.context:
        try:
            ctx = json.loads(args.context)
        except json.JSONDecodeError as e:
            print(f"error: --context is not valid JSON: {e}", file=sys.stderr)
            return 2
        # If date is ISO string, convert to unix seconds
        if isinstance(ctx.get("date"), str):
            from datetime import datetime
            try:
                ctx["date"] = datetime.fromisoformat(ctx["date"]).timestamp()
            except ValueError as e:
                print(f"error: --context.date not ISO parseable: {e}",
                      file=sys.stderr)
                return 2
    result = query_cmd(
        store, llm, args.question,
        retrieve_k=args.k,
        verify=not args.no_verify,
        use_cache=not args.no_cache,
        context=ctx,
        verifier=_verifier(args, llm),
    )
    if args.json:
        print(json.dumps({
            "query": result.query,
            "answer": result.answer,
            "claim_ids_used": result.claim_ids_used,
            "from_cache": result.from_cache,
            "citations": [
                {"sentence": c.sentence, "claim_ids": c.claim_ids,
                 "verdict": c.verdict, "reason": c.reason}
                for c in result.citations
            ],
        }, indent=2))
    else:
        if result.from_cache:
            print("(from cache — run with --no-cache to regenerate)\n")
        print(result.answer)
        if result.claim_ids_used:
            print(f"\n_used claims: {', '.join(str(i) for i in result.claim_ids_used[:20])}"
                  f"{'...' if len(result.claim_ids_used) > 20 else ''}_")
    return 0


def cmd_show(args) -> int:
    store = _store(args)
    if args.claim_id:
        row = store.get_claim(args.claim_id)
        if not row:
            print(f"no claim with id {args.claim_id}")
            return 1
        span = store.get_span_text(args.claim_id) or ""
        source = store.get_source(row["source_id"])
        print(f"claim #{row['id']}  [{row['status']}]  conf={row['confidence']:.2f}")
        print(f"  {row['subject']} — {row['predicate']} — {row['object']}")
        print(f"  source: {source['path'] if source else '?'} (chars {row['span_start']}-{row['span_end']})")
        print(f"  span:\n    {span!r}")
        return 0
    # no id: show all claims (limited)
    claims = store.all_active_claims()[: args.limit]
    for c in claims:
        print(f"#{c['id']:>4}  {c['subject'][:30]:30s}  {c['predicate'][:20]:20s}  {c['object'][:40]}")
    if len(claims) == args.limit:
        print(f"\n(showing first {args.limit}; increase --limit to see more)")
    return 0


def cmd_sources(args) -> int:
    store = _store(args)
    rows = store.list_sources()
    if not rows:
        print("no sources ingested")
        return 0
    for r in rows:
        n_claims = store.conn.execute(
            "SELECT COUNT(*) FROM claims WHERE source_id = ? AND status = 'active'",
            (r["id"],),
        ).fetchone()[0]
        print(f"#{r['id']:>4}  {r['path']}  ({n_claims} active claims)")
    return 0


def cmd_remove(args) -> int:
    store = _store(args)
    n = store.remove_source(args.source_id)
    if n == 0:
        print(f"no source with id {args.source_id}")
        return 1
    print(f"removed source {args.source_id}, claims cascade-deleted, views invalidated")
    return 0


def cmd_lint(args) -> int:
    store = _store(args)
    llm = _llm(args)
    summary = lint_cmd(store, llm, verbose=args.verbose)
    print(json.dumps(summary, indent=2))
    open_contradictions = store.list_contradictions(only_open=True)
    if open_contradictions:
        print(f"\n{len(open_contradictions)} open contradictions:")
        for c in open_contradictions[:20]:
            a = store.get_claim(c["claim_a_id"])
            b = store.get_claim(c["claim_b_id"])
            print(f"  [{a['id']}] {a['subject']} — {a['predicate']} — {a['object']!r}")
            print(f"  [{b['id']}] {b['subject']} — {b['predicate']} — {b['object']!r}")
            print()
    if args.resolve_by_recency:
        n = resolve_by_recency(store)
        print(f"resolved {n} contradictions by recency (newer supersedes older)")
    return 0


def cmd_stats(args) -> int:
    store = _store(args)
    print(_fmt_stats(store.stats()))
    return 0


def cmd_clear_cache(args) -> int:
    store = _store(args)
    n = store.clear_cache()
    print(f"cleared {n} cached views")
    return 0


def cmd_report(args) -> int:
    """One-shot diagnostic snapshot (P2.2).

    Reuses ``cmd_report_agent`` for the JSON data path and optionally
    pretty-prints it as text or markdown. Useful before running LLM-gated
    commands (concept-derive / contradiction-scan) to eyeball claim
    density and subject coverage.
    """
    store = _store(args)

    # Reuse agent-mode handler to compute the report. It prints JSON to
    # stdout; capture it through the store directly instead of reparsing.
    # Simpler: call the same composition helper locally.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_report_agent(args, store)
    report = json.loads(buf.getvalue())
    if not report.get("ok"):
        print(buf.getvalue())
        return 1
    data = report["data"]

    if args.format == "json":
        print(json.dumps(data, indent=2, default=str))
        return 0

    lines = _format_report_text(data, markdown=(args.format == "markdown"))
    print("\n".join(lines))
    return 0


def _format_report_text(data: dict, *, markdown: bool) -> list[str]:
    h1 = "# " if markdown else ""
    h2 = "## " if markdown else ""
    bullet = "- "

    lines = [f"{h1}Aleph store report"]
    stats = data["stats"]
    lines.append("")
    lines.append(f"{h2}Totals")
    for key in ("sources", "claims_active", "claims_superseded",
                "contradictions_open", "cached_views"):
        lines.append(f"{bullet}{key}: {stats.get(key)}")

    lines.append("")
    lines.append(f"{h2}Sources by domain")
    for domain, count in sorted(data["sources"]["by_domain"].items()):
        lines.append(f"{bullet}{domain}: {count}")
    if data["sources"]["legal_by_authority_type"]:
        lines.append("")
        lines.append(f"{h2}Legal sources by authority type")
        for at, count in sorted(
            data["sources"]["legal_by_authority_type"].items()
        ):
            lines.append(f"{bullet}{at}: {count}")

    lines.append("")
    lines.append(f"{h2}Top claim subjects (active)")
    for item in data["claims"]["top_subjects"][:20]:
        lines.append(f"{bullet}{item['subject']}: {item['count']}")

    lines.append("")
    lines.append(f"{h2}Concepts by status")
    for st, n in sorted(data["concepts"]["by_status"].items()):
        lines.append(f"{bullet}{st}: {n}")

    lines.append("")
    lines.append(f"{h2}Contradictions")
    lines.append(f"{bullet}open: {data['contradictions']['open']}")
    lines.append(f"{bullet}resolved: {data['contradictions']['resolved']}")
    lines.append(
        f"{bullet}cross_subject: {data['contradictions']['cross_subject']}"
    )
    for disp, n in sorted(
        (data["contradictions"]["by_disposition"] or {}).items(),
        key=lambda p: (p[0] or ""),
    ):
        lines.append(f"{bullet}{disp or 'null'}: {n}")

    lines.append("")
    lines.append(f"{h2}Aliases")
    lines.append(f"{bullet}total: {data['aliases']['total']}")
    for t in data["aliases"]["top_targets"]:
        lines.append(
            f"{bullet}{t['canonical_to']}: {t['alias_count']} alias(es)"
        )

    gaps = data["hints"]["subjects_without_concepts"]
    if gaps:
        lines.append("")
        lines.append(
            f"{h2}Subjects without supporting concepts "
            "(candidates for concept-derive)"
        )
        for s in gaps:
            lines.append(f"{bullet}{s}")

    cand = data["hints"]["contradiction_scan_candidates"]
    if cand:
        lines.append("")
        lines.append(
            f"{h2}Possible contradiction pairs "
            "(candidates for contradiction-scan)"
        )
        for pair in cand:
            a, b = pair["claim_a"], pair["claim_b"]
            lines.append(
                f"{bullet}{pair['subject']}: "
                f"[{a['id']}] {a['predicate']} {a['object']!r} "
                f"vs [{b['id']}] {b['predicate']} {b['object']!r}"
            )

    return lines


def _fmt_stats(s: dict) -> str:
    return (
        f"sources:              {s['sources']}\n"
        f"active claims:        {s['claims_active']}\n"
        f"superseded claims:    {s['claims_superseded']}\n"
        f"open contradictions:  {s['contradictions_open']}\n"
        f"cached views:         {s['cached_views']}"
    )


def main(argv: list[str] | None = None) -> int:
    # common flags available both globally and on each subcommand
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS: when a flag is set on the outer parser and the subparser inherits
    # it via parents=[common], argparse would otherwise overwrite the outer value
    # with the subparser's default (None). SUPPRESS makes the subparser leave the
    # attribute alone unless the flag is actually passed after the subcommand.
    common.add_argument("--db", default=argparse.SUPPRESS,
                        help="path to sqlite file (default: ./aleph.db or ~/.aleph/aleph.db)")
    common.add_argument("--model", default=argparse.SUPPRESS,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    common.add_argument("--locale", default=argparse.SUPPRESS,
                        help="subject normalization locale (en|it); seeds the "
                             "store on first connection. Use config-set locale "
                             "to change later.")
    common.add_argument("--mock-llm", dest="mock_llm", default=argparse.SUPPRESS,
                        help="path to a MockLLM fixtures file (JSON or YAML). "
                             "When set, the LLM adapter is swapped for a "
                             "deterministic one that reads canned responses — "
                             "useful for CI and smoke tests (P2.1). Also "
                             "honours the ALEPH_LLM_FIXTURES env var.")

    parser = argparse.ArgumentParser(
        prog="aleph",
        parents=[common],
        description=(
            "Aleph: a stateless-view knowledge base for LLMs. "
            "Atomic claims point back to source spans; prose views are generated on demand and verified."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", parents=[common], help="ingest a file or directory")
    p.add_argument("paths", nargs="+", help="files or directories (.txt, .md)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--extract-conditions", action="store_true",
                   help="run scope/method/limitation extraction pre-pass (WS-D)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", parents=[common], help="ask a question; generates and verifies an answer")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=30, help="how many claims to retrieve (default 30)")
    p.add_argument("--no-verify", action="store_true", help="skip the verifier pass")
    p.add_argument("--no-cache", action="store_true", help="bypass view cache")
    p.add_argument("--context", help='JSON, e.g. \'{"jurisdiction":"US-CA","date":"2026-04-22"}\'')
    p.add_argument("--json", action="store_true", help="emit full structured result")
    p.add_argument("--verifier", choices=["llm", "layered"], default="llm",
                   help="claim-citation verifier: llm (default) or layered "
                        "(deterministic -> entailment if ALEPH_ENTAILER=nli -> llm)")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("show", parents=[common], help="show a claim or list claims")
    p.add_argument("claim_id", type=int, nargs="?", help="claim id (omit to list)")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("sources", parents=[common], help="list ingested sources")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("remove", parents=[common], help="remove a source and all its claims")
    p.add_argument("source_id", type=int)
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("lint", parents=[common], help="scan for contradictions")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--resolve-by-recency", action="store_true",
                   help="auto-supersede older claim when contradictions detected")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("stats", parents=[common], help="print stats about the knowledge base")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("clear-cache", parents=[common], help="drop all cached views")
    p.set_defaults(func=cmd_clear_cache)

    # P2.2: one-shot diagnostics snapshot
    p = sub.add_parser("report", parents=[common],
                       help="one-shot snapshot of store state (P2.2)")
    p.add_argument("--format", choices=["text", "markdown", "json"],
                   default="text")
    p.set_defaults(func=cmd_report)

    # agent commands: direct CLI-level operations Claude Code calls.
    agent_command_names = register_agent_commands(sub, common)

    args = parser.parse_args(argv)
    mock_fixtures = getattr(args, "mock_llm", None) or os.environ.get(
        "ALEPH_LLM_FIXTURES"
    )
    if (
        args.cmd in LLM_COMMANDS
        and not mock_fixtures
        and not os.environ.get("ANTHROPIC_API_KEY")
    ):
        print("error: ANTHROPIC_API_KEY not set "
              "(this command runs the LLM directly; use the agent-mode commands instead, "
              "or pass --mock-llm PATH / set ALEPH_LLM_FIXTURES for offline mode)",
              file=sys.stderr)
        return 2
    # agent commands receive (args, store) while API-mode ones take just (args)
    if args.cmd in agent_command_names:
        try:
            store = _store(args)
        except ValueError as e:
            # e.g. unknown --locale or locale conflict — preserve the JSON
            # envelope contract for agent-mode callers.
            print(json.dumps({
                "ok": False,
                "error": {"code": "invalid_locale", "message": str(e)},
            }, ensure_ascii=False))
            return 1
        except SchemaVersionError as e:
            print(json.dumps({
                "ok": False,
                "error": {"code": "schema_too_new", "message": str(e)},
            }, ensure_ascii=False))
            return 1
        return args.func(args, store)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
