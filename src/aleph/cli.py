"""Command line interface for aleph."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent_cli import register_agent_commands
from .db import Store
from .ingest import ingest_paths
from .lint import lint as lint_cmd, resolve_by_recency
from .llm import LLM, DEFAULT_MODEL
from .query import query as query_cmd


# subcommands that need an LLM (API mode). Everything else is agent-mode and
# needs no API key.
LLM_COMMANDS = {"ingest", "ask", "lint"}


def _default_db_path() -> Path:
    # per-project db: ./aleph.db if you're in a project, else ~/.aleph/aleph.db
    local = Path.cwd() / "aleph.db"
    if local.exists():
        return local
    home = Path.home() / ".aleph" / "aleph.db"
    return home


def _store(args) -> Store:
    db = getattr(args, "db", None)
    return Store(Path(db) if db else _default_db_path())


def _llm(args) -> LLM:
    return LLM(model=getattr(args, "model", None) or DEFAULT_MODEL)


def cmd_ingest(args) -> int:
    store = _store(args)
    llm = _llm(args)
    paths = [Path(p) for p in args.paths]
    results = ingest_paths(store, llm, paths, verbose=args.verbose)
    for r in results:
        if r["status"] == "ingested":
            print(f"  + {r['path']}  ({r['claims_added']} claims, "
                  f"{r['claims_dropped_ungrounded']} dropped)")
        else:
            print(f"  . {r['path']}  [{r['status']}: {r.get('reason', '')}]")
    print()
    print(_fmt_stats(store.stats()))
    return 0


def cmd_ask(args) -> int:
    store = _store(args)
    llm = _llm(args)
    result = query_cmd(
        store, llm, args.question,
        retrieve_k=args.k,
        verify=not args.no_verify,
        use_cache=not args.no_cache,
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
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", parents=[common], help="ask a question; generates and verifies an answer")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=30, help="how many claims to retrieve (default 30)")
    p.add_argument("--no-verify", action="store_true", help="skip the verifier pass")
    p.add_argument("--no-cache", action="store_true", help="bypass view cache")
    p.add_argument("--json", action="store_true", help="emit full structured result")
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

    # agent commands: direct CLI-level operations Claude Code calls.
    agent_command_names = register_agent_commands(sub, common)

    args = parser.parse_args(argv)
    if args.cmd in LLM_COMMANDS and not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set "
              "(this command runs the LLM directly; use the agent-mode commands instead)",
              file=sys.stderr)
        return 2
    # agent commands receive (args, store) while API-mode ones take just (args)
    if args.cmd in agent_command_names:
        store = _store(args)
        return args.func(args, store)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
