"""Retrieval eval: claim recall@k over a labeled query set (Phase 3).

A query is one JSON object per line::

    {"id": "c01", "query": "...", "labeler": "human:<who>" | "draft:<who>",
     "relevant": [{"source": "corpus/art-52-cp.txt",
                   "span_contains": "verbatim fragment of the span"}]}

Each ``relevant`` entry names every active claim whose source path ends with
the path components of ``source`` and whose span contains ``span_contains`` (whitespace-normalized).
Naming claims by their text rather than their ids lets labels survive a
store rebuild. An entry that matches no claim is reported in ``unresolved``,
and a query with no resolvable entry is left out of the means.

- recall@k = relevant claims in the top k / relevant claims
- hit rate = queries with at least one relevant claim in the top k / queries

Run::

    python -m aleph.retrieval_eval --db aleph.db \\
        --queries benchmark/retrieval_eval/corpus_queries.jsonl \\
        --retriever fts,keyword,embedding,hybrid -k 10 [--out R.json]

``--from-ground-truth benchmark/ground_truth.json`` builds a throwaway store
from the benchmark's hand-written claims instead of opening ``--db``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from .db import Store

_REQUIRED = ("id", "query", "relevant", "labeler")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_queries(path) -> list[dict]:
    queries, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [k for k in _REQUIRED if not row.get(k)]
        if missing:
            raise ValueError(f"line {n}: missing {', '.join(missing)}")
        for spec in row["relevant"]:
            if not spec.get("source") or not spec.get("span_contains"):
                raise ValueError(f"line {n}: each relevant entry needs source and span_contains")
        if row["id"] in seen:
            raise ValueError(f"line {n}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        queries.append(row)
    return queries


def labels_status(queries: list[dict]) -> str:
    return "draft" if any(q["labeler"].startswith("draft:") for q in queries) else "human"


def resolve_relevant(store: Store, spec: dict) -> set[int]:
    """Active claims matching a relevant entry."""
    fragment = _norm(spec["span_contains"])
    rows = store.conn.execute(
        "SELECT c.id, s.path, SUBSTR(s.content, c.span_start + 1, c.span_end - c.span_start) "
        "AS span FROM claims c JOIN sources s ON s.id = c.source_id "
        "WHERE c.status = 'active'"
    ).fetchall()
    want = spec["source"].replace("\\", "/")
    while want.startswith("./"):
        want = want[2:]
    want = want.lstrip("/")

    def same_file(path: str) -> bool:
        path = path.replace("\\", "/")
        return path == want or path.endswith("/" + want)

    return {r["id"] for r in rows if same_file(r["path"]) and fragment in _norm(r["span"])}


def evaluate(store: Store, queries: list[dict], retriever, k: int = 10) -> dict:
    from .query import _extract_keywords

    per_query, unresolved = {}, []
    for q in queries:
        relevant: set[int] = set()
        for spec in q["relevant"]:
            ids = resolve_relevant(store, spec)
            if not ids:
                unresolved.append({"query_id": q["id"], **spec})
            relevant |= ids
        if not relevant:
            continue
        top = [r["id"] for r in retriever.search(_extract_keywords(q["query"]), limit=k)]
        found = relevant & set(top)
        per_query[q["id"]] = {"recall": len(found) / len(relevant),
                              "relevant": sorted(relevant), "retrieved": top}
    n = len(per_query)
    return {
        "k": k,
        "n_queries": n,
        "mean_recall": sum(p["recall"] for p in per_query.values()) / n if n else None,
        "hit_rate": sum(p["recall"] > 0 for p in per_query.values()) / n if n else None,
        "unresolved": unresolved,
        "per_query": per_query,
    }


def build_store_from_ground_truth(ground_truth, db_path) -> Store:
    """A store holding the benchmark's hand-written claims (no LLM)."""
    from .ingest import _locate_span

    gt = json.loads(Path(ground_truth).read_text(encoding="utf-8"))
    root = Path(ground_truth).resolve().parent.parent
    store = Store(Path(db_path))
    for name, src in gt["sources"].items():
        path = Path(gt["corpus_dir"]) / name
        content = (root / path).read_text(encoding="utf-8")
        sid = store.add_source(str(path), content)
        for c in src["claims"]:
            located = _locate_span(content, c["span"], 0)
            if located is None:
                raise ValueError(f"ground-truth span not in {path}: {c['span'][:60]!r}")
            store.add_claim(sid, c["subject"], c["predicate"], c["object"],
                            located[0], located[1], c["confidence"])
    return store


def _retriever(name: str, store: Store):
    from . import retrieval

    return {
        "keyword": retrieval.KeywordRetriever,
        "fts": retrieval.FTSRetriever,
        "embedding": retrieval.EmbeddingRetriever,
        "hybrid": retrieval.hybrid_retriever,
    }[name](store)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aleph.retrieval_eval",
                                 description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="store to evaluate against")
    src.add_argument("--from-ground-truth", help="build a store from this ground_truth.json")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--retriever", default="fts",
                    help="comma-separated: keyword, fts, embedding, hybrid")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    queries = load_queries(args.queries)
    with tempfile.TemporaryDirectory() as tmp:
        store = (build_store_from_ground_truth(args.from_ground_truth, Path(tmp) / "gt.db")
                 if args.from_ground_truth else Store(Path(args.db)))
        try:
            results = {name: evaluate(store, queries, _retriever(name, store), args.k)
                       for name in args.retriever.split(",")}
        finally:
            store.close()
    report = {"labels_status": labels_status(queries), "k": args.k, "retrievers": results}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    print(json.dumps({"labels_status": report["labels_status"], "k": args.k, **{
        name: {"mean_recall": r["mean_recall"], "hit_rate": r["hit_rate"],
               "n_queries": r["n_queries"], "unresolved": len(r["unresolved"])}
        for name, r in results.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
