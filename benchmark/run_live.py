"""Aleph live benchmark — runs against the real Anthropic API.

Requires ANTHROPIC_API_KEY. Produces the same table shape as run.py, except:
- Numbers are not deterministic (real LLM).
- The "Without Aleph" naive-baseline column is not computed here; the offline
  fake-vs-fake benchmark (benchmark/run.py) owns that comparison. What this
  script proves is that Aleph's guarantees — verified citations, contradiction
  detection, cache-hit savings — survive contact with a real model.

Writes benchmark/results_live.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aleph.db import Store  # noqa: E402
from aleph.ingest import ingest_file  # noqa: E402
from aleph.lint import lint as lint_cmd  # noqa: E402
from aleph.llm import LLM  # noqa: E402
from aleph.query import query as query_cmd  # noqa: E402


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — this script hits the real API.", file=sys.stderr)
        return 2

    gt = json.loads((ROOT / "benchmark" / "ground_truth.json").read_text())
    corpus_dir = ROOT / "benchmark" / "corpus"
    llm = LLM()

    with tempfile.TemporaryDirectory() as d:
        store = Store(Path(d) / "live.db")

        # --- ingest ---
        t0 = time.perf_counter()
        ingest_results = []
        for fname in sorted(p.name for p in corpus_dir.glob("*.md")):
            r = ingest_file(store, llm, corpus_dir / fname)
            ingest_results.append(r)
        ingest_s = time.perf_counter() - t0
        active = store.stats()["claims_active"]

        # --- grounding (Aleph side only) ---
        per_q = []
        for q in gt["questions"]:
            store.clear_cache()
            res = query_cmd(store, llm, q["query"], use_cache=False)
            total = len(res.citations)
            grounded = sum(1 for c in res.citations if c.verdict == "GROUNDED")
            per_q.append({
                "query": q["query"],
                "sentences_cited": total,
                "grounded": grounded,
            })
        tot = sum(q["sentences_cited"] for q in per_q) or 1
        grounded_pct = round(100.0 * sum(q["grounded"] for q in per_q) / tot, 1)

        # --- contradictions ---
        lint_summary = lint_cmd(store, llm)
        open_contras = len(store.list_contradictions(only_open=True))

        # --- cache ---
        q = gt["questions"][0]["query"]
        store.clear_cache()
        t0 = time.perf_counter()
        query_cmd(store, llm, q, use_cache=True)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        query_cmd(store, llm, q, use_cache=True)
        warm_ms = (time.perf_counter() - t0) * 1000.0

        # --- deletion ---
        target = store.conn.execute(
            "SELECT id FROM sources WHERE path LIKE ?",
            ("%policy_data_retention.md",),
        ).fetchone()
        before = store.stats()
        t0 = time.perf_counter()
        store.remove_source(target["id"])
        wall_ms = (time.perf_counter() - t0) * 1000.0
        after = store.stats()

        results = {
            "mode": "live",
            "model": llm.model,
            "ingest": {
                "wall_s": round(ingest_s, 1),
                "claims_extracted": active,
                "per_file": ingest_results,
            },
            "grounding": {
                "per_question": per_q,
                "grounded_pct": grounded_pct,
            },
            "contradictions": {
                "seeded": len(gt["seeded_contradictions"]),
                "lint_summary": lint_summary,
                "open_contradictions": open_contras,
            },
            "cache_hit": {
                "cold_ms": round(cold_ms, 1),
                "warm_ms": round(warm_ms, 1),
                "speedup_x": round(cold_ms / max(warm_ms, 0.01), 1),
            },
            "deletion_propagation": {
                "views_before": before["cached_views"],
                "views_after": after["cached_views"],
                "views_invalidated": before["cached_views"] - after["cached_views"],
                "wall_ms": round(wall_ms, 2),
            },
        }
        store.close()

    (ROOT / "benchmark" / "results_live.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
