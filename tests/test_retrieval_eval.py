"""Retrieval eval (Phase 3): claim recall@k over a labeled query set.

Relevant claims are named by source file + a verbatim span fragment, not by
claim id, so labels survive rebuilding a store."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aleph.db import Store
from aleph.retrieval_eval import (
    build_store_from_ground_truth, evaluate, load_queries, main, resolve_relevant,
)

ROOT = Path(__file__).resolve().parent.parent
BENCH_QUERIES = ROOT / "benchmark" / "retrieval_eval" / "benchmark_queries.jsonl"
CORPUS_QUERIES = ROOT / "benchmark" / "retrieval_eval" / "corpus_queries.jsonl"
GROUND_TRUTH = ROOT / "benchmark" / "ground_truth.json"


def _write(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _q(i, relevant, labeler="human:a"):
    return {"id": f"q{i}", "query": "x", "relevant": relevant, "labeler": labeler}


def test_load_validates(tmp_path):
    with pytest.raises(ValueError, match="relevant"):
        load_queries(_write(tmp_path, [_q(1, [])]))
    with pytest.raises(ValueError, match="span_contains"):
        load_queries(_write(tmp_path, [_q(1, [{"source": "a.txt"}])]))
    with pytest.raises(ValueError, match="duplicate"):
        rel = [{"source": "a", "span_contains": "b"}]
        load_queries(_write(tmp_path, [_q(1, rel), _q(1, rel)]))


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    sid = s.add_source("docs/a.txt", "Alpha weighs 5 kg.\nBeta weighs 7 kg.")
    s.add_claim(sid, "alpha", "weighs", "5 kg", 0, 18, 0.9)
    s.add_claim(sid, "beta", "weighs", "7 kg", 19, 36, 0.9)
    yield s
    s.close()


def test_resolve_relevant_matches_source_suffix_and_normalized_span(store):
    assert resolve_relevant(store, {"source": "a.txt", "span_contains": "Alpha  weighs"}) == {1}
    assert resolve_relevant(store, {"source": "other.txt", "span_contains": "Alpha"}) == set()


class Stub:
    def __init__(self, ids):
        self.ids = ids

    def search(self, keywords, limit=30, **kw):
        return [{"id": i} for i in self.ids[:limit]]


def test_recall_at_k(store, tmp_path):
    queries = load_queries(_write(tmp_path, [
        _q(1, [{"source": "a.txt", "span_contains": "Alpha"},
               {"source": "a.txt", "span_contains": "Beta"}]),
        _q(2, [{"source": "a.txt", "span_contains": "Beta"}]),
        _q(3, [{"source": "a.txt", "span_contains": "Gamma"}]),  # unresolvable
    ]))
    r = evaluate(store, queries, Stub([1, 3]), k=2)
    assert r["per_query"]["q1"]["recall"] == 0.5
    assert r["per_query"]["q2"]["recall"] == 0.0
    assert r["mean_recall"] == pytest.approx(0.25)
    assert r["hit_rate"] == pytest.approx(0.5)
    assert r["unresolved"] == [{"query_id": "q3", "source": "a.txt", "span_contains": "Gamma"}]
    assert r["n_queries"] == 2


def test_benchmark_run_compares_retrievers(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["--from-ground-truth", str(GROUND_TRUTH), "--queries", str(BENCH_QUERIES),
                 "--retriever", "fts,keyword", "-k", "5", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert set(report["retrievers"]) == {"fts", "keyword"}
    for r in report["retrievers"].values():
        assert r["unresolved"] == [] and r["n_queries"] >= 6
    assert report["labels_status"] == "draft"


def test_ground_truth_store_holds_every_claim(tmp_path):
    store = build_store_from_ground_truth(GROUND_TRUTH, tmp_path / "gt.db")
    try:
        gt = json.loads(GROUND_TRUTH.read_text())
        assert store.stats()["claims_active"] == sum(len(v["claims"]) for v in gt["sources"].values())
    finally:
        store.close()


def test_corpus_queries_point_at_real_spans():
    queries = load_queries(CORPUS_QUERIES)
    assert len(queries) >= 15
    for q in queries:
        for spec in q["relevant"]:
            text = (ROOT / spec["source"]).read_text(encoding="utf-8")
            norm = lambda s: re.sub(r"\s+", " ", s)
            assert norm(spec["span_contains"]) in norm(text), (q["id"], spec)


def test_source_match_is_by_path_component(tmp_path):
    s = Store(tmp_path / "p.db")
    sid = s.add_source("docs/data.txt", "Alpha weighs 5 kg.")
    s.add_claim(sid, "alpha", "weighs", "5 kg", 0, 18, 0.9)
    try:
        assert resolve_relevant(s, {"source": "a.txt", "span_contains": "Alpha"}) == set()
        assert resolve_relevant(s, {"source": "data.txt", "span_contains": "Alpha"}) == {1}
        assert resolve_relevant(s, {"source": "docs/data.txt", "span_contains": "Alpha"}) == {1}
    finally:
        s.close()
