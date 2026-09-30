"""Phase 3: hybrid retrieval by reciprocal rank fusion (RRF)."""
from __future__ import annotations

import pytest

from aleph.retrieval import HybridRetriever, default_retriever


class Listed:
    """A retriever that returns fixed ids, as row-like dicts."""

    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def search(self, keywords, limit=30, *, include_retracted=False):
        self.calls.append((limit, include_retracted))
        return [{"id": i, "src": id(self)} for i in self.ids[:limit]]


def _ids(rows):
    return [r["id"] for r in rows]


def test_rrf_scores_and_order():
    # k=60: d2 is 2nd and 1st -> 1/62 + 1/61; d1 is 1st only -> 1/61
    h = HybridRetriever([Listed([1, 2, 3]), Listed([2, 4])], k=60)
    assert _ids(h.search(["q"], limit=10)) == [2, 1, 4, 3]


def test_agreement_beats_a_single_top_rank():
    h = HybridRetriever([Listed([9, 5]), Listed([7, 5])])
    assert _ids(h.search(["q"], limit=1)) == [5]


def test_ties_break_by_best_rank_then_id():
    h = HybridRetriever([Listed([3, 1]), Listed([1, 3])])
    assert _ids(h.search(["q"])) == [1, 3]


def test_limit_and_pool():
    a, b = Listed(list(range(100))), Listed(list(range(100, 200)))
    h = HybridRetriever([a, b], pool=40)
    assert len(h.search(["q"], limit=5)) == 5
    assert a.calls == [(40, False)]


def test_pool_is_at_least_the_limit():
    a = Listed(list(range(100)))
    HybridRetriever([a], pool=10).search(["q"], limit=25)
    assert a.calls == [(25, False)]


def test_include_retracted_is_passed_through():
    a = Listed([1])
    HybridRetriever([a]).search(["q"], include_retracted=True)
    assert a.calls[0][1] is True


def test_empty_keywords_return_nothing():
    assert HybridRetriever([Listed([1])]).search([]) == []


def test_needs_at_least_one_retriever():
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_env_selects_hybrid_and_reports_missing_embeddings(tmp_path, monkeypatch):
    from aleph.db import Store
    store = Store(tmp_path / "t.db")
    monkeypatch.setenv("ALEPH_RETRIEVER", "hybrid")
    try:
        try:
            r = default_retriever(store)
        except RuntimeError as e:  # sentence-transformers not installed
            assert "sentence-transformers" in str(e)
        else:
            assert isinstance(r, HybridRetriever)
    finally:
        store.close()


def test_minimal_protocol_retrievers_work():
    class Minimal:
        def search(self, keywords, limit=30):
            return [{"id": 1}]
    assert _ids(HybridRetriever([Minimal()]).search(["q"])) == [1]
