"""Phase 2 ingest: extracted propositions, fidelity flags, and LLM cost per
1k source tokens."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from aleph.db import Store
from aleph.ingest import EXTRACT_SYSTEM, ingest_file
from aleph.llm import LLM, MockLLM, Usage, cost_usd

TEXT = "Model S packs retain about 90% of their original capacity after 200,000 miles."


class ExtractLLM:
    """Returns fixed extraction items and reports usage like the real adapter."""

    model = "claude-opus-4-7"

    def __init__(self, items, input_tokens=1000, output_tokens=200):
        self.items = items
        self.usage = Usage()
        self._in, self._out = input_tokens, output_tokens

    def complete_json(self, system, user, max_tokens=4096):
        self.usage.add(input_tokens=self._in, output_tokens=self._out)
        return self.items


def _item(obj, proposition=None):
    d = {"subject": "Model S pack", "predicate": "retains", "object": obj,
         "span": TEXT, "confidence": 0.9}
    if proposition is not None:
        d["proposition"] = proposition
    return d


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "tesla.txt"
    p.write_text(TEXT, encoding="utf-8")
    return p


def test_extraction_prompt_asks_for_a_proposition():
    assert '"proposition"' in EXTRACT_SYSTEM


def test_ingest_stores_proposition_and_counts_fidelity_flags(store, src):
    llm = ExtractLLM([
        _item("about 90% capacity", "Model S packs retain about 90% of their capacity."),
        _item("about 80% capacity"),
    ])
    r = ingest_file(store, llm, src)
    assert r["claims_added"] == 2
    assert r["claims_flagged_fidelity"] == 1
    props = [row["proposition"] for row in store.all_active_claims()]
    assert "Model S packs retain about 90% of their capacity." in props
    assert len(store.list_reviews()) == 1


def test_ingest_reports_cost_per_1k_source_tokens(store, src):
    llm = ExtractLLM([_item("about 90% capacity")], input_tokens=1000, output_tokens=200)
    r = ingest_file(store, llm, src)
    u = r["llm_usage"]
    assert u["calls"] == 1
    assert u["input_tokens"] == 1000 and u["output_tokens"] == 200
    # claude-opus-4-7: $5 / $25 per MTok
    assert r["cost_usd"] == pytest.approx(1000 * 5e-6 + 200 * 25e-6)
    assert r["source_chars"] == len(TEXT)
    assert r["source_tokens_estimate"] == pytest.approx(len(TEXT) / 4)
    assert r["cost_per_1k_source_tokens"] == pytest.approx(
        r["cost_usd"] / (len(TEXT) / 4) * 1000)


def test_ingest_without_usage_reporting_still_works(store, src):
    class Bare:
        def complete_json(self, system, user, max_tokens=4096):
            return [_item("about 90% capacity")]
    r = ingest_file(store, Bare(), src)
    assert r["claims_added"] == 1
    assert r["llm_usage"] is None and r["cost_usd"] is None


def test_cost_usd_prices_cache_tokens_and_unknown_models():
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000,
              cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
    assert cost_usd("claude-sonnet-5", u) == pytest.approx(2 + 10 + 2 * 1.25 + 2 * 0.1)
    assert cost_usd("some-other-model", u) is None


def test_llm_adapter_accumulates_response_usage(monkeypatch):
    llm = LLM.__new__(LLM)
    llm.model = "claude-haiku-4-5"
    llm.max_attempts = 1
    llm.backoff_base = 0
    llm.usage = Usage()
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=3,
                              cache_creation_input_tokens=None,
                              cache_read_input_tokens=2),
    )
    llm.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
    assert llm.complete("s", "u") == "hi"
    assert (llm.usage.calls, llm.usage.input_tokens, llm.usage.output_tokens,
            llm.usage.cache_read_input_tokens) == (1, 10, 3, 2)


def test_mock_llm_reports_zero_usage(tmp_path):
    fx = tmp_path / "f.json"
    fx.write_text("[]")
    assert MockLLM(fx).usage.to_dict()["calls"] == 0


def test_cost_report_includes_the_conditions_pre_pass(store, src, monkeypatch):
    from aleph import ingest as ingest_mod

    llm = ExtractLLM([_item("about 90% capacity")], input_tokens=1000, output_tokens=0)

    def scope_pass(store_, llm_, source_id, text):
        llm_.usage.add(input_tokens=1000)
        return []
    monkeypatch.setattr(ingest_mod, "_extract_scope_claims", scope_pass)
    r = ingest_file(store, llm, src, extract_conditions=True)
    assert r["llm_usage"]["calls"] == 2
    assert r["llm_usage"]["input_tokens"] == 2000
