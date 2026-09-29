"""Phase 2 API-mode CLI: `ask --verifier layered` and ingest cost output,
driven offline through MockLLM fixtures."""
from __future__ import annotations

import json

from aleph.cli import main
from aleph.db import Store

SPAN = "Model S packs retain about 90% of their original capacity after 200,000 miles."


def _fixtures(tmp_path, answer):
    fx = tmp_path / "fx.json"
    fx.write_text(json.dumps([
        {"match": {"system_contains": "You extract atomic claims"},
         "response": [{"proposition": SPAN, "subject": "Model S pack",
                       "predicate": "retains", "object": "about 90% capacity",
                       "span": SPAN, "confidence": 0.9}]},
        {"match": {"system_contains": "You answer a question"}, "response": answer},
        {"match": {"system_contains": "You are a verifier"},
         "response": {"verdict": "GROUNDED", "reason": "baseline"}},
    ]))
    return fx


def _ingest(tmp_path, capsys, fx):
    src = tmp_path / "tesla.txt"
    src.write_text(SPAN, encoding="utf-8")
    db = tmp_path / "a.db"
    assert main(["--db", str(db), "--mock-llm", str(fx), "ingest", str(src)]) == 0
    return db, capsys.readouterr().out


def test_ingest_prints_cost_line(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _, out = _ingest(tmp_path, capsys, _fixtures(tmp_path, "x"))
    # MockLLM has no priced model, so cost is reported as unknown, not zero.
    assert "LLM cost: unknown" in out


def test_ask_verifier_layered(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fx = _fixtures(tmp_path, "Model S packs retain about 80% of capacity [claim:1].")
    db, _ = _ingest(tmp_path, capsys, fx)
    base = ["--db", str(db), "--mock-llm", str(fx), "ask", "model s pack capacity",
            "--json", "--no-cache"]
    assert main(base) == 0
    assert json.loads(capsys.readouterr().out)["citations"][0]["verdict"] == "GROUNDED"
    assert main(base + ["--verifier", "layered"]) == 0
    [c] = json.loads(capsys.readouterr().out)["citations"]
    assert c["verdict"] == "UNGROUNDED" and c["reason"].startswith("[deterministic]")
    store = Store(db)
    try:
        assert store.get_claim(1)["proposition"] == SPAN
    finally:
        store.close()
