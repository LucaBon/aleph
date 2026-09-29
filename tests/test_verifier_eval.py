"""Verifier eval harness: pair-file validation, false-accept / false-reject
arithmetic, and an offline run over the shipped pair set."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aleph.verifier import SUPPORTED, UNCERTAIN, UNSUPPORTED, VerifierResult
from aleph.verifier_eval import labels_status, load_pairs, main, score

PAIRS = Path(__file__).resolve().parent.parent / "benchmark" / "verifier_eval" / "pairs.jsonl"


def _write(tmp_path, rows):
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _pair(i, label, labeler="human:a", **kw):
    return {"id": f"p{i}", "span": "s", "sentence": "t", "label": label,
            "labeler": labeler, **kw}


def test_load_rejects_bad_label_duplicate_id_and_missing_field(tmp_path):
    with pytest.raises(ValueError, match="label"):
        load_pairs(_write(tmp_path, [_pair(1, "PARTIAL")]))
    with pytest.raises(ValueError, match="duplicate"):
        load_pairs(_write(tmp_path, [_pair(1, SUPPORTED), _pair(1, SUPPORTED)]))
    with pytest.raises(ValueError, match="labeler"):
        load_pairs(_write(tmp_path, [{"id": "x", "span": "s", "sentence": "t",
                                      "label": SUPPORTED}]))


def test_labels_status_is_draft_until_every_label_is_human(tmp_path):
    human = load_pairs(_write(tmp_path, [_pair(1, SUPPORTED), _pair(2, UNSUPPORTED)]))
    assert labels_status(human) == "human"
    mixed = load_pairs(_write(tmp_path, [_pair(1, SUPPORTED),
                                         _pair(2, UNSUPPORTED, labeler="draft:claude")]))
    assert labels_status(mixed) == "draft"


def test_score_arithmetic(tmp_path):
    pairs = load_pairs(_write(tmp_path, [
        _pair(1, SUPPORTED), _pair(2, SUPPORTED), _pair(3, SUPPORTED), _pair(4, SUPPORTED),
        _pair(5, UNSUPPORTED, error_type="number"), _pair(6, UNSUPPORTED, error_type="number"),
        _pair(7, UNSUPPORTED, error_type="scope"),
    ]))
    preds = {
        "p1": SUPPORTED, "p2": SUPPORTED, "p3": UNSUPPORTED, "p4": UNCERTAIN,
        "p5": UNSUPPORTED, "p6": SUPPORTED, "p7": UNCERTAIN,
    }
    s = score(pairs, {k: VerifierResult(v, "llm") for k, v in preds.items()})
    assert s["n"] == 7
    # false accept: gold UNSUPPORTED predicted SUPPORTED -> p6 of 3
    assert s["false_accept_rate"] == pytest.approx(1 / 3)
    # false reject: gold SUPPORTED not predicted SUPPORTED -> p3, p4 of 4
    assert s["false_reject_rate"] == pytest.approx(2 / 4)
    assert s["uncertain_rate"] == pytest.approx(2 / 7)
    assert s["confusion"][SUPPORTED] == {SUPPORTED: 2, UNSUPPORTED: 1, UNCERTAIN: 1}
    assert s["by_error_type"]["number"] == {"n": 2, "false_accepts": 1}
    assert s["by_error_type"]["scope"] == {"n": 1, "false_accepts": 0}


def test_score_with_no_negatives_reports_none(tmp_path):
    pairs = load_pairs(_write(tmp_path, [_pair(1, SUPPORTED)]))
    s = score(pairs, {"p1": VerifierResult(SUPPORTED, "llm")})
    assert s["false_accept_rate"] is None and s["false_reject_rate"] == 0


def test_shipped_pairs_are_valid_and_balanced():
    pairs = load_pairs(PAIRS)
    labels = [p["label"] for p in pairs]
    assert len(pairs) >= 40
    assert labels.count(SUPPORTED) >= 15 and labels.count(UNSUPPORTED) >= 15
    assert all(p.get("error_type") for p in pairs if p["label"] == UNSUPPORTED)


def test_deterministic_run_offline(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["--pairs", str(PAIRS), "--verifier", "deterministic",
                 "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["verifier"] == "deterministic"
    assert report["labels_status"] in {"draft", "human"}
    assert report["n"] == len(load_pairs(PAIRS))
    # The deterministic layer alone never accepts a paraphrase it can't prove,
    # so whatever it does accept must be right.
    assert report["false_accept_rate"] == 0
    assert set(report["layers"]) <= {"deterministic", "none"}


def test_llm_verifiers_run_against_a_mock(tmp_path, capsys):
    fx = tmp_path / "fx.json"
    fx.write_text(json.dumps([
        {"match": {"system_contains": "You are a verifier"},
         "response": {"verdict": "SUPPORTED", "reason": "mock"}}]))
    for name in ("baseline", "layered"):
        out = tmp_path / f"{name}.json"
        assert main(["--pairs", str(PAIRS), "--verifier", name,
                     "--mock-llm", str(fx), "--out", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["verifier"] == name and report["n"] > 0
