"""Verifier eval: false-accept and false-reject rates on labeled pairs.

A pair is one JSON object per line::

    {"id": "v001", "span": "...", "context": "...", "sentence": "...",
     "label": "SUPPORTED" | "UNSUPPORTED", "error_type": "number",
     "labeler": "human:<who>" | "draft:<who>", "source": "path"}

``context`` and ``source`` are optional; ``error_type`` names what is wrong
with an UNSUPPORTED sentence. A label is a human judgement only when its
``labeler`` does not start with ``draft:``; a report is marked
``labels_status: "draft"`` until every label is human.

Rates (UNCERTAIN goes to a human, so it is never an accept):

- false-accept rate = predicted SUPPORTED / gold UNSUPPORTED
- false-reject rate = predicted UNSUPPORTED or UNCERTAIN / gold SUPPORTED

Run::

    python -m aleph.verifier_eval --pairs benchmark/verifier_eval/pairs.jsonl \\
        --verifier baseline|layered|deterministic [--mock-llm F] [--out R.json]

``baseline`` is ask's default span check; ``layered`` is
:class:`aleph.verifier.LayeredVerifier` (entailment layer on with
``ALEPH_ENTAILER=nli``); ``deterministic`` is the layered verifier with no
LLM or entailer, so it needs no API key.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from .verifier import (
    SUPPORTED, UNCERTAIN, UNSUPPORTED, VERDICTS, LayeredVerifier, VerifierResult,
    baseline_verdict, default_entailer,
)

LABELS = (SUPPORTED, UNSUPPORTED)
_REQUIRED = ("id", "span", "sentence", "label", "labeler")


def load_pairs(path) -> list[dict]:
    pairs, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [k for k in _REQUIRED if not row.get(k)]
        if missing:
            raise ValueError(f"line {n}: missing {', '.join(missing)}")
        if row["label"] not in LABELS:
            raise ValueError(f"line {n}: label must be one of {LABELS}, got {row['label']!r}")
        if row["id"] in seen:
            raise ValueError(f"line {n}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        pairs.append(row)
    return pairs


def labels_status(pairs: list[dict]) -> str:
    return "draft" if any(p["labeler"].startswith("draft:") for p in pairs) else "human"


def _rate(num: int, den: int):
    return num / den if den else None


def score(pairs: list[dict], predictions: dict[str, VerifierResult]) -> dict:
    confusion = {g: Counter() for g in LABELS}
    by_type: dict[str, dict] = {}
    for p in pairs:
        pred = predictions[p["id"]].verdict
        confusion[p["label"]][pred] += 1
        if p["label"] == UNSUPPORTED:
            t = by_type.setdefault(p.get("error_type") or "unspecified",
                                   {"n": 0, "false_accepts": 0})
            t["n"] += 1
            t["false_accepts"] += pred == SUPPORTED
    gold_s = sum(confusion[SUPPORTED].values())
    gold_u = sum(confusion[UNSUPPORTED].values())
    uncertain = confusion[SUPPORTED][UNCERTAIN] + confusion[UNSUPPORTED][UNCERTAIN]
    return {
        "n": len(pairs),
        "n_supported": gold_s,
        "n_unsupported": gold_u,
        "false_accept_rate": _rate(confusion[UNSUPPORTED][SUPPORTED], gold_u),
        "false_reject_rate": _rate(gold_s - confusion[SUPPORTED][SUPPORTED], gold_s),
        "uncertain_rate": _rate(uncertain, len(pairs)),
        "confusion": {g: {v: confusion[g][v] for v in VERDICTS if confusion[g][v]}
                      for g in LABELS},
        "by_error_type": dict(sorted(by_type.items())),
    }


def _predictor(name: str, llm) -> Callable[[dict], VerifierResult]:
    if name == "baseline":
        return lambda p: baseline_verdict(llm, p["sentence"], p["span"])
    if name == "layered":
        v = LayeredVerifier(llm, entailer=default_entailer())
    else:  # deterministic
        v = LayeredVerifier(None)
    return lambda p: v.verify(p["sentence"], p["span"], p.get("context", ""))


def evaluate(pairs: list[dict], name: str, llm=None) -> dict:
    predict = _predictor(name, llm)
    predictions = {p["id"]: predict(p) for p in pairs}
    report = {"verifier": name, "labels_status": labels_status(pairs),
              **score(pairs, predictions)}
    report["layers"] = dict(Counter(r.layer for r in predictions.values()))
    report["model"] = getattr(llm, "model", None) if name != "deterministic" else None
    usage = getattr(llm, "usage", None)
    report["llm_usage"] = usage.to_dict() if usage is not None else None
    report["errors"] = [
        {"id": p["id"], "label": p["label"], "predicted": predictions[p["id"]].verdict,
         "layer": predictions[p["id"]].layer, "reason": predictions[p["id"]].reason}
        for p in pairs
        if predictions[p["id"]].verdict != p["label"]
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aleph.verifier_eval",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--verifier", required=True,
                    choices=["baseline", "layered", "deterministic"])
    ap.add_argument("--mock-llm", default=None, help="MockLLM fixtures instead of the API")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    pairs = load_pairs(args.pairs)
    llm = None
    if args.verifier != "deterministic":
        from .llm import DEFAULT_MODEL, LLM, MockLLM
        llm = MockLLM(args.mock_llm) if args.mock_llm else LLM(model=args.model or DEFAULT_MODEL)
    report = evaluate(pairs, args.verifier, llm)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    summary = {k: report[k] for k in ("verifier", "labels_status", "n",
                                      "false_accept_rate", "false_reject_rate",
                                      "uncertain_rate")}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
