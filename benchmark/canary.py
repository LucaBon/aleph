"""Canary leakage benchmark: does a retracted source leave any trace?

1. Build a small store: real sources plus one fabricated "canary" source
   carrying distinctive invented facts (a made-up material, an odd number).
2. Let the canary influence everything it can:
   - answers cached as views;
   - an active concept it supports;
   - a contradiction resolved in its favour (supersede, canary kept);
   - a replicate disposition bumping a real claim's confidence;
   - an alias merge introduced alongside it.
3. Retract the canary source, then regenerate every answer.
4. Look for the canary everywhere a reader could see it: cached views,
   regenerated answers, active/attested concepts, resolved contradictions,
   and the confidence and status of the real claims it touched.

It also reports:
- the over-invalidation rate: views dropped that never depended on the canary;
- regeneration churn: unrelated answers that changed after regeneration.

This measures leakage *inside* the system. Parametric leakage (a live LLM
reciting a retracted fact from its training data) needs a real model and
is out of scope here; that's the verifier's job.

Deterministic, no API key:  python benchmark/canary.py [--json]
Exit code 1 if any leak or invariant violation is found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aleph import authority
from aleph.contradictions import dispose
from aleph.db import Store
from aleph.query import query

CANARY_MARKERS = ["zorblaxite", "47.3 percent"]

REAL_SOURCES = {
    "battery_study.txt": (
        "Field data shows the battery retains 90 percent capacity after 200000 miles. "
        "The battery warranty covers 8 years of service."
    ),
    "motor_notes.txt": "The motor delivers 420 newton metres of peak torque.",
}
CANARY_SOURCE = (
    "canary.txt",
    (
        "Lab tests show the battery retains 47.3 percent capacity after 200000 miles. "
        "The battery warranty covers 8 years of service thanks to its zorblaxite anode."
    ),
)

QUESTIONS = {
    "capacity": "battery capacity retained",
    "warranty": "battery warranty years",
    "anode": "zorblaxite anode battery",
    "torque": "motor torque",  # unrelated to the canary
}


class EchoLLM:
    """Synthesizer that restates every retrieved claim's span and cites it.

    It is maximally 'leaky' by design: anything retrieval hands it ends up
    in the answer. Concepts are restated with their statement. The
    verifier always says GROUNDED, because what's being measured here is
    propagation, not verification.
    """

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        out = []
        for cid, span in re.findall(r"\[claim:(\d+)\][^\n]*\n\s+source span: '([^']*)'", user):
            out.append(f"{span} [claim:{cid}].")
        for kid, stmt in re.findall(r"\[concept:(\d+)\] \([^)]*\) ([^\n]+)", user):
            out.append(f"{stmt} [concept:{kid}].")
        return " ".join(out) or "No supporting claims."

    def complete_json(self, system: str, user: str, max_tokens: int = 4096):
        return {"verdict": "GROUNDED", "reason": "canary benchmark"}


def _claim(store: Store, sid: int, text: str, subject: str, predicate: str) -> int:
    content = store.get_source(sid)["content"]
    i = content.index(text)
    return store.add_claim(sid, subject, predicate, text, i, i + len(text), 0.8)


def _has_canary(text: str) -> list[str]:
    low = (text or "").lower()
    return [m for m in CANARY_MARKERS if m in low]


def run() -> dict:
    tmp = tempfile.TemporaryDirectory()
    store = Store(Path(tmp.name) / "canary.db")
    llm = EchoLLM()

    # ---------- build ----------
    ids = {name: store.add_source(name, text) for name, text in REAL_SOURCES.items()}
    canary = store.add_source(*CANARY_SOURCE)
    store.set_source_metadata(canary, "scientific", {"peer_reviewed": False})

    real_cap = _claim(store, ids["battery_study.txt"], "90 percent capacity", "battery", "retains")
    real_war = _claim(store, ids["battery_study.txt"], "8 years of service", "battery", "warranty")
    _claim(store, ids["motor_notes.txt"], "420 newton metres of peak torque", "motor", "torque")
    can_cap = _claim(store, canary, "47.3 percent capacity", "battery", "retains")
    can_war = _claim(store, canary, "8 years of service", "battery", "warranty")
    can_anode = _claim(store, canary, "zorblaxite anode", "zorblaxite anode", "anode")

    # canary wins a contradiction: the real capacity claim is superseded
    ct_sup = store.add_contradiction(real_cap, can_cap)
    dispose(store, ct_sup, "supersede", keep=can_cap, drop=real_cap)
    # canary "replicates" the real warranty claim, bumping its confidence
    ct_rep = store.add_contradiction(real_war, can_war)
    real_war_conf_before = store.get_claim(real_war)["confidence"]
    dispose(store, ct_rep, "replicate")
    # a concept that leans on the canary
    concept = store.add_concept(
        "battery", "Zorblaxite anodes give the battery its 8 year warranty",
        "synthesis", 0.7, [(can_anode, "premise"), (real_war, "premise")],
    )
    store.update_concept_status(concept, "active", validation_verdict="GROUNDED",
                                validation_reason="canary benchmark")
    # an alias introduced alongside the canary
    store.add_alias("zorblaxite anode", "battery")

    # ---------- use ----------
    before = {k: query(store, llm, q).answer for k, q in QUESTIONS.items()}
    views_before = {
        r["id"]: (r["query"], set(json.loads(r["claim_ids"])), set(json.loads(r["concept_ids"])))
        for r in store.conn.execute("SELECT id, query, claim_ids, concept_ids FROM view_cache")
    }
    canary_claims = {can_cap, can_war, can_anode}
    dependent_views = {
        vid for vid, (_q, cl, co) in views_before.items()
        if cl & canary_claims or concept in co
    }
    leaked_before = {k: _has_canary(a) for k, a in before.items() if _has_canary(a)}

    # ---------- retract ----------
    authority.retract_source(store, canary, "fabricated for leakage test")
    views_after_ids = {r["id"] for r in store.conn.execute("SELECT id FROM view_cache")}
    invalidated = set(views_before) - views_after_ids
    over_invalidated = invalidated - dependent_views
    under_invalidated = dependent_views - invalidated

    # ---------- regenerate + search ----------
    after = {k: query(store, llm, q).answer for k, q in QUESTIONS.items()}
    leaks: list[dict] = []
    for k, a in after.items():
        if _has_canary(a):
            leaks.append({"channel": "regenerated_answer", "question": k, "markers": _has_canary(a)})
    for r in store.conn.execute("SELECT query, response FROM view_cache"):
        if _has_canary(r["response"]):
            leaks.append({"channel": "cached_view", "question": r["query"]})
    for r in store.conn.execute(
        "SELECT id, statement FROM concepts WHERE status IN ('active', 'attested')"
    ):
        if _has_canary(r["statement"]):
            leaks.append({"channel": "citable_concept", "concept_id": r["id"]})
    for r in store.conn.execute(
        "SELECT id, disposition, claim_a_id, claim_b_id FROM contradictions "
        "WHERE disposition != 'unresolved'"
    ):
        if canary_claims & {r["claim_a_id"], r["claim_b_id"]}:
            leaks.append({"channel": "resolution", "contradiction_id": r["id"],
                          "disposition": r["disposition"]})
    if store.get_claim(real_cap)["status"] != "active":
        leaks.append({"channel": "real_claim_still_superseded", "claim_id": real_cap})
    if abs(store.get_claim(real_war)["confidence"] - real_war_conf_before) > 1e-9:
        leaks.append({"channel": "confidence_bump_kept", "claim_id": real_war})

    # Aliases aren't tied to a source, so retraction can't know which merge
    # the canary motivated. Report it; alias-undo is the fix.
    alias_note = [
        {"from": r["alias_from"], "to": r["canonical_to"]}
        for r in store.conn.execute("SELECT * FROM subject_aliases")
        if _has_canary(r["alias_from"])
    ]

    unrelated = [k for k in QUESTIONS if not leaked_before.get(k)]
    churn = [k for k in unrelated if before[k] != after[k]]
    violations = store.check_invariants()

    results = {
        "canary_reached_answers_before_retraction": sorted(leaked_before),
        "leaks_after_retraction": leaks,
        "views_before": len(views_before),
        "views_dependent_on_canary": len(dependent_views),
        "views_invalidated": len(invalidated),
        "over_invalidated": len(over_invalidated),
        "over_invalidation_rate": (len(over_invalidated) / len(invalidated)) if invalidated else 0.0,
        "under_invalidated": len(under_invalidated),
        "regeneration_churn_unrelated": churn,
        "aliases_needing_manual_review": alias_note,
        "invariant_violations": violations,
    }
    store.close()
    tmp.cleanup()
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()
    r = run()
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"canary reached answers before retraction: {r['canary_reached_answers_before_retraction']}")
        print(f"leaks after retraction:                  {len(r['leaks_after_retraction'])}")
        for leak in r["leaks_after_retraction"]:
            print(f"  - {leak}")
        print(f"views invalidated:                       {r['views_invalidated']}/{r['views_before']} "
              f"(dependent: {r['views_dependent_on_canary']}, over: {r['over_invalidated']}, "
              f"under: {r['under_invalidated']})")
        print(f"regeneration churn (unrelated answers):  {r['regeneration_churn_unrelated'] or 'none'}")
        print(f"aliases needing manual review:           {r['aliases_needing_manual_review'] or 'none'}")
        print(f"invariant violations:                    {len(r['invariant_violations'])}")
    ok = (not r["leaks_after_retraction"] and not r["invariant_violations"]
          and not r["under_invalidated"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
