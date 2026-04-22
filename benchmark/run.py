"""Aleph offline benchmark — deterministic, no API key required.

Produces four numbers for the README:
  1. Grounding rate vs naive LLM
  2. Deletion-propagation (views invalidated + wall ms)
  3. Contradiction-detection recall (vs seeded pairs in ground_truth.json)
  4. Cache-hit cost ratio (LLM calls avoided on a warm ask)

Also reports the free entity-resolution sanity check: distinct surface forms of
"supervised fine-tuning", "paginator api" etc. collapse to canonical subjects.

Usage:
    python benchmark/run.py
    python benchmark/run.py --check-readme  # fail if README quotes stale numbers

The script writes benchmark/results.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aleph.db import Store, normalize_subject  # noqa: E402
from aleph.ingest import ingest_file  # noqa: E402
from aleph.lint import lint as lint_cmd  # noqa: E402
from aleph.query import query as query_cmd, _split_sentences, _parse_citations  # noqa: E402

from fake_llm import BenchmarkFakeLLM, load_ground_truth  # noqa: E402


def build_id_map(store: Store, ground_truth: dict) -> dict[str, int]:
    """After ingest, match each ground-truth claim back to its real DB id by
    (span_start, span_end) inside the right source."""
    mapping: dict[str, int] = {}
    for fname, block in ground_truth["sources"].items():
        src_row = store.conn.execute(
            "SELECT id, content FROM sources WHERE path LIKE ?", (f"%{fname}",)
        ).fetchone()
        if not src_row:
            continue
        source_id, content = src_row["id"], src_row["content"]
        for c in block["claims"]:
            start = content.find(c["span"])
            if start < 0:
                continue
            end = start + len(c["span"])
            row = store.conn.execute(
                "SELECT id FROM claims WHERE source_id = ? AND span_start = ? AND span_end = ?",
                (source_id, start, end),
            ).fetchone()
            if row:
                mapping[c["local_id"]] = row["id"]
    return mapping


def ingest_phase(store: Store, llm: BenchmarkFakeLLM, corpus_dir: Path) -> dict[str, int]:
    for fname in sorted(p.name for p in corpus_dir.glob("*.md")):
        ingest_file(store, llm, corpus_dir / fname)
    return build_id_map(store, llm.ground_truth)


def _is_factual(sentence: str) -> bool:
    """Heuristic: any sentence that makes a quantitative or declarative claim.
    Skips empty/header-only lines."""
    s = sentence.strip()
    return len(s) > 10 and not s.startswith(("#", "_", "-"))


def metric_grounding(store: Store, llm: BenchmarkFakeLLM) -> dict:
    """For each question, compare:
      - Aleph pipeline: answer with [claim:N] cites, per-sentence verifier.
      - Naive pipeline: freehand answer, no Aleph context.
        Every sentence is scored by finding its best-matching source span
        (by numeric overlap) and asking the same verifier.
    A sentence is 'grounded' if it has a valid [claim:N] cite AND the verifier
    returns GROUNDED. Naive answers have no cites, so they're grounded only if
    we grant them credit for matching a source span — which the numeric-match
    verifier in fake_llm will deny when the numbers disagree.
    """
    corpus_text = {}
    for fname in llm.ground_truth["sources"]:
        corpus_text[fname] = (ROOT / "benchmark" / "corpus" / fname).read_text()

    per_question = []
    for q in llm.ground_truth["questions"]:
        # ALEPH path
        store.clear_cache()
        result = query_cmd(store, llm, q["query"], retrieve_k=20, verify=True, use_cache=False)
        aleph_sents = [s for s in _split_sentences(result.answer) if _is_factual(s)]
        aleph_verdicts = {c.sentence: c.verdict for c in result.citations}
        aleph_cited = sum(1 for s in aleph_sents if _parse_citations(s))
        aleph_grounded = sum(
            1 for s in aleph_sents
            if _parse_citations(s) and aleph_verdicts.get(s) == "GROUNDED"
        )

        # NAIVE path: freehand answer, score each sentence by best corpus span
        naive_answer = llm.naive_answer(q["query"])
        naive_sents = [s for s in _split_sentences(naive_answer) if _is_factual(s)]
        naive_cited = sum(1 for s in naive_sents if _parse_citations(s))  # will be 0
        naive_grounded = 0
        for s in naive_sents:
            # find the best candidate span (any sentence in any corpus file that shares >= 1 content word)
            best_span = _best_matching_span(s, corpus_text)
            if not best_span:
                continue
            verdict = llm._verify(
                f"Source span:\n---\n{best_span}\n---\n\nSentence: {s}\n\nVerdict"
            ).get("verdict")
            if verdict == "GROUNDED":
                naive_grounded += 1

        per_question.append({
            "query": q["query"],
            "aleph_sentences": len(aleph_sents),
            "aleph_cited": aleph_cited,
            "aleph_grounded": aleph_grounded,
            "naive_sentences": len(naive_sents),
            "naive_cited": naive_cited,
            "naive_grounded": naive_grounded,
        })

    tot_aleph_s = sum(r["aleph_sentences"] for r in per_question)
    tot_aleph_g = sum(r["aleph_grounded"] for r in per_question)
    tot_naive_s = sum(r["naive_sentences"] for r in per_question)
    tot_naive_g = sum(r["naive_grounded"] for r in per_question)
    return {
        "per_question": per_question,
        "aleph_grounded_pct": round(100.0 * tot_aleph_g / max(tot_aleph_s, 1), 1),
        "naive_grounded_pct": round(100.0 * tot_naive_g / max(tot_naive_s, 1), 1),
        "aleph_sentences_total": tot_aleph_s,
        "naive_sentences_total": tot_naive_s,
    }


def _best_matching_span(sentence: str, corpus_text: dict[str, str]) -> str:
    """Return the ~300-char chunk around the token most likely to anchor the sentence.

    This is the corpus-span a charitable reader would pick to check the sentence
    against; our stance is 'if the naive answer's numbers don't match any span
    that discusses the same topic, it's ungrounded'.
    """
    words = set(re.findall(r"[a-z]{4,}", sentence.lower()))
    best_score, best = 0, ""
    for text in corpus_text.values():
        for para in text.split("\n\n"):
            para_words = set(re.findall(r"[a-z]{4,}", para.lower()))
            score = len(words & para_words)
            if score > best_score:
                best_score, best = score, para
    return best


def metric_contradictions(store: Store, llm: BenchmarkFakeLLM, id_map: dict[str, int]) -> dict:
    seeded = llm.ground_truth["seeded_contradictions"]
    expected_pairs = set()
    for pair in seeded:
        a = id_map.get(pair["a"])
        b = id_map.get(pair["b"])
        if a is not None and b is not None:
            expected_pairs.add(tuple(sorted([a, b])))

    summary = lint_cmd(store, llm)
    found_rows = store.list_contradictions(only_open=True)
    found_pairs = {
        tuple(sorted([r["claim_a_id"], r["claim_b_id"]])) for r in found_rows
    }

    tp = len(expected_pairs & found_pairs)
    fp = len(found_pairs - expected_pairs)
    fn = len(expected_pairs - found_pairs)
    recall = tp / max(len(expected_pairs), 1)
    precision = tp / max(len(found_pairs), 1) if found_pairs else 1.0
    return {
        "seeded": len(expected_pairs),
        "found": len(found_pairs),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "recall_pct": round(100.0 * recall, 1),
        "precision_pct": round(100.0 * precision, 1),
        "lint_summary": summary,
    }


def metric_deletion(store: Store, llm: BenchmarkFakeLLM) -> dict:
    # Prime cache with all three questions.
    for q in llm.ground_truth["questions"]:
        query_cmd(store, llm, q["query"], use_cache=True)
    stats_before = store.stats()
    # Pick the policy source — its claims are cited by one of the cached views.
    target = store.conn.execute(
        "SELECT id FROM sources WHERE path LIKE ?", ("%policy_data_retention.md",)
    ).fetchone()
    if not target:
        return {"error": "policy source not found"}
    source_id = target["id"]
    n_claims = store.conn.execute(
        "SELECT COUNT(*) FROM claims WHERE source_id = ?", (source_id,)
    ).fetchone()[0]

    t0 = time.perf_counter()
    store.remove_source(source_id)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    stats_after = store.stats()
    return {
        "source_removed": "policy_data_retention.md",
        "claims_cascaded": n_claims,
        "views_before": stats_before["cached_views"],
        "views_after": stats_after["cached_views"],
        "views_invalidated": stats_before["cached_views"] - stats_after["cached_views"],
        "wall_ms": round(wall_ms, 2),
    }


def metric_cache(store: Store, llm: BenchmarkFakeLLM) -> dict:
    # Use the first question that still has sources in the store (we just
    # removed the policy source, so avoid that one).
    question = next(
        q["query"] for q in llm.ground_truth["questions"]
        if "acme" not in q["query"].lower()
    )
    store.clear_cache()
    calls_before = llm.calls
    t0 = time.perf_counter()
    query_cmd(store, llm, question, use_cache=True)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    cold_calls = llm.calls - calls_before

    calls_before = llm.calls
    t0 = time.perf_counter()
    query_cmd(store, llm, question, use_cache=True)
    warm_ms = (time.perf_counter() - t0) * 1000.0
    warm_calls = llm.calls - calls_before

    return {
        "question": question,
        "cold_llm_calls": cold_calls,
        "warm_llm_calls": warm_calls,
        "llm_calls_avoided": cold_calls - warm_calls,
        "cold_ms": round(cold_ms, 2),
        "warm_ms": round(warm_ms, 2),
    }


def metric_entity_resolution(store: Store, llm: BenchmarkFakeLLM) -> dict:
    """Free sanity check: the corpus uses varied surface forms for the same
    entities ("Paginator API v1", "The Paginator API v2", "paginator api").
    After ingest + normalization they should collapse to a small set of canonical
    subjects.
    """
    surface_forms = set()
    for block in llm.ground_truth["sources"].values():
        for c in block["claims"]:
            surface_forms.add(c["subject"])
    canonical = {normalize_subject(s) for s in surface_forms}
    distinct_in_store = store.conn.execute(
        "SELECT COUNT(DISTINCT subject) FROM claims WHERE status = 'active'"
    ).fetchone()[0]
    return {
        "surface_forms_input": len(surface_forms),
        "canonical_subjects": len(canonical),
        "distinct_subjects_in_store": distinct_in_store,
    }


def run(corpus_dir: Path) -> dict:
    ground_truth = load_ground_truth(ROOT)
    llm = BenchmarkFakeLLM(ground_truth)

    with tempfile.TemporaryDirectory() as d:
        store = Store(Path(d) / "benchmark.db")

        id_map = ingest_phase(store, llm, corpus_dir)
        llm.set_id_map(id_map)

        grounding = metric_grounding(store, llm)
        entity_res = metric_entity_resolution(store, llm)
        contradictions = metric_contradictions(store, llm, id_map)
        cache = metric_cache(store, llm)
        deletion = metric_deletion(store, llm)  # runs last: it removes a source

        results = {
            "mode": "offline",
            "corpus_files": sorted(p.name for p in corpus_dir.glob("*.md")),
            "metrics": {
                "grounding": grounding,
                "contradictions": contradictions,
                "deletion_propagation": deletion,
                "cache_hit": cache,
                "entity_resolution": entity_res,
            },
            "total_llm_calls": llm.calls,
        }
        store.close()
        return results


def markdown_table(results: dict) -> str:
    m = results["metrics"]
    g = m["grounding"]
    c = m["contradictions"]
    d = m["deletion_propagation"]
    ch = m["cache_hit"]
    e = m["entity_resolution"]
    return "\n".join([
        "| Metric | With Aleph | Without Aleph |",
        "|---|---|---|",
        f"| Sentences with source-verified citations | **{g['aleph_grounded_pct']:.0f}%** "
        f"({sum(pq['aleph_grounded'] for pq in g['per_question'])}/{g['aleph_sentences_total']}) "
        f"| {g['naive_grounded_pct']:.0f}% "
        f"({sum(pq['naive_grounded'] for pq in g['per_question'])}/{g['naive_sentences_total']}) |",
        f"| Seeded contradictions caught by `lint` | **{c['true_positives']}/{c['seeded']}** "
        f"({c['recall_pct']:.0f}% recall, {c['precision_pct']:.0f}% precision) | n/a |",
        f"| Downstream views invalidated on `source-remove` | "
        f"**{d['views_invalidated']}/{d['views_before']}** in {d['wall_ms']:.1f} ms | "
        "stale prose persists |",
        f"| LLM calls on cache hit | **{ch['warm_llm_calls']}** "
        f"(cold: {ch['cold_llm_calls']}) | n/a |",
        f"| Surface subject forms → canonical | "
        f"**{e['surface_forms_input']} → {e['distinct_subjects_in_store']}** | n/a |",
    ])


# ------- README drift check -------

NUM_RE = re.compile(r"\*\*([^*]+)\*\*")


def extract_readme_numbers(readme_text: str) -> list[str]:
    """Pull out every **bolded** cell from the benchmark table in README, if present.

    Looks for a fenced block marked `<!-- BENCHMARK:START -->` ... `<!-- BENCHMARK:END -->`.
    Returns [] if the block is missing — that's a soft signal to write it.
    """
    m = re.search(r"<!-- BENCHMARK:START -->(.*?)<!-- BENCHMARK:END -->", readme_text, re.S)
    if not m:
        return []
    return NUM_RE.findall(m.group(1))


def current_table_numbers(results: dict) -> list[str]:
    """The bolded cells produced by markdown_table(), in order."""
    table = markdown_table(results)
    return NUM_RE.findall(table)


def check_readme_drift(results: dict) -> int:
    readme = (ROOT / "README.md").read_text()
    expected = current_table_numbers(results)
    found = extract_readme_numbers(readme)
    if not found:
        print("README has no BENCHMARK block yet — skipping drift check.", file=sys.stderr)
        return 0
    if expected != found:
        print("README benchmark numbers drifted from benchmark/run.py output:", file=sys.stderr)
        print(f"  README:   {found}", file=sys.stderr)
        print(f"  expected: {expected}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-readme", action="store_true",
                    help="Fail if README benchmark numbers differ from freshly computed ones.")
    args = ap.parse_args()

    corpus_dir = ROOT / "benchmark" / "corpus"
    results = run(corpus_dir)
    out_path = ROOT / "benchmark" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(markdown_table(results))
    print()
    print(f"_full results: {out_path.relative_to(ROOT)}  (total LLM calls: {results['total_llm_calls']})_")

    if args.check_readme:
        return check_readme_drift(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
