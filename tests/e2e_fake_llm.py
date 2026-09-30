"""End-to-end test with a deterministic fake LLM.

Proves the whole pipeline — ingest, claim extraction, retrieval, synthesis,
verifier, contradiction detection, resolution — wires together correctly.
"""
from pathlib import Path
import tempfile

from aleph.db import Store
from aleph.ingest import ingest_file
from aleph.query import query
from aleph.lint import lint, resolve_by_recency


class FakeLLM:
    """A deterministic LLM that inspects the system prompt to decide what to return."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=2048):
        self.calls += 1
        # synthesis step
        if "You answer a question" in system:
            return (
                "Tesla Model S packs retain about 90% of original capacity after 200,000 miles [claim:1]. "
                "Independent teardowns of high-mileage fleets found 87-92% retention [claim:2]. "
                "An older figure stated Tesla batteries last 8 years on average [claim:7]. "
                "This figure has been superseded by fleet data showing packs last well beyond 10 years [claim:8]."
            )
        return "UNKNOWN"

    def complete_json(self, system, user, max_tokens=4096):
        self.calls += 1
        # claim extraction
        if "You extract atomic claims" in system:
            # extract spans from the tesla document
            claims = []
            patterns = [
                ("tesla model s packs", "retain", "90% of original capacity after 200,000 miles",
                 "are expected\nto retain about 90% of their original capacity after 200,000 miles"),
                ("high-mileage model s fleets", "retain", "87-92% of capacity",
                 "finding average capacity retention of 87-92% in high-mileage fleets"),
                ("battery degradation", "is", "non-linear",
                 "Degradation is non-linear"),
                ("lfp cells", "handle", "3000 full charge cycles to 80% capacity",
                 "LFP cells handle about 3,000 full\ncharge cycles to 80% capacity"),
                ("nca cells", "handle", "1500 full charge cycles to 80% capacity",
                 "versus roughly 1,500 for NCA"),
                ("nca cells", "have energy density of", "260 wh/kg",
                 "NCA\ncells have higher energy density — about 260 Wh/kg"),
                ("tesla batteries", "last on average", "8 years",
                 "Tesla batteries last 8\nyears on average before requiring replacement"),
                ("tesla batteries", "last", "well beyond 10 years under typical use",
                 "most packs last well beyond 10 years under typical use"),
            ]
            for subj, pred, obj, span in patterns:
                if span in user:
                    claims.append({
                        "subject": subj, "predicate": pred, "object": obj,
                        "span": span, "confidence": 0.9,
                    })
            return claims

        # verifier
        if "You are a verifier" in system:
            # simple rule: if the sentence uses numbers/terms from the span, GROUNDED;
            # specifically flag the outdated "8 years" sentence as PARTIAL
            if "8 years on average" in user:
                return {"verdict": "GROUNDED", "reason": "span states this directly"}
            return {"verdict": "GROUNDED", "reason": "sentence reflects span content"}

        # contradiction judge
        if "You judge whether two claims" in system:
            # mark the 8-years vs 10-years pair as contradicting
            if "8 years" in user and "10 years" in user:
                return {"contradicts": True, "reason": "different lifespans for same subject"}
            # LFP vs NCA cycle counts — different subjects, not a contradiction
            return {"contradicts": False, "reason": "different subjects or facets"}

        return {}


def main():
    fake = FakeLLM()
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        store = Store(db)

        # 1. INGEST
        print("=" * 60)
        print("STEP 1: INGEST")
        print("=" * 60)
        example = Path(__file__).resolve().parents[1] / "examples" / "tesla_batteries.md"
        result = ingest_file(store, fake, example)
        print(f"  {result}")
        print()
        print(f"  stats: {store.stats()}")
        print()
        for row in store.all_active_claims():
            print(f"  #{row['id']:>2}  {row['subject']:<35s} {row['predicate']:<25s} {row['object']}")

        # 2. ASK
        print()
        print("=" * 60)
        print("STEP 2: ASK (with verifier)")
        print("=" * 60)
        qr = query(store, fake, "How long do Tesla batteries actually last?")
        print(f"  from_cache: {qr.from_cache}")
        print(f"  claim_ids_used: {qr.claim_ids_used}")
        print()
        print("  answer:")
        for line in qr.answer.split("\n"):
            print(f"    {line}")
        print()
        print("  per-sentence citations:")
        for c in qr.citations:
            print(f"    [{c.verdict:<10s}] claims={c.claim_ids}  {c.sentence[:80]}{'...' if len(c.sentence) > 80 else ''}")

        # 3. CACHE HIT
        print()
        print("=" * 60)
        print("STEP 3: ASK AGAIN (should hit cache)")
        print("=" * 60)
        qr2 = query(store, fake, "How long do Tesla batteries actually last?")
        print(f"  from_cache: {qr2.from_cache}  (should be True)")

        # 4. LINT (find contradictions)
        print()
        print("=" * 60)
        print("STEP 4: LINT")
        print("=" * 60)
        lint_result = lint(store, fake)
        print(f"  {lint_result}")
        for c in store.list_contradictions(only_open=True):
            a = store.get_claim(c["claim_a_id"])
            b = store.get_claim(c["claim_b_id"])
            print(f"    contradiction: [{a['id']}] '{a['object']}' vs [{b['id']}] '{b['object']}'")

        # 5. RESOLVE
        print()
        print("=" * 60)
        print("STEP 5: RESOLVE BY RECENCY")
        print("=" * 60)
        # Recency is the *source* date. Both claims come from the one example
        # file, which has no date, so nothing can be ordered: the pair stays
        # open. (Extraction time used to decide this; it no longer does.)
        n = resolve_by_recency(store)
        print(f"  resolved: {n}")
        assert n == 0, "an undated pair from a single source must stay open"
        assert store.list_contradictions(only_open=True), "the pair should still be open"
        print(f"  stats: {store.stats()}")

        # 6. REMOVE A SOURCE
        print()
        print("=" * 60)
        print("STEP 6: REMOVE SOURCE (should cascade)")
        print("=" * 60)
        sources = store.list_sources()
        removed = store.remove_source(sources[0]["id"])
        print(f"  removed: {removed} source")
        print(f"  stats after: {store.stats()}")

        print()
        print("=" * 60)
        print("ALL STEPS PASSED  (LLM calls: {})".format(fake.calls))
        print("=" * 60)


class Tier1FakeLLM:
    """FakeLLM variant that exercises Tier 1 invariants:
      - multi-citation sentence where one cited claim is grounded and one is not
      - span-aware verifier verdicts (not just sentence-keyword lookup)
      - whitespace-variant spans from extraction (source has ' ', LLM returns '\n')
    """

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=2048):
        self.calls += 1
        if "You answer a question" in system:
            # synthesis: a multi-citation sentence where claim 1 supports the
            # energy-density part but claim 2 (about cycles) does not.
            return (
                "NCA cells have energy density of 260 Wh/kg and handle 1500 cycles [claim:1,2].\n"
                "NCA cells have energy density of 260 Wh/kg [claim:1]."
            )
        return "UNKNOWN"

    def complete_json(self, system, user, max_tokens=4096):
        self.calls += 1
        if "You extract atomic claims" in system:
            # whitespace-variant test: source has "NCA cells have" (single space)
            # — we return span with a newline inserted, which the old _locate_span
            # would drop. Tier 1.4 should recover.
            return [
                {
                    "subject": "nca cells",
                    "predicate": "have energy density of",
                    "object": "260 Wh/kg",
                    "span": "NCA cells\nhave energy density of 260 Wh/kg",
                    "confidence": 0.9,
                },
                {
                    "subject": "nca cells",
                    "predicate": "handle",
                    "object": "1500 cycles to 80% capacity",
                    "span": "roughly 1,500 for NCA",
                    "confidence": 0.8,
                },
            ]
        if "You are a verifier" in system:
            # span-aware: if the sentence talks about cycles but the span only
            # mentions energy density (or vice versa), that's UNGROUNDED for
            # that specific claim.
            span_part = ""
            sent_part = user
            if "Source span:" in user:
                rest = user.split("Source span:", 1)[1]
                # span is between the first pair of --- delimiters
                if "---" in rest:
                    body = rest.split("---", 2)
                    if len(body) >= 3:
                        span_part = body[1]
            if "Sentence:" in user:
                sent_part = user.split("Sentence:", 1)[1]
            span_has_density = "Wh/kg" in span_part or "energy density" in span_part
            sent_has_cycles = "cycles" in sent_part
            span_has_cycles = "cycles" in span_part or "1,500" in span_part or "1500" in span_part
            sent_has_density = "Wh/kg" in sent_part or "energy density" in sent_part
            # sentence claims cycles, but span doesn't support cycles
            if sent_has_cycles and not span_has_cycles:
                return {"verdict": "UNGROUNDED", "reason": "span has no cycle count"}
            # sentence claims density, but span doesn't support density
            if sent_has_density and not span_has_density:
                return {"verdict": "UNGROUNDED", "reason": "span has no density figure"}
            return {"verdict": "GROUNDED", "reason": "span supports sentence"}
        return {}


def tier1_regression():
    """Assert the Tier 1 correctness fixes work end-to-end."""
    from aleph.query import query as run_query
    print()
    print("=" * 60)
    print("TIER 1 REGRESSION CHECKS")
    print("=" * 60)

    fake = Tier1FakeLLM()
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "nca.md"
        # source has single spaces; extraction will return a span with '\n'
        src.write_text(
            "NCA cells have energy density of 260 Wh/kg and cycle counts "
            "of roughly 1,500 for NCA chemistries.\n"
        )
        store = Store(Path(d) / "t1.db")

        # --- Tier 1.4: whitespace-tolerant span location ---
        result = ingest_file(store, fake, src)
        print(f"  ingest: {result}")
        assert result["claims_added"] == 2, f"expected 2 claims, got {result}"
        assert result["claims_dropped_ungrounded"] == 0, \
            f"Tier 1.4 broken: whitespace-variant span was dropped: {result}"

        claims = store.all_active_claims()
        claim_ids = {c["subject"] + "/" + c["predicate"]: c["id"] for c in claims}
        density_id = next(c["id"] for c in claims if c["predicate"] == "have energy density of")
        cycles_id = next(c["id"] for c in claims if c["predicate"] == "handle")
        print(f"  claim ids: density={density_id}, cycles={cycles_id}")

        # rewrite the synthesis so the sentence cites the density claim FIRST
        # and the cycles claim SECOND, but says something only the density span
        # supports. Old code checked only the first cited claim — it would say
        # GROUNDED. New code checks every cited claim — it must flag the cycles
        # claim as UNGROUNDED.
        fake_orig_complete = fake.complete

        def patched_complete(system, user, max_tokens=2048):
            if "You answer a question" in system:
                return (
                    f"NCA cells have energy density of 260 Wh/kg [claim:{density_id},{cycles_id}].\n"
                    f"NCA cells have energy density of 260 Wh/kg [claim:{density_id}]."
                )
            return fake_orig_complete(system, user, max_tokens)
        fake.complete = patched_complete

        # --- Tier 1.1: verifier checks every cited claim ---
        qr = run_query(store, fake, "NCA cell properties")
        multi = [c for c in qr.citations if len(c.claim_ids) > 1]
        assert multi, (
            f"Tier 1.1 broken: no multi-citation sentence was verified: {qr.citations}"
        )
        c = multi[0]
        # First cited = density: span has "260 Wh/kg" → GROUNDED
        assert c.per_claim[density_id][0] == "GROUNDED", (
            f"density claim's span should support the sentence, got {c.per_claim[density_id]}"
        )
        # Second cited = cycles: span has "1,500", not density → UNGROUNDED.
        # Under old code (first-claim-only), this was never checked.
        assert c.per_claim[cycles_id][0] == "UNGROUNDED", (
            f"Tier 1.1 broken: non-first cited claim not verified. "
            f"per_claim={c.per_claim}"
        )
        # Aggregate: worst wins → UNGROUNDED. Sentence is flagged.
        assert c.verdict == "UNGROUNDED", (
            f"Tier 1.1 broken: aggregate verdict should be UNGROUNDED, got {c.verdict}"
        )
        print("  Tier 1.1 OK: second cited claim flagged UNGROUNDED (old code missed it)")

        # --- Tier 1.3: claim_ids_used = cited ids, not retrieved ---
        # Answer cites density_id (twice) and cycles_id (once). Both retrieved.
        assert set(qr.claim_ids_used) == {density_id, cycles_id}, (
            f"Tier 1.3 broken: expected cited-only ids, got {qr.claim_ids_used}"
        )
        print("  Tier 1.3 OK: claim_ids_used = cited, not retrieved")

        # --- Tier 1.2: supersede invalidates cached views citing the claim ---
        assert store.stats()["cached_views"] == 1
        # create a new claim that will supersede the density claim
        sid = store.list_sources()[0]["id"]
        new_density_id = store.add_claim(
            sid, "nca cells", "have energy density of", "270 Wh/kg",
            0, 10, 0.9
        )
        store.supersede_claim(density_id, new_density_id)
        assert store.stats()["cached_views"] == 0, \
            "Tier 1.2 broken: supersede did not invalidate cache entry citing the claim"
        print(f"  Tier 1.2 OK: supersede invalidated the cached view citing [claim:{density_id}]")

        # --- Tier 1.3 stronger: a view that does NOT cite the superseded claim survives ---
        # cache a view that only mentions cycles_id (not the superseded density id)
        store.cache_view("just cycles?",
                         f"NCA cells cycle count is 1500 [claim:{cycles_id}].",
                         [cycles_id])
        assert store.stats()["cached_views"] == 1
        # supersede a *different* claim (the new_density_id) — this view didn't
        # cite it, so the cache entry must survive
        newer_density_id = store.add_claim(
            sid, "nca cells", "have energy density of", "280 Wh/kg", 0, 10, 0.9
        )
        store.supersede_claim(new_density_id, newer_density_id)
        assert store.stats()["cached_views"] == 1, \
            "Tier 1.3 broken: supersede of a non-cited claim wiped an unrelated cache entry"
        print("  Tier 1.3 OK: supersede of non-cited claim did not invalidate unrelated cache entry")

        # --- Tier 1.5 + 1.6 smoke: annotator handles the new verdict set and split merge works ---
        # (already exercised above via the multi-citation verification path)

    # --- Tier 4.5: ingest returns per-drop diagnostics ---
    class _DropFake:
        """Extracts one good claim and one with a span that can't be located."""
        def complete(self, system, user, max_tokens=2048):
            return "UNKNOWN"

        def complete_json(self, system, user, max_tokens=4096):
            if "You extract atomic claims" in system:
                return [
                    {"subject": "fox", "predicate": "is", "object": "red",
                     "span": "red fox jumps", "confidence": 0.9},
                    {"subject": "ghost", "predicate": "is", "object": "invisible",
                     "span": "this string is not in the source at all",
                     "confidence": 0.8},
                    {"subject": "", "predicate": "", "object": "",
                     "span": "", "confidence": 0.5},
                ]
            return {}

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src.md"
        src.write_text("The red fox jumps over the lazy dog.\n")
        store = Store(Path(d) / "drop.db")
        result = ingest_file(store, _DropFake(), src)
        assert "dropped" in result, f"Tier 4.5 broken: no 'dropped' key: {result}"
        assert result["claims_added"] == 1, result
        assert result["claims_dropped_ungrounded"] == 2, result
        reasons = {d["reason"] for d in result["dropped"]}
        assert "span_not_in_source" in reasons, f"missing reason: {reasons}"
        assert "missing_fields" in reasons, f"missing reason: {reasons}"
        # drop entries carry the claim context for debugging
        span_drop = [d for d in result["dropped"]
                     if d["reason"] == "span_not_in_source"][0]
        assert span_drop["subject"] == "ghost", span_drop
        assert "not in the source" in span_drop["span_preview"], span_drop
    print("  Tier 4.5 OK: ingest returns per-drop reason + claim context")

    print("  ALL TIER 1 ASSERTIONS PASSED")


def tier2_regression():
    """Tier 2: LLM retry/backoff and SQLite WAL mode."""
    print()
    print("=" * 60)
    print("TIER 2 REGRESSION CHECKS")
    print("=" * 60)

    # --- Tier 2.2: WAL mode + busy_timeout pragmas ---
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "wal.db")
        mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = s.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert mode.lower() == "wal", f"Tier 2.2 broken: journal_mode is {mode!r}, expected wal"
        assert timeout >= 5000, f"Tier 2.2 broken: busy_timeout is {timeout}, expected >= 5000"
        s.close()
    print(f"  Tier 2.2 OK: journal_mode=wal, busy_timeout={timeout}")

    # --- Tier 2.1: LLM retries transient errors ---
    # anthropic>=1.8 is built on httpx2; older releases on httpx.
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    from anthropic import APITimeoutError
    from aleph.llm import LLM

    llm = LLM(model="test", api_key="dummy", max_attempts=3, backoff_base=0.001)

    class _FakeBlock:
        type = "text"
        text = "ok"

    class _FakeResp:
        content = [_FakeBlock()]

    calls = [0]

    def fake_create(**kwargs):
        calls[0] += 1
        if calls[0] < 3:
            raise APITimeoutError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            )
        return _FakeResp()

    llm.client.messages.create = fake_create
    result = llm.complete("sys", "user")
    assert result == "ok", f"Tier 2.1 broken: expected 'ok', got {result!r}"
    assert calls[0] == 3, f"Tier 2.1 broken: expected 3 attempts, got {calls[0]}"
    print(f"  Tier 2.1 OK: LLM retried {calls[0] - 1} transient failures and succeeded")

    # --- Tier 2.1 negative: non-retryable errors bubble up immediately ---
    from anthropic import BadRequestError
    calls[0] = 0

    def always_bad(**kwargs):
        calls[0] += 1
        # minimally-constructed; SDK accepts message + response + body
        resp = httpx.Response(400, request=httpx.Request(
            "POST", "https://api.anthropic.com/v1/messages"
        ))
        raise BadRequestError("bad", response=resp, body=None)

    llm.client.messages.create = always_bad
    try:
        llm.complete("sys", "user")
    except BadRequestError:
        pass
    else:
        raise AssertionError("Tier 2.1 broken: BadRequestError should not be retried")
    assert calls[0] == 1, (
        f"Tier 2.1 broken: non-retryable error retried {calls[0]} times"
    )
    print("  Tier 2.1 OK: non-retryable errors raise on first attempt")

    print("  ALL TIER 2 ASSERTIONS PASSED")


def tier23_regression():
    """Tier 2.3: structured JSONL logging at decision points."""
    import logging
    import json as _json
    from aleph.log import set_level
    from aleph.ingest import _locate_span

    print()
    print("=" * 60)
    print("TIER 2.3 STRUCTURED LOGGING")
    print("=" * 60)

    set_level("debug")
    captured: list[dict] = []

    class _Cap(logging.Handler):
        def emit(self, rec):
            try:
                captured.append(_json.loads(rec.getMessage()))
            except Exception:
                pass

    aleph_logger = logging.getLogger("aleph")
    cap = _Cap()
    aleph_logger.addHandler(cap)

    try:
        # span_loose_match: the whitespace-tolerant fallback
        chunk = "The cat sat on the mat."
        located = _locate_span(chunk, "cat\nsat", 0)
        assert located == (4, 11), f"expected (4, 11), got {located}"

        # cache_invalidated: via supersede
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "t23.db")
            src_id = store.add_source("/fake", "Hello world")
            c1 = store.add_claim(src_id, "hello", "is", "world", 0, 11, 0.9)
            c2 = store.add_claim(src_id, "hello", "is", "world", 0, 11, 0.9)
            store.cache_view("q", f"Hi [claim:{c1}].", [c1])
            store.supersede_claim(c1, c2)
            store.close()
    finally:
        aleph_logger.removeHandler(cap)
        set_level("warning")

    event_names = {e.get("event") for e in captured}
    print(f"  captured events: {sorted(event_names)}")
    assert "span_loose_match" in event_names, (
        f"Tier 2.3 broken: missing span_loose_match in {event_names}"
    )
    assert "cache_invalidated" in event_names, (
        f"Tier 2.3 broken: missing cache_invalidated in {event_names}"
    )
    # every event has t, level, event keys
    for e in captured:
        assert set(("t", "level", "event")).issubset(e.keys()), (
            f"Tier 2.3 broken: malformed event {e}"
        )
    print("  Tier 2.3 OK: structured events emitted at decision points")


def tier3_regression():
    """Tier 3: retrieval — Retriever protocol, FTS5, inline span fetch."""
    from aleph.retrieval import (
        Retriever, KeywordRetriever, FTSRetriever, default_retriever,
    )
    from aleph.query import query as run_query

    print()
    print("=" * 60)
    print("TIER 3 REGRESSION CHECKS")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "cells.md"
        # source contains both "ion" as a substring (in "million") and as a
        # standalone word (in "lithium ion"). This is the classic failure mode
        # of substring-LIKE: it matches inside words.
        src.write_text(
            "Lithium ion cells store energy.\n"
            "Tesla makes a million cars a year.\n"
            "LFP cells cycle 3000 times.\n"
        )
        store = Store(Path(d) / "t3.db")
        assert store.fts_enabled, (
            "Tier 3.2 broken: FTS5 not enabled on a fresh store "
            "(SQLite lacks FTS5?)"
        )
        sid = store.add_source(str(src), src.read_text())
        # subject/predicate/object deliberately contain "ion" inside words
        c_lithium = store.add_claim(
            sid, "lithium ion", "store", "energy",
            0, 30, 0.9,
        )
        c_million = store.add_claim(
            sid, "tesla", "makes", "a million cars a year",
            32, 75, 0.9,
        )
        c_lfp = store.add_claim(
            sid, "lfp cell", "cycle", "3000 times",
            76, 105, 0.9,
        )

        # --- Tier 3.3: search_claims returns span_text inline ---
        rows = store.search_claims(["lithium"], limit=10)
        assert rows, "keyword retriever returned nothing for 'lithium'"
        assert "span_text" in rows[0].keys(), (
            f"Tier 3.3 broken: span_text not in row keys: {rows[0].keys()}"
        )
        assert rows[0]["span_text"], (
            f"Tier 3.3 broken: span_text empty: {dict(rows[0])}"
        )
        print("  Tier 3.3 OK: search_claims returns span_text inline (no N+1)")

        # --- Tier 3.2: FTS does word-boundary matching, keyword does not ---
        kw = KeywordRetriever(store)
        fts = FTSRetriever(store)
        kw_ids = {r["id"] for r in kw.search(["ion"], limit=10)}
        fts_ids = {r["id"] for r in fts.search(["ion"], limit=10)}
        # LIKE '%ion%' matches "lithium ion" (true positive) AND "million"
        # (false positive, since "ion" is inside "million"). FTS tokenizes so
        # "million" doesn't match the standalone token "ion".
        assert c_million in kw_ids, (
            f"keyword should match inside 'million', got {kw_ids}"
        )
        assert c_million not in fts_ids, (
            f"Tier 3.2 broken: FTS should not match 'ion' inside 'million', "
            f"got {fts_ids}"
        )
        assert c_lithium in fts_ids, (
            f"FTS should still match standalone 'ion' in 'lithium ion', "
            f"got {fts_ids}"
        )
        print("  Tier 3.2 OK: FTS5 word-boundary match beats keyword LIKE")

        # --- Tier 3.1: Retriever protocol can be plugged into query() ---
        # a custom Retriever that only ever returns the LFP claim, proving the
        # protocol wiring (query.query() calls retriever.search, not
        # store.search_claims directly)
        class OnlyLFP:
            def __init__(self, store):
                self.store = store

            def search(self, keywords, limit=30):
                rows = self.store.conn.execute(
                    "SELECT c.*, "
                    "SUBSTR(s.content, c.span_start+1, c.span_end-c.span_start) "
                    "AS span_text FROM claims c JOIN sources s "
                    "ON c.source_id = s.id WHERE c.id = ?",
                    (c_lfp,),
                ).fetchall()
                return rows

        class _StubLLM:
            def complete(self, system, user, max_tokens=2048):
                return f"LFP cycles 3000 times [claim:{c_lfp}]."

            def complete_json(self, system, user, max_tokens=4096):
                return {"verdict": "GROUNDED", "reason": "ok"}

        qr = run_query(
            store, _StubLLM(), "whatever", retriever=OnlyLFP(store),
        )
        assert qr.claim_ids_used == [c_lfp], (
            f"Tier 3.1 broken: custom retriever not honored, used {qr.claim_ids_used}"
        )
        print("  Tier 3.1 OK: custom Retriever plugs into query() cleanly")

        # --- default_retriever picks FTS when available ---
        default = default_retriever(store)
        assert isinstance(default, FTSRetriever), (
            f"Tier 3.1 broken: default_retriever should be FTS, got {type(default).__name__}"
        )
        print("  Tier 3.1 OK: default_retriever returns FTSRetriever when fts_enabled")

    print("  ALL TIER 3 ASSERTIONS PASSED")


def phase0_schema_regression():
    """Phase 0: schema foundation — new tables, columns, methods, cascades."""
    import sqlite3

    print()
    print("=" * 60)
    print("PHASE 0 SCHEMA REGRESSION CHECKS")
    print("=" * 60)

    # ---- 1. Fresh store: verify all new tables + columns ----
    with tempfile.TemporaryDirectory() as d:
        store = Store(Path(d) / "p0.db")

        # concepts table columns
        concept_cols = {
            r[1]: r[2]
            for r in store.conn.execute("PRAGMA table_info(concepts)").fetchall()
        }
        for col, typ in [
            ("id", "INTEGER"), ("subject", "TEXT"), ("statement", "TEXT"),
            ("inference_type", "TEXT"), ("confidence", "REAL"),
            ("status", "TEXT"), ("superseded_by", "INTEGER"),
            ("validation_verdict", "TEXT"), ("validation_reason", "TEXT"),
            ("last_validated_at", "REAL"), ("derived_at", "REAL"),
        ]:
            assert col in concept_cols, f"concepts missing column {col}"
            assert concept_cols[col] == typ, (
                f"concepts.{col} type {concept_cols[col]} != {typ}"
            )
        print("  concepts table: OK")

        # concept_supports table columns
        cs_cols = {
            r[1]: r[2]
            for r in store.conn.execute("PRAGMA table_info(concept_supports)").fetchall()
        }
        for col in ["concept_id", "claim_id", "role"]:
            assert col in cs_cols, f"concept_supports missing column {col}"
        print("  concept_supports table: OK")

        # claim_conditions table columns
        cc_cols = {
            r[1]: r[2]
            for r in store.conn.execute("PRAGMA table_info(claim_conditions)").fetchall()
        }
        for col in ["claim_id", "condition_claim_id", "kind", "explicit", "confidence"]:
            assert col in cc_cols, f"claim_conditions missing column {col}"
        print("  claim_conditions table: OK")

        # source_metadata table columns
        sm_cols = {
            r[1]: r[2]
            for r in store.conn.execute("PRAGMA table_info(source_metadata)").fetchall()
        }
        for col in ["source_id", "domain", "metadata", "updated_at"]:
            assert col in sm_cols, f"source_metadata missing column {col}"
        print("  source_metadata table: OK")

        # contradiction_rules table columns
        cr_cols = {
            r[1]: r[2]
            for r in store.conn.execute("PRAGMA table_info(contradiction_rules)").fetchall()
        }
        for col in ["contradiction_id", "rule", "applies_when", "rationale_concept_id", "decided_at"]:
            assert col in cr_cols, f"contradiction_rules missing column {col}"
        print("  contradiction_rules table: OK")

        # contradictions ALTER columns
        ct_cols = {
            r[1] for r in store.conn.execute("PRAGMA table_info(contradictions)").fetchall()
        }
        for col in ["kind", "disposition", "disposition_at", "candidate_disposition", "overlap_score"]:
            assert col in ct_cols, f"contradictions missing ALTER column {col}"
        print("  contradictions ALTER columns: OK")

        # view_cache concept_ids column
        vc_cols = {
            r[1] for r in store.conn.execute("PRAGMA table_info(view_cache)").fetchall()
        }
        assert "concept_ids" in vc_cols, "view_cache missing concept_ids column"
        print("  view_cache concept_ids column: OK")

        # ---- 2. Functional test: full cascade ----
        # source + claim
        src_id = store.add_source("/test/doc.md", "The quick brown fox jumps over the lazy dog.")
        assert src_id is not None
        c1 = store.add_claim(src_id, "fox", "jumps over", "lazy dog", 0, 44, 0.9)
        c2 = store.add_claim(src_id, "fox", "is", "quick", 4, 9, 0.8)
        c3 = store.add_claim(src_id, "fox", "is", "brown", 10, 15, 0.85)

        # concept + supports
        cpt_id = store.add_concept(
            "fox", "The fox is agile and brown",
            "summary", 0.8,
            [(c1, "premise"), (c2, "corroborating")],
            status="active",
        )
        assert cpt_id is not None
        cpt = store.get_concept(cpt_id)
        assert cpt["status"] == "active"
        assert cpt["inference_type"] == "summary"
        supports = store.get_concept_supports(cpt_id)
        assert len(supports) == 2
        concepts = store.list_concepts(subject="fox")
        assert len(concepts) == 1
        concepts_active = store.list_concepts(status="active")
        assert len(concepts_active) == 1
        print("  concept CRUD: OK")

        # claim_conditions
        store.add_claim_condition(c1, c2, "scope", True, 0.9)
        store.add_claim_condition(c1, c3, "limitation", False, 0.7)
        conds = store.get_claim_conditions(c1)
        assert len(conds) == 2
        # conditions_overlap: c1 has conditions {c2, c3}
        # create another claim with one overlapping condition
        c4 = store.add_claim(src_id, "fox", "is", "fast", 16, 20, 0.75)
        store.add_claim_condition(c4, c2, "scope", True, 0.9)
        overlap = store.conditions_overlap(c1, c4)
        # c1 conditions: {c2, c3}, c4 conditions: {c2}. Jaccard = 1/2 = 0.5
        assert abs(overlap - 0.5) < 1e-9, f"expected 0.5, got {overlap}"
        # no conditions => 0.0
        assert store.conditions_overlap(c2, c3) == 0.0
        print("  claim_conditions + overlap: OK")

        # contradiction with disposition
        ct_id = store.add_contradiction_with_kind(
            c1, c2, "categorical",
            candidate_disposition="coexist", overlap_score=0.5,
        )
        assert ct_id is not None
        # duplicate returns None
        assert store.add_contradiction_with_kind(c1, c2, "categorical") is None
        store.update_contradiction_disposition(
            ct_id, "coexist",
            rule="different facets",
            applies_when={"domain": "zoology"},
            rationale_concept_id=cpt_id,
            overlap_score=0.5,
        )
        full = store.list_contradictions_full(disposition="coexist")
        assert len(full) == 1
        assert full[0]["rule"] == "different facets"
        assert full[0]["status"] == "resolved"
        print("  contradiction disposition: OK")

        # source_metadata
        store.set_source_metadata(src_id, "scientific", {"authors": ["A"], "peer_reviewed": True})
        meta = store.get_source_metadata(src_id)
        assert meta is not None
        assert meta["domain"] == "scientific"
        assert meta["metadata"]["peer_reviewed"] is True
        by_domain = store.list_sources_by_domain("scientific")
        assert len(by_domain) == 1
        # invalid domain raises
        try:
            store.set_source_metadata(src_id, "invalid_domain", {})
            assert False, "should have raised ValueError"
        except ValueError:
            pass
        print("  source_metadata: OK")

        # cache a view citing concept + claims
        import json as _json
        store.conn.execute(
            "INSERT INTO view_cache (query_hash, query, response, claim_ids, concept_ids, generated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("hash1", "test query", "test response",
             _json.dumps([c1, c2]), _json.dumps([cpt_id]), 1.0),
        )
        store.conn.commit()
        assert store.stats()["cached_views"] == 1

        # retract_source: claims -> retracted, concepts -> stale, cache invalidated
        n_retracted = store.retract_source(src_id)
        assert n_retracted >= 2, f"expected >=2 retracted claims, got {n_retracted}"
        # check claim statuses
        claim1 = store.get_claim(c1)
        assert claim1["status"] == "retracted", f"claim status: {claim1['status']}"
        claim2 = store.get_claim(c2)
        assert claim2["status"] == "retracted", f"claim status: {claim2['status']}"
        # concept should be stale
        cpt_after = store.get_concept(cpt_id)
        assert cpt_after["status"] == "stale", f"concept status: {cpt_after['status']}"
        # cache should be invalidated
        assert store.stats()["cached_views"] == 0, "cache not invalidated after retract"
        print("  retract_source cascade: OK")

        # update_concept_status with invalidation
        # Re-cache a view citing the concept
        store.conn.execute(
            "INSERT INTO view_cache (query_hash, query, response, claim_ids, concept_ids, generated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("hash2", "test query 2", "test response 2",
             _json.dumps([]), _json.dumps([cpt_id]), 1.0),
        )
        store.conn.commit()
        assert store.stats()["cached_views"] == 1
        store.update_concept_status(cpt_id, "invalidated")
        assert store.stats()["cached_views"] == 0, "cache not invalidated after concept invalidation"
        print("  update_concept_status invalidation: OK")

        db_path = Path(d) / "p0.db"

        # ---- 3. Idempotent re-open ----
        store.close()
        store2 = Store(db_path)
        # Must not raise; schema is idempotent
        assert store2.get_concept(cpt_id) is not None
        store2.close()
        print("  idempotent re-open: OK")

    # ---- 4. Old DB migration test ----
    with tempfile.TemporaryDirectory() as d:
        old_db_path = Path(d) / "old.db"
        conn = sqlite3.connect(old_db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        # Write ONLY the pre-migration schema (no new tables, no ALTER columns)
        old_schema = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    ingested_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by INTEGER REFERENCES claims(id),
    extracted_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS subject_aliases (
    alias_from TEXT PRIMARY KEY,
    canonical_to TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    claim_b_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'open',
    resolved_to INTEGER REFERENCES claims(id),
    detected_at REAL NOT NULL,
    UNIQUE(claim_a_id, claim_b_id)
);

CREATE TABLE IF NOT EXISTS view_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL UNIQUE,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    claim_ids TEXT NOT NULL,
    generated_at REAL NOT NULL
);
"""
        conn.executescript(old_schema)
        # Insert some pre-existing data
        import time as _time
        conn.execute(
            "INSERT INTO sources (path, sha256, content, ingested_at) VALUES (?, ?, ?, ?)",
            ("/old/doc.txt", "abc123", "Old content here.", _time.time()),
        )
        conn.execute(
            "INSERT INTO claims (source_id, subject, predicate, object, span_start, span_end, "
            "confidence, extracted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "old", "is", "data", 0, 10, 0.9, _time.time()),
        )
        conn.execute(
            "INSERT INTO contradictions (claim_a_id, claim_b_id, detected_at) VALUES (?, ?, ?)",
            (1, 1, _time.time()),  # self-contradiction for testing; real data wouldn't do this
        )
        conn.execute(
            "INSERT INTO view_cache (query_hash, query, response, claim_ids, generated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("oldhash", "old query", "old response", "[1]", _time.time()),
        )
        conn.commit()
        conn.close()

        # Open with new Store -- must NOT raise
        store3 = Store(old_db_path)

        # Verify ALTERed columns exist on the old contradictions row
        ct_row = store3.conn.execute("SELECT * FROM contradictions WHERE id = 1").fetchone()
        assert ct_row["kind"] == "categorical", f"default kind: {ct_row['kind']}"
        assert ct_row["disposition"] == "unresolved", f"default disposition: {ct_row['disposition']}"

        # Verify ALTERed view_cache concept_ids defaults
        vc_row = store3.conn.execute("SELECT * FROM view_cache WHERE id = 1").fetchone()
        assert vc_row["concept_ids"] == "[]", f"default concept_ids: {vc_row['concept_ids']}"

        # New tables should exist
        tables = {
            r[0] for r in store3.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for t in ["concepts", "concept_supports", "claim_conditions",
                   "source_metadata", "contradiction_rules"]:
            assert t in tables, f"old DB migration missing table {t}"

        # Re-open again -- idempotent
        store3.close()
        store4 = Store(old_db_path)
        store4.close()
        print("  old DB migration + idempotent re-open: OK")

    print("  ALL PHASE 0 SCHEMA ASSERTIONS PASSED")


def tier4_concepts_regression():
    """Tier 4: Concepts layer — derive, validate, rebuild, invalidate, staleness cascades."""
    import json as _json
    import subprocess
    import sys

    print()
    print("=" * 60)
    print("TIER 4 CONCEPTS REGRESSION CHECKS")
    print("=" * 60)

    # -- FakeLLM for concepts that dispatches on prompt substrings --
    class ConceptFakeLLM:
        """Dispatches on system prompt substrings per 90-prompts.md."""
        def __init__(self):
            self.calls = 0

        def complete(self, system, user, max_tokens=2048):
            self.calls += 1
            return "UNKNOWN"

        def complete_json(self, system, user, max_tokens=4096):
            self.calls += 1

            # CONCEPT_DERIVE: "You propose concepts"
            if "You propose concepts" in system:
                # extract claim IDs from user message for realistic proposals
                import re
                ids = [int(m) for m in re.findall(r"\[claim:(\d+)\]", user)]
                if not ids:
                    return []
                return [{
                    "statement": "Tesla packs retain roughly 90% capacity at 200k miles across multiple fleet studies.",
                    "inference_type": "summary",
                    "confidence": 0.85,
                    "supports": [{"claim_id": cid, "role": "premise"} for cid in ids],
                }]

            # CONCEPT_VALIDATE: "You are a validator"
            if "You are a validator" in system:
                # If statement mentions something not in spans, UNGROUNDED
                # Check for overreach: if "cost" or "replacement" in statement
                # but not in spans, return PARTIAL
                stmt = ""
                if "Concept statement:" in user:
                    stmt = user.split("Concept statement:", 1)[1].split("\n")[0].strip()
                spans_text = user.lower()
                if "overreach" in stmt.lower() or "replacement" in stmt.lower():
                    return {"verdict": "PARTIAL", "reason": "statement includes elements not in spans"}
                return {"verdict": "GROUNDED", "reason": "every element of the statement is supported by at least one span"}

            # Existing verifier
            if "You are a verifier" in system:
                return {"verdict": "GROUNDED", "reason": "ok"}

            return {}

    # -- helpers to run agent CLI commands --
    def run_cmd(args_list):
        """Run aleph CLI command and return parsed JSON output."""
        result = subprocess.run(
            [sys.executable, "-m", "aleph"] + args_list,
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            return _json.loads(result.stdout.strip())
        return None

    # -- set up test data --
    fake = ConceptFakeLLM()
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "t4.db"
        store = Store(db_path)

        # create a source with content
        content = (
            "Tesla Model S packs retain about 90% of their original capacity "
            "after 200,000 miles. "
            "Independent teardowns found 87-92% retention in high-mileage fleets. "
            "Battery packs typically last well beyond 10 years under typical use. "
            "Replacement costs range from $12,000 to $18,000."
        )
        src_id = store.add_source("/test/tesla.md", content)

        # create 3 claims with spans that exist in the content
        c1 = store.add_claim(
            src_id, "tesla model s pack", "retain",
            "90% of original capacity after 200,000 miles",
            span_start=content.find("retain about 90%"),
            span_end=content.find("retain about 90%") + len("retain about 90% of their original capacity after 200,000 miles"),
            confidence=0.9,
        )
        c2 = store.add_claim(
            src_id, "tesla model s pack", "show",
            "87-92% retention in high-mileage fleets",
            span_start=content.find("87-92% retention"),
            span_end=content.find("87-92% retention") + len("87-92% retention in high-mileage fleets"),
            confidence=0.85,
        )
        c3 = store.add_claim(
            src_id, "tesla model s pack", "last",
            "well beyond 10 years under typical use",
            span_start=content.find("last well beyond"),
            span_end=content.find("last well beyond") + len("last well beyond 10 years under typical use"),
            confidence=0.8,
        )

        # ---- T4.1: concept-add happy path ----
        # Statement that IS grounded by the claims' spans
        from aleph.concepts import validate_concept, derive_concepts, rebuild_concept, invalidate_concept

        concept_id = store.add_concept(
            subject="tesla model s pack",
            statement="Tesla packs retain roughly 90% capacity at 200k miles across multiple fleet studies.",
            inference_type="summary",
            confidence=0.85,
            support_claim_ids=[(c1, "premise"), (c2, "corroborating"), (c3, "corroborating")],
            status="draft",
        )
        verdict, reason = validate_concept(store, fake, concept_id)
        row = store.get_concept(concept_id)
        assert row["status"] == "active", f"T4.1: expected active, got {row['status']}"
        assert row["validation_verdict"] == "GROUNDED", \
            f"T4.1: expected GROUNDED, got {row['validation_verdict']}"
        print(f"  Tier 4.1 OK: concept-add happy path -> status='active', verdict='GROUNDED'")

        # ---- T4.2: statement that overreaches -> draft, PARTIAL ----
        overreach_id = store.add_concept(
            subject="tesla model s pack",
            statement="Overreach: replacement costs are declining rapidly.",
            inference_type="synthesis",
            confidence=0.6,
            support_claim_ids=[(c1, "premise"), (c2, "corroborating")],
            status="draft",
        )
        v2, r2 = validate_concept(store, fake, overreach_id)
        row2 = store.get_concept(overreach_id)
        assert row2["status"] == "draft", \
            f"T4.2: expected draft, got {row2['status']}"
        assert row2["validation_verdict"] in ("UNGROUNDED", "PARTIAL"), \
            f"T4.2: expected UNGROUNDED or PARTIAL, got {row2['validation_verdict']}"
        print(f"  Tier 4.2 OK: overreaching statement -> status='draft', verdict='{row2['validation_verdict']}'")

        # ---- T4.3: concept-id in support -> error support_must_be_claim ----
        # The cmd_concept_add handler checks store.get_claim(id) for every
        # support id. If get_claim returns None, it rejects with error code
        # 'support_must_be_claim'. Use a non-existent id (99999) to exercise
        # this path — it doesn't matter whether it's "conceptually" a concept
        # or not; any id that isn't a claim must be rejected.
        import io, contextlib
        class _FakeArgs:
            subject = "tesla model s pack"
            statement = "Tesla packs retain capacity."
            inference_type = "summary"
            support = f"{c2}:premise,99999:corroborating"
            confidence = 0.7
            skip_validation = True
            model = None
        from aleph.agent_cli import cmd_concept_add
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_concept_add(_FakeArgs(), store)
        output = _json.loads(buf.getvalue())
        assert output["ok"] is False, f"T4.3: expected error, got {output}"
        assert output["error"]["code"] == "support_must_be_claim", \
            f"T4.3: expected error code 'support_must_be_claim', got {output['error']['code']}"
        print(f"  Tier 4.3 OK: non-claim id in support -> error 'support_must_be_claim'")

        # ---- T4.4: supersede one supporting claim -> concept becomes 'stale' ----
        # Create a new claim to supersede c1
        c1_new = store.add_claim(
            src_id, "tesla model s pack", "retain",
            "92% of original capacity after 200,000 miles",
            span_start=0, span_end=10,
            confidence=0.95,
        )
        store.supersede_claim(c1, c1_new)
        stale_row = store.get_concept(concept_id)
        assert stale_row["status"] == "stale", \
            f"T4.4: expected stale after claim supersede, got {stale_row['status']}"
        print(f"  Tier 4.4 OK: superseding supporting claim -> concept becomes 'stale'")

        # ---- T4.5: remove_source of supporting claim's source -> concept stale ----
        # Create a fresh concept with claims from a new source
        content2 = "Foxes are red. Foxes run fast."
        src2_id = store.add_source("/test/fox.md", content2)
        fox_c1 = store.add_claim(src2_id, "fox", "is", "red", 0, 14, 0.9)
        fox_c2 = store.add_claim(src2_id, "fox", "run", "fast", 15, 30, 0.85)
        fox_concept = store.add_concept(
            "fox", "Foxes are fast red animals.", "summary", 0.8,
            [(fox_c1, "premise"), (fox_c2, "premise")],
            status="active",
        )
        store.remove_source(src2_id)
        fox_row = store.get_concept(fox_concept)
        assert fox_row["status"] == "stale", \
            f"T4.5: expected stale after remove_source, got {fox_row['status']}"
        print(f"  Tier 4.5 OK: remove_source of supporting claim's source -> concept 'stale'")

        # ---- T4.6: alias-add that rewrites supporting-claim subject -> concept stale ----
        content3 = "The EV battery retains capacity. The EV battery lasts long."
        src3_id = store.add_source("/test/ev.md", content3)
        ev_c1 = store.add_claim(src3_id, "ev battery", "retains", "capacity",
                                content3.find("retains capacity"),
                                content3.find("retains capacity") + len("retains capacity"),
                                0.9)
        ev_c2 = store.add_claim(src3_id, "ev battery", "lasts", "long",
                                content3.find("lasts long"),
                                content3.find("lasts long") + len("lasts long"),
                                0.85)
        ev_concept = store.add_concept(
            "ev battery", "EV batteries retain capacity and last long.", "summary", 0.8,
            [(ev_c1, "premise"), (ev_c2, "premise")],
            status="active",
        )
        # verify active before alias
        assert store.get_concept(ev_concept)["status"] == "active"
        store.add_alias("ev battery", "electric vehicle battery")
        ev_row = store.get_concept(ev_concept)
        assert ev_row["status"] == "stale", \
            f"T4.6: expected stale after alias-add, got {ev_row['status']}"
        print(f"  Tier 4.6 OK: alias-add rewriting supporting-claim subject -> concept 'stale'")

        # ---- T4.7: concept-rebuild on stale -> NEW row, old superseded ----
        # Use the stale concept from T4.4 (concept_id)
        # First, make sure there are active claims for the subject still
        # c2, c3 are still active; c1 is superseded, c1_new is active
        stale_before = store.get_concept(concept_id)
        assert stale_before["status"] == "stale", f"precondition: expected stale, got {stale_before['status']}"

        new_concept_id = rebuild_concept(store, fake, concept_id)
        assert new_concept_id != concept_id, \
            f"T4.7: rebuild must create a NEW concept id, got same {new_concept_id}"
        old_after = store.get_concept(concept_id)
        assert old_after["status"] == "superseded", \
            f"T4.7: old concept should be superseded, got {old_after['status']}"
        assert old_after["superseded_by"] == new_concept_id, \
            f"T4.7: old concept superseded_by should be {new_concept_id}, got {old_after['superseded_by']}"
        new_row = store.get_concept(new_concept_id)
        assert new_row is not None, "T4.7: new concept row should exist"
        print(f"  Tier 4.7 OK: concept-rebuild creates new row ({new_concept_id}), old ({concept_id}) superseded")

        # ---- T4.8: cache a view citing the concept -> invalidate -> view gone ----
        # Cache a view that cites a concept
        test_concept = store.add_concept(
            "test", "Test concept for cache invalidation.", "summary", 0.8,
            [(c2, "premise")],
            status="active",
        )
        # Insert view_cache row with concept_ids
        store.conn.execute(
            "INSERT INTO view_cache (query_hash, query, response, claim_ids, concept_ids, generated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("t4hash", "test concept query", "Answer citing concept [concept:{}]".format(test_concept),
             _json.dumps([c2]), _json.dumps([test_concept]), 1.0),
        )
        store.conn.commit()
        cached_before = store.conn.execute(
            "SELECT COUNT(*) FROM view_cache WHERE query_hash = 't4hash'"
        ).fetchone()[0]
        assert cached_before == 1, f"T4.8 precondition: expected 1 cached view, got {cached_before}"

        invalidate_concept(store, test_concept, "testing invalidation cascade")
        cached_after = store.conn.execute(
            "SELECT COUNT(*) FROM view_cache WHERE query_hash = 't4hash'"
        ).fetchone()[0]
        assert cached_after == 0, \
            f"T4.8: expected 0 cached views after invalidation, got {cached_after}"
        inv_row = store.get_concept(test_concept)
        assert inv_row["status"] == "invalidated", \
            f"T4.8: expected invalidated, got {inv_row['status']}"
        print(f"  Tier 4.8 OK: concept-invalidate drops cached view citing the concept")

        store.close()

    print("  ALL TIER 4 ASSERTIONS PASSED")


def tier5_contradictions_regression():
    """Tier 5: Contradictions & Dispositions — detect, classify, dispose."""
    import json as _json

    print()
    print("=" * 60)
    print("TIER 5 CONTRADICTIONS REGRESSION CHECKS")
    print("=" * 60)

    from aleph.contradictions import (
        classify_pair, detect_all, dispose, _ALLOWED_DISPOSITIONS,
    )
    from aleph.lint import lint as lint_cmd

    # -- FakeLLM that dispatches on the DETECT prompt substring --
    class DetectFakeLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, system, user, max_tokens=2048):
            self.calls += 1
            return "UNKNOWN"

        def complete_json(self, system, user, max_tokens=4096):
            self.calls += 1

            # DETECT prompt dispatch
            if "You judge whether two claims" in system:
                # numeric: "8 years" vs "10 years"
                if "8 years" in user and "10 years" in user:
                    return {
                        "contradicts": True,
                        "kind": "numeric",
                        "candidate_disposition": "supersede",
                        "reason": "different numeric values for same measure",
                    }
                # negation: "is" vs "is not"
                if "is not" in user:
                    return {
                        "contradicts": True,
                        "kind": "negation",
                        "candidate_disposition": "dispute",
                        "reason": "direct negation of predicate",
                    }
                # default: no contradiction
                return {
                    "contradicts": False,
                    "kind": "unknown",
                    "candidate_disposition": "unresolved",
                    "reason": "different facets",
                }

            return {}

    fake = DetectFakeLLM()

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "t5.db"
        store = Store(db_path)

        # Create source and claims for testing
        content = (
            "Battery life is 8 years on average. "
            "Battery life is 10 years on average. "
            "Battery is safe. "
            "Battery is not safe. "
            "Battery has 3000 cycles. "
            "Battery has good performance. "
        )
        src_id = store.add_source("/test/battery.md", content)

        # claims: numeric pair
        c1 = store.add_claim(
            src_id, "battery life", "is",
            "8 years on average",
            span_start=content.find("8 years on average"),
            span_end=content.find("8 years on average") + len("8 years on average"),
            confidence=0.9,
        )
        c2 = store.add_claim(
            src_id, "battery life", "is",
            "10 years on average",
            span_start=content.find("10 years on average"),
            span_end=content.find("10 years on average") + len("10 years on average"),
            confidence=0.85,
        )

        # claims: negation pair — objects differ so classify_pair doesn't
        # short-circuit on identical objects
        c3 = store.add_claim(
            src_id, "battery", "is",
            "safe for consumers",
            span_start=content.find("Battery is safe"),
            span_end=content.find("Battery is safe") + len("Battery is safe"),
            confidence=0.8,
        )
        c4 = store.add_claim(
            src_id, "battery", "is not",
            "safe enough",
            span_start=content.find("Battery is not safe"),
            span_end=content.find("Battery is not safe") + len("Battery is not safe"),
            confidence=0.75,
        )

        # claims: non-contradicting pair
        c5 = store.add_claim(
            src_id, "battery", "has",
            "3000 cycles",
            span_start=content.find("3000 cycles"),
            span_end=content.find("3000 cycles") + len("3000 cycles"),
            confidence=0.9,
        )
        c6 = store.add_claim(
            src_id, "battery", "has",
            "good performance",
            span_start=content.find("good performance"),
            span_end=content.find("good performance") + len("good performance"),
            confidence=0.85,
        )

        # ---- T5.1: Numeric pre-filter ----
        # classify_pair on the numeric pair
        candidate = classify_pair(store, store.get_claim(c1), store.get_claim(c2))
        assert candidate is not None, "T5.1: classify_pair should not return None for numeric pair"
        assert candidate.kind == "numeric", \
            f"T5.1: expected kind='numeric', got {candidate.kind!r}"
        print("  Tier 5.1 OK: numeric pre-filter classifies (8 years vs 10 years) as kind='numeric'")

        # ---- T5.2: Negation pre-filter ----
        candidate_neg = classify_pair(store, store.get_claim(c3), store.get_claim(c4))
        assert candidate_neg is not None, "T5.2: classify_pair should not return None for negation pair"
        assert candidate_neg.kind == "negation", \
            f"T5.2: expected kind='negation', got {candidate_neg.kind!r}"
        print("  Tier 5.2 OK: negation pre-filter classifies (is vs is not) as kind='negation'")

        # ---- T5.3: detect_all populates kind and candidate_disposition ----
        summary = detect_all(store, fake)
        assert summary["contradictions_confirmed"] >= 2, \
            f"T5.3: expected >= 2 confirmed, got {summary['contradictions_confirmed']}"
        assert "numeric" in summary["by_kind"], \
            f"T5.3: expected 'numeric' in by_kind, got {summary['by_kind']}"
        # Check that written rows have kind and candidate_disposition
        all_ct = store.conn.execute(
            "SELECT * FROM contradictions WHERE kind IS NOT NULL"
        ).fetchall()
        assert len(all_ct) >= 2, \
            f"T5.3: expected >= 2 contradictions with kind, got {len(all_ct)}"
        has_numeric = any(r["kind"] == "numeric" for r in all_ct)
        assert has_numeric, "T5.3: no contradiction with kind='numeric' found"
        has_cand_disp = any(
            r["candidate_disposition"] is not None and r["candidate_disposition"] != "unresolved"
            for r in all_ct
        )
        assert has_cand_disp, "T5.3: no contradiction with a non-unresolved candidate_disposition"
        print("  Tier 5.3 OK: detect_all populates kind and candidate_disposition on rows")

        # find the numeric contradiction for dispose tests
        numeric_ct = store.conn.execute(
            "SELECT * FROM contradictions WHERE kind = 'numeric'"
        ).fetchone()
        assert numeric_ct is not None

        # ---- T5.4: Dispose supersede cascades ----
        # cache a view citing c1 so we can test invalidation
        store.cache_view("battery life?",
                         f"Battery lasts 8 years [claim:{c1}].",
                         [c1])
        assert store.stats()["cached_views"] >= 1

        # Make c2 newer
        store.conn.execute(
            "UPDATE claims SET extracted_at = extracted_at + 100 WHERE id = ?",
            (c2,),
        )
        store.conn.commit()

        result = dispose(store, numeric_ct["id"], "supersede", keep=c2, drop=c1)
        assert result["applied"] is True, f"T5.4: dispose returned {result}"
        # c1 should be superseded
        c1_row = store.get_claim(c1)
        assert c1_row["status"] == "superseded", \
            f"T5.4: expected c1 superseded, got {c1_row['status']}"
        # cache should be invalidated
        assert store.stats()["cached_views"] == 0, \
            "T5.4: cached view citing superseded claim should be gone"
        print("  Tier 5.4 OK: dispose supersede cascades claim supersede + cache invalidation")

        # ---- T5.5: Dispose coexist without rule -> error ----
        negation_ct = store.conn.execute(
            "SELECT * FROM contradictions WHERE kind = 'negation'"
        ).fetchone()
        assert negation_ct is not None, "T5.5: no negation contradiction found"

        try:
            dispose(store, negation_ct["id"], "coexist")
            assert False, "T5.5: should have raised ValueError"
        except ValueError as e:
            assert "rule_required_for_disposition" in str(e), \
                f"T5.5: expected 'rule_required_for_disposition', got {e}"
        print("  Tier 5.5 OK: dispose coexist without rule -> error 'rule_required_for_disposition'")

        # ---- T5.6: Dispose coexist with rule -> both active + rule row ----
        result = dispose(
            store, negation_ct["id"], "coexist",
            rule="safety depends on usage context",
            applies_when={"context": "laboratory"},
        )
        assert result["applied"] is True
        # both claims should still be active
        c3_row = store.get_claim(c3)
        c4_row = store.get_claim(c4)
        assert c3_row["status"] == "active", \
            f"T5.6: c3 should be active, got {c3_row['status']}"
        assert c4_row["status"] == "active", \
            f"T5.6: c4 should be active, got {c4_row['status']}"
        # contradiction_rules row should exist
        rule_row = store.conn.execute(
            "SELECT * FROM contradiction_rules WHERE contradiction_id = ?",
            (negation_ct["id"],),
        ).fetchone()
        assert rule_row is not None, "T5.6: contradiction_rules row should exist"
        assert rule_row["rule"] == "safety depends on usage context", \
            f"T5.6: unexpected rule: {rule_row['rule']}"
        print("  Tier 5.6 OK: dispose coexist with rule -> both active + rule row written")

        # ---- T5.7: Dispose reconcile with rationale_concept_id ----
        # Create a new contradiction for this test
        c7 = store.add_claim(
            src_id, "battery", "degrades",
            "5% per year",
            span_start=content.find("Battery has good performance"),
            span_end=content.find("Battery has good performance") + 10,
            confidence=0.8,
        )
        c8 = store.add_claim(
            src_id, "battery", "degrades",
            "2% per year",
            span_start=content.find("Battery has good performance"),
            span_end=content.find("Battery has good performance") + 10,
            confidence=0.75,
        )
        ct_reconcile = store.add_contradiction_with_kind(c7, c8, "numeric")
        assert ct_reconcile is not None

        # Create an ACTIVE concept for rationale
        active_concept = store.add_concept(
            "battery", "Battery degradation depends on conditions.", "summary", 0.8,
            [(c5, "premise")],
            status="active",
        )
        # Create a DRAFT concept
        draft_concept = store.add_concept(
            "battery", "Draft concept.", "summary", 0.5,
            [(c5, "premise")],
            status="draft",
        )

        # draft concept -> error rationale_concept_not_active
        try:
            dispose(
                store, ct_reconcile, "reconcile",
                rule="conditions differ",
                rationale_concept_id=draft_concept,
            )
            assert False, "T5.7: should have raised ValueError for draft concept"
        except ValueError as e:
            assert "rationale_concept_not_active" in str(e), \
                f"T5.7: expected 'rationale_concept_not_active', got {e}"

        # active concept -> accepted
        result = dispose(
            store, ct_reconcile, "reconcile",
            rule="conditions differ",
            rationale_concept_id=active_concept,
        )
        assert result["applied"] is True
        print("  Tier 5.7 OK: reconcile with active concept accepted; draft concept rejected")

        # ---- T5.8: Dispose replicate -> confidence bump ----
        c9 = store.add_claim(
            src_id, "battery", "weighs",
            "500 kg",
            span_start=content.find("Battery has good"),
            span_end=content.find("Battery has good") + 10,
            confidence=0.80,
        )
        c10 = store.add_claim(
            src_id, "battery", "weighs",
            "500 kilograms",
            span_start=content.find("Battery has good"),
            span_end=content.find("Battery has good") + 10,
            confidence=0.90,
        )
        ct_replicate = store.add_contradiction_with_kind(c9, c10, "categorical")
        assert ct_replicate is not None

        conf_before_9 = store.get_claim(c9)["confidence"]
        conf_before_10 = store.get_claim(c10)["confidence"]

        result = dispose(store, ct_replicate, "replicate")
        assert result["applied"] is True

        conf_after_9 = store.get_claim(c9)["confidence"]
        conf_after_10 = store.get_claim(c10)["confidence"]
        assert abs(conf_after_9 - (conf_before_9 + 0.05)) < 1e-9, \
            f"T5.8: c9 confidence should be {conf_before_9 + 0.05}, got {conf_after_9}"
        assert abs(conf_after_10 - (conf_before_10 + 0.05)) < 1e-9, \
            f"T5.8: c10 confidence should be {conf_before_10 + 0.05}, got {conf_after_10}"
        print("  Tier 5.8 OK: dispose replicate bumps both claims' confidence by 0.05")

        # ---- T5.8b: replicate confidence cap at 1.0 ----
        # set c9 confidence to 0.99 and replicate again
        store.conn.execute(
            "UPDATE claims SET confidence = 0.99 WHERE id = ?", (c9,)
        )
        store.conn.commit()
        ct_cap = store.add_contradiction_with_kind(c9, c10, "categorical")
        if ct_cap is not None:
            dispose(store, ct_cap, "replicate")
            conf_capped = store.get_claim(c9)["confidence"]
            assert conf_capped <= 1.0, \
                f"T5.8b: confidence should be capped at 1.0, got {conf_capped}"
            print("  Tier 5.8b OK: replicate confidence capped at 1.0")

        # ---- T5.9: Legacy lint still works ----
        # Create a fresh store for lint test to avoid interference
        lint_store = Store(Path(d) / "t5_lint.db")
        lint_content = (
            "Tesla batteries last 8 years on average. "
            "Tesla batteries last well beyond 10 years under typical use. "
            "LFP cells handle 3000 cycles. "
            "NCA cells handle 1500 cycles. "
        )
        lint_src = lint_store.add_source("/test/lint.md", lint_content)
        lint_store.add_claim(
            lint_src, "tesla battery", "last",
            "8 years on average",
            span_start=lint_content.find("8 years on average"),
            span_end=lint_content.find("8 years on average") + len("8 years on average"),
            confidence=0.9,
        )
        lint_store.add_claim(
            lint_src, "tesla battery", "last",
            "well beyond 10 years under typical use",
            span_start=lint_content.find("well beyond 10 years"),
            span_end=lint_content.find("well beyond 10 years") + len("well beyond 10 years under typical use"),
            confidence=0.85,
        )
        lint_store.add_claim(
            lint_src, "lfp cell", "handle",
            "3000 cycles",
            span_start=lint_content.find("3000 cycles"),
            span_end=lint_content.find("3000 cycles") + len("3000 cycles"),
            confidence=0.9,
        )
        lint_store.add_claim(
            lint_src, "nca cell", "handle",
            "1500 cycles",
            span_start=lint_content.find("1500 cycles"),
            span_end=lint_content.find("1500 cycles") + len("1500 cycles"),
            confidence=0.85,
        )

        lint_fake = DetectFakeLLM()
        lint_result = lint_cmd(lint_store, lint_fake)
        assert "candidate_groups" in lint_result, \
            f"T5.9: lint result missing 'candidate_groups': {lint_result}"
        assert "pairs_checked" in lint_result, \
            f"T5.9: lint result missing 'pairs_checked': {lint_result}"
        assert "contradictions_confirmed" in lint_result, \
            f"T5.9: lint result missing 'contradictions_confirmed': {lint_result}"
        assert lint_result["contradictions_confirmed"] >= 0, \
            f"T5.9: contradictions_confirmed should be >= 0"
        lint_store.close()
        print("  Tier 5.9 OK: legacy lint(store, llm) still works with backward-compatible keys")

        store.close()

    print("  ALL TIER 5 ASSERTIONS PASSED")


def tier6_authority_regression():
    """Tier 6: Source Metadata & Authority — schemas, validation, ranking, retraction cascade."""
    import json as _json
    import time as _time

    print()
    print("=" * 60)
    print("TIER 6 AUTHORITY REGRESSION CHECKS")
    print("=" * 60)

    from aleph.authority import (
        validate_metadata, authority_rank, is_retracted, is_effective_at,
        set_metadata, retract_source, unretract_source,
        DOMAIN_SCHEMAS,
    )

    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "tier6.db"
        store = Store(db)

        # --- setup: add a source with content ---
        src_content = (
            "Tesla battery packs retain about 90% of their original capacity "
            "after 200,000 miles of driving according to fleet data."
        )
        sid = store.add_source("science_paper.md", src_content)
        assert sid is not None, "source should be created"

        # add a second source for legal tests
        legal_content = "California Civil Code Section 1942.5 protects tenants from retaliation."
        legal_sid = store.add_source("legal_doc.md", legal_content)
        assert legal_sid is not None

        # ---- T6.1: Valid legal metadata -> stored, no validation_errors ----
        legal_meta = {
            "jurisdiction": "US-CA",
            "authority_type": "statute",
            "authority_level": 3,
            "specificity": 2,
            "issued_at": 1577836800,
        }
        result = set_metadata(store, legal_sid, "legal", legal_meta)
        assert result["validation_errors"] == [], \
            f"T6.1 expected no errors, got {result['validation_errors']}"
        assert result["source_id"] == legal_sid
        assert result["domain"] == "legal"
        # verify it persisted
        persisted = store.get_source_metadata(legal_sid)
        assert persisted is not None
        assert persisted["domain"] == "legal"
        assert persisted["metadata"]["jurisdiction"] == "US-CA"
        print("  Tier 6.1 OK: valid legal metadata stored with no validation_errors")

        # ---- T6.2: Wrong-type field -> validation_errors; strict refuses ----
        bad_meta = {
            "jurisdiction": "US-CA",
            "expires_at": "not-a-number",  # should be int or float
        }
        errors = validate_metadata("legal", bad_meta)
        assert len(errors) == 1, f"T6.2 expected 1 error, got {len(errors)}"
        assert errors[0].field == "expires_at"
        assert "int or float" in errors[0].reason

        # set_metadata still writes (lenient mode)
        result_lenient = set_metadata(store, legal_sid, "legal", bad_meta)
        assert len(result_lenient["validation_errors"]) == 1
        persisted2 = store.get_source_metadata(legal_sid)
        assert persisted2["metadata"]["expires_at"] == "not-a-number", \
            "lenient mode should still persist"

        # Simulate strict mode: check errors first, refuse if any
        errors_strict = validate_metadata("legal", bad_meta)
        assert len(errors_strict) > 0, "strict should have errors to refuse on"
        # (The CLI handler checks errors and returns _err before calling set_metadata)

        # Restore valid metadata for further tests
        set_metadata(store, legal_sid, "legal", legal_meta)
        print("  Tier 6.2 OK: wrong-type field produces validation_errors; strict refuses")

        # ---- T6.3: Unknown field -> stored with no error ----
        meta_with_unknown = {
            "jurisdiction": "US-CA",
            "color": "blue",  # not in schema
        }
        errors_unk = validate_metadata("legal", meta_with_unknown)
        assert len(errors_unk) == 0, \
            f"T6.3 expected 0 errors for unknown field, got {len(errors_unk)}"
        result_unk = set_metadata(store, legal_sid, "legal", meta_with_unknown)
        assert result_unk["validation_errors"] == []
        persisted3 = store.get_source_metadata(legal_sid)
        assert persisted3["metadata"]["color"] == "blue", "unknown field should persist"
        print("  Tier 6.3 OK: unknown field stored with no error")

        # ---- T6.4: authority_rank ordering ----
        # Higher authority_level should rank higher regardless of specificity
        high_rank = {"domain": "legal", "metadata": {
            "authority_level": 5, "specificity": 2, "issued_at": 100,
        }}
        low_rank = {"domain": "legal", "metadata": {
            "authority_level": 3, "specificity": 9, "issued_at": 100,
        }}
        rank_high = authority_rank(high_rank)
        rank_low = authority_rank(low_rank)
        assert rank_high > rank_low, \
            f"T6.4 expected {rank_high} > {rank_low} (level is primary sort)"

        # Scientific ranking
        sci_high = {"domain": "scientific", "metadata": {
            "peer_reviewed": True, "citation_count": 50, "published_at": 100,
        }}
        sci_low = {"domain": "scientific", "metadata": {
            "peer_reviewed": False, "citation_count": 200, "published_at": 100,
        }}
        sci_rank_high = authority_rank(sci_high)
        sci_rank_low = authority_rank(sci_low)
        assert sci_rank_high > sci_rank_low, \
            f"T6.4 expected {sci_rank_high} > {sci_rank_low} (peer_reviewed is primary sort)"
        print("  Tier 6.4 OK: authority_rank: level is primary sort for legal, peer_reviewed for scientific")

        # ---- T6.5: Retract cascade ----
        # Setup: scientific source with claim, concept, and cached view
        set_metadata(store, sid, "scientific", {
            "peer_reviewed": True,
            "venue": "Nature",
            "citation_count": 100,
        })

        # Add claim grounded in source
        span = "retain about 90% of their original capacity"
        idx = src_content.find(span)
        assert idx >= 0, "span must be in source"
        claim_id = store.add_claim(
            source_id=sid,
            subject="tesla battery",
            predicate="retain",
            object_="90% capacity at 200k mi",
            span_start=idx,
            span_end=idx + len(span),
            confidence=0.9,
        )

        # Add concept supported by this claim
        concept_id = store.add_concept(
            subject="tesla battery",
            statement="Tesla packs retain ~90% capacity",
            inference_type="summary",
            confidence=0.85,
            support_claim_ids=[(claim_id, "premise")],
            status="active",
        )

        # Cache a view citing this claim
        store.cache_view("how long do tesla batteries last?",
                         "They retain 90% [claim:{}].".format(claim_id),
                         [claim_id])
        cached = store.get_cached_view("how long do tesla batteries last?")
        assert cached is not None, "view should be cached"
        # view-cache / view-get and `ask` must agree on the key.
        from aleph.query import _compute_cache_hash
        assert cached["query_hash"] == _compute_cache_hash(
            "How long do Tesla batteries last?", None), \
            "agent-mode and ask cache keys must match"

        # NOW RETRACT
        retract_result = retract_source(store, sid, "data fabrication")
        assert "error" not in retract_result, f"retract should succeed: {retract_result}"
        assert retract_result["claims_retracted"] >= 1, "should retract at least 1 claim"

        # Verify claim is retracted
        claim_row = store.get_claim(claim_id)
        assert claim_row["status"] == "retracted", \
            f"T6.5 claim should be retracted, got {claim_row['status']}"

        # Verify concept is stale
        concept_row = store.get_concept(concept_id)
        assert concept_row["status"] == "stale", \
            f"T6.5 concept should be stale, got {concept_row['status']}"

        # Verify cached view is gone
        cached_after = store.get_cached_view("how long do tesla batteries last?")
        assert cached_after is None, "T6.5 cached view should be invalidated"

        # Verify metadata updated
        meta_after = store.get_source_metadata(sid)
        assert meta_after["metadata"]["retracted"] is True
        assert meta_after["metadata"]["retraction_reason"] == "data fabrication"
        print("  Tier 6.5 OK: retract cascade: claim=retracted, concept=stale, view=invalidated")

        # ---- T6.6: Retract non-scientific -> error ----
        # Restore legal metadata on legal_sid
        set_metadata(store, legal_sid, "legal", legal_meta)
        err_result = retract_source(store, legal_sid, "some reason")
        assert err_result.get("error") == "not_scientific_domain", \
            f"T6.6 expected not_scientific_domain, got {err_result}"
        print("  Tier 6.6 OK: retract_source on non-scientific returns error not_scientific_domain")

        # ---- T6.7: Unretract reverses ----
        unretract_result = unretract_source(store, sid, "erratum correction")
        assert "error" not in unretract_result, f"unretract should succeed: {unretract_result}"

        # Claim should be back to active
        claim_row2 = store.get_claim(claim_id)
        assert claim_row2["status"] == "active", \
            f"T6.7 claim should be active after unretract, got {claim_row2['status']}"

        # Metadata should show retracted=False
        meta_unret = store.get_source_metadata(sid)
        assert meta_unret["metadata"]["retracted"] is False, \
            f"T6.7 metadata.retracted should be False, got {meta_unret['metadata']['retracted']}"
        assert "retracted_at" not in meta_unret["metadata"], \
            "T6.7 retracted_at should be cleared after unretract"
        print("  Tier 6.7 OK: unretract reverses: claim=active, metadata.retracted=False")

        # ---- Additional: is_retracted / is_effective_at helpers ----
        assert is_retracted({"domain": "scientific", "metadata": {"retracted": True}}) is True
        assert is_retracted({"domain": "scientific", "metadata": {"retracted": False}}) is False
        assert is_retracted({"domain": "legal", "metadata": {"retracted": True}}) is False
        assert is_retracted({"domain": "scientific", "metadata": {}}) is False
        assert is_retracted({}) is False  # just inner dict, no domain

        now = _time.time()
        assert is_effective_at({"effective_at": now - 100, "expires_at": now + 100}, now) is True
        assert is_effective_at({"effective_at": now + 100}, now) is False
        assert is_effective_at({"expires_at": now - 100}, now) is False
        assert is_effective_at({}, now) is True  # no fields = effective
        print("  Tier 6.8 OK: is_retracted and is_effective_at helpers correct")

        # ---- Cleanup: unknown domain validation ----
        domain_errors = validate_metadata("martian_law", {"field": "value"})
        assert len(domain_errors) == 1
        assert domain_errors[0].field == "domain"
        print("  Tier 6.9 OK: unknown domain fails validation")

        store.close()

    print("  ALL TIER 6 ASSERTIONS PASSED")


def tier7_conditions_regression():
    """Tier 7: Claim Conditions & Scientific Ingestion (WS-D)."""
    import json as _json
    import io
    import contextlib

    print()
    print("=" * 60)
    print("TIER 7 CONDITIONS REGRESSION CHECKS")
    print("=" * 60)

    from aleph.conditions import (
        link_condition, unlink_condition, conditions_for_claim, overlap,
        VALID_KINDS, extract_scope_from_source, infer_conditions,
    )
    from aleph.agent_cli import (
        cmd_claim_condition_add, cmd_claim_condition_remove,
        cmd_claim_conditions_list, cmd_claim_add,
    )
    from aleph.ingest import ingest_file

    # -- FakeLLM for conditions that dispatches on prompt substrings --
    class ConditionsFakeLLM:
        """Dispatches on system prompt substrings per 90-prompts.md."""
        def __init__(self):
            self.calls = 0

        def complete(self, system, user, max_tokens=2048):
            self.calls += 1
            return "UNKNOWN"

        def complete_json(self, system, user, max_tokens=4096):
            self.calls += 1

            # EXTRACT_SCOPE: "You extract SCOPE claims"
            if "You extract SCOPE claims" in system:
                return [
                    {
                        "subject": "the study",
                        "predicate": "used",
                        "object": "BALB/c mice aged 8-12 weeks",
                        "span": "used BALB/c mice aged 8-12 weeks",
                        "kind": "sample",
                        "confidence": 0.95,
                    },
                    {
                        "subject": "the experiment",
                        "predicate": "employed",
                        "object": "double-blind randomized controlled trial",
                        "span": "double-blind randomized controlled trial",
                        "kind": "method",
                        "confidence": 0.9,
                    },
                ]

            # EXTRACT_CONDITIONS: "You propose which EXISTING"
            if "You propose which EXISTING" in system:
                # Parse out the available scope claim IDs from user msg
                import re
                ids = [int(m) for m in re.findall(
                    r"\[claim:(\d+)\] kind=", user)]
                applies = []
                for cid in ids:
                    applies.append({
                        "condition_claim_id": cid,
                        "kind": "scope",
                        "explicit": False,
                        "reason": "document-wide scope",
                    })
                return {"applies": applies}

            # Existing atomic extraction
            if "You extract atomic claims" in system:
                return [
                    {
                        "subject": "treatment group",
                        "predicate": "showed",
                        "object": "50% tumor reduction",
                        "span": "treatment group showed 50% tumor reduction",
                        "confidence": 0.9,
                    },
                    {
                        "subject": "control group",
                        "predicate": "showed",
                        "object": "no significant change",
                        "span": "control group showed no significant change",
                        "confidence": 0.85,
                    },
                ]

            return {}

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "t7.db"
        store = Store(db_path)

        # Create a source with scientific content
        content = (
            "This study used BALB/c mice aged 8-12 weeks in a "
            "double-blind randomized controlled trial. "
            "The treatment group showed 50% tumor reduction while the "
            "control group showed no significant change. "
            "The sample size was 200 subjects across 4 cohorts."
        )
        src_id = store.add_source("/test/science.md", content)

        # Create some claims for condition testing
        c1 = store.add_claim(
            src_id, "treatment group", "showed",
            "50% tumor reduction",
            span_start=content.find("treatment group showed"),
            span_end=content.find("treatment group showed")
            + len("treatment group showed 50% tumor reduction"),
            confidence=0.9,
        )
        c2 = store.add_claim(
            src_id, "control group", "showed",
            "no significant change",
            span_start=content.find("control group showed"),
            span_end=content.find("control group showed")
            + len("control group showed no significant change"),
            confidence=0.85,
        )
        # A scope claim (manually added for unit tests)
        c3 = store.add_claim(
            src_id, "the study", "applies-to",
            "BALB/c mice aged 8-12 weeks",
            span_start=content.find("used BALB/c mice"),
            span_end=content.find("used BALB/c mice")
            + len("used BALB/c mice aged 8-12 weeks"),
            confidence=0.95,
        )
        c4 = store.add_claim(
            src_id, "the experiment", "applies-to",
            "double-blind randomized controlled trial",
            span_start=content.find("double-blind randomized"),
            span_end=content.find("double-blind randomized")
            + len("double-blind randomized controlled trial"),
            confidence=0.9,
        )
        c5 = store.add_claim(
            src_id, "the sample", "applies-to",
            "200 subjects across 4 cohorts",
            span_start=content.find("200 subjects"),
            span_end=content.find("200 subjects")
            + len("200 subjects across 4 cohorts"),
            confidence=0.85,
        )

        # ---- T7.1: claim-condition-add + claim-conditions-list ----
        class _Args71Add:
            claim = c1
            condition = c3
            kind = "sample"
            explicit = True
            confidence = 0.9

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_claim_condition_add(_Args71Add(), store)
        out = _json.loads(buf.getvalue())
        assert out["ok"] is True, f"T7.1 add: {out}"
        assert out["data"]["kind"] == "sample"
        assert out["data"]["explicit"] is True

        class _Args71List:
            claim_id = c1

        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            cmd_claim_conditions_list(_Args71List(), store)
        out2 = _json.loads(buf2.getvalue())
        assert out2["ok"] is True
        conds = out2["data"]["conditions"]
        assert len(conds) == 1
        assert conds[0]["kind"] == "sample"
        assert conds[0]["explicit"] is True
        print("  Tier 7.1 OK: claim-condition-add writes; "
              "claim-conditions-list returns with kind+explicit")

        # ---- T7.2: Self-reference rejected ----
        class _Args72:
            claim = c1
            condition = c1
            kind = "sample"
            explicit = True
            confidence = 0.9

        buf3 = io.StringIO()
        with contextlib.redirect_stdout(buf3):
            rc = cmd_claim_condition_add(_Args72(), store)
        out3 = _json.loads(buf3.getvalue())
        assert out3["ok"] is False, f"T7.2: self-ref should fail: {out3}"
        assert out3["error"]["code"] == "invalid_condition"
        print("  Tier 7.2 OK: self-reference rejected")

        # ---- T7.3: Invalid kind rejected ----
        class _Args73:
            claim = c1
            condition = c3
            kind = "bogus"
            explicit = True
            confidence = 0.9

        buf4 = io.StringIO()
        with contextlib.redirect_stdout(buf4):
            rc = cmd_claim_condition_add(_Args73(), store)
        out4 = _json.loads(buf4.getvalue())
        assert out4["ok"] is False, f"T7.3: invalid kind should fail: {out4}"
        assert out4["error"]["code"] == "invalid_condition"
        print("  Tier 7.3 OK: invalid kind rejected")

        # ---- T7.4: claim-add --conditions atomically links ----
        class _Args74:
            source_id = src_id
            subject = "test claim"
            predicate = "is"
            object = "tested"
            span = "treatment group showed 50% tumor reduction"
            confidence = 0.8
            conditions = f"{c3}:sample,{c4}:method"

        buf5 = io.StringIO()
        with contextlib.redirect_stdout(buf5):
            rc = cmd_claim_add(_Args74(), store)
        out5 = _json.loads(buf5.getvalue())
        assert out5["ok"] is True, f"T7.4: should succeed: {out5}"
        new_claim_id = out5["data"]["claim_id"]
        # Verify both links exist
        linked = store.get_claim_conditions(new_claim_id)
        linked_ids = {r["condition_claim_id"] for r in linked}
        assert c3 in linked_ids, f"T7.4: c3 not linked: {linked_ids}"
        assert c4 in linked_ids, f"T7.4: c4 not linked: {linked_ids}"
        print("  Tier 7.4 OK: claim-add --conditions atomically "
              "writes claim + both links")

        # ---- T7.5: claim-add --conditions with bad ID rolls back ----
        claims_before = store.conn.execute(
            "SELECT COUNT(*) FROM claims").fetchone()[0]
        conditions_before = store.conn.execute(
            "SELECT COUNT(*) FROM claim_conditions").fetchone()[0]

        class _Args75:
            source_id = src_id
            subject = "rollback test"
            predicate = "is"
            object = "tested"
            span = "control group showed no significant change"
            confidence = 0.8
            conditions = f"{c3}:sample,99:method"

        buf6 = io.StringIO()
        with contextlib.redirect_stdout(buf6):
            rc = cmd_claim_add(_Args75(), store)
        out6 = _json.loads(buf6.getvalue())
        assert out6["ok"] is False, f"T7.5: should fail: {out6}"
        assert out6["error"]["code"] == "invalid_condition"

        claims_after = store.conn.execute(
            "SELECT COUNT(*) FROM claims").fetchone()[0]
        conditions_after = store.conn.execute(
            "SELECT COUNT(*) FROM claim_conditions").fetchone()[0]
        assert claims_after == claims_before, \
            f"T7.5: claim should be rolled back: {claims_before} -> {claims_after}"
        assert conditions_after == conditions_before, \
            f"T7.5: conditions should be rolled back: {conditions_before} -> {conditions_after}"
        print("  Tier 7.5 OK: claim-add --conditions with bad ID "
              "rolls back (no claim row, no condition rows)")

        # ---- T7.6: conditions_overlap Jaccard ----
        # c1 has condition {c3} from T7.1
        # Create c2 conditions: {c3, c4, c5}
        link_condition(store, c2, c3, "sample")
        link_condition(store, c2, c4, "method")
        link_condition(store, c2, c5, "scope")
        # c1 conditions: {c3}, c2 conditions: {c3, c4, c5}
        # Jaccard = 1 / 3 = 0.333...
        # Actually let's set up the exact test from spec:
        # two claims with condition sets {c3,c4,c5} and {c4,c5,X}
        # where X is a new condition. That gives {c3,c4,c5} vs {c4,c5,X}
        # intersection={c4,c5}, union={c3,c4,c5,X} -> 2/4 = 0.5
        # Let me create a claim_a with {c3,c4,c5} and claim_b with {c4,c5,X}
        claim_a = store.add_claim(
            src_id, "overlap test a", "is", "a",
            span_start=0, span_end=4, confidence=0.8,
        )
        claim_b = store.add_claim(
            src_id, "overlap test b", "is", "b",
            span_start=0, span_end=4, confidence=0.8,
        )
        # Create a 4th condition claim for claim_b only
        c6 = store.add_claim(
            src_id, "extra condition", "applies-to",
            "extra scope",
            span_start=0, span_end=4, confidence=0.8,
        )
        # claim_a conditions: {c3, c4, c5}
        link_condition(store, claim_a, c3, "sample")
        link_condition(store, claim_a, c4, "method")
        link_condition(store, claim_a, c5, "scope")
        # claim_b conditions: {c4, c5, c6}
        link_condition(store, claim_b, c4, "method")
        link_condition(store, claim_b, c5, "scope")
        link_condition(store, claim_b, c6, "scope")
        # Jaccard: intersection={c4,c5}, union={c3,c4,c5,c6} -> 2/4 = 0.5
        ov = overlap(store, claim_a, claim_b)
        assert abs(ov - 0.5) < 1e-9, \
            f"T7.6: expected overlap 0.5, got {ov}"
        print("  Tier 7.6 OK: conditions_overlap returns 0.5 "
              "for sets {c3,c4,c5} vs {c4,c5,c6}")

        # ---- T7.7: Ingest with extract_conditions=True ----
        fake = ConditionsFakeLLM()

        sci_content = (
            "This study used BALB/c mice aged 8-12 weeks in a "
            "double-blind randomized controlled trial. "
            "The treatment group showed 50% tumor reduction while the "
            "control group showed no significant change."
        )
        sci_file = Path(d) / "science_ingest.md"
        sci_file.write_text(sci_content)

        result = ingest_file(
            store, fake, sci_file, extract_conditions=True,
        )
        assert result["status"] == "ingested", f"T7.7: {result}"
        assert result["scope_claims_added"] >= 1, \
            f"T7.7: expected scope claims, got {result}"
        scope_ids = result["scope_claim_ids"]
        assert len(scope_ids) >= 1, \
            f"T7.7: expected scope_claim_ids, got {result}"
        # Verify atomic claims were linked to scope claims
        # Get all atomic claims from this source (non-scope ones)
        new_src_id = result["source_id"]
        all_claims = store.conn.execute(
            "SELECT * FROM claims WHERE source_id = ? AND "
            "predicate != 'applies-to'",
            (new_src_id,),
        ).fetchall()
        assert len(all_claims) >= 1, \
            f"T7.7: expected atomic claims, got {len(all_claims)}"
        for ac in all_claims:
            conds = store.get_claim_conditions(ac["id"])
            assert len(conds) >= 1, \
                (f"T7.7: atomic claim {ac['id']} should have "
                 f"conditions, got {len(conds)}")
            # Verify explicit=True
            for cond in conds:
                assert cond["explicit"] == 1, \
                    f"T7.7: condition should be explicit=True"
        print("  Tier 7.7 OK: ingest with extract_conditions=True "
              "creates scope claims + links every atomic claim")

        # ---- T7.8: Ingest with extract_conditions=False ----
        sci_content2 = (
            "Another study used BALB/c mice aged 8-12 weeks in a "
            "double-blind randomized controlled trial. "
            "The treatment group showed 50% tumor reduction while the "
            "control group showed no significant change."
        )
        sci_file2 = Path(d) / "science_noext.md"
        sci_file2.write_text(sci_content2)

        result2 = ingest_file(
            store, fake, sci_file2, extract_conditions=False,
        )
        assert result2["status"] == "ingested", f"T7.8: {result2}"
        # No scope_claims_added key when extract_conditions=False
        assert "scope_claims_added" not in result2, \
            f"T7.8: should not have scope_claims_added: {result2}"
        # No scope claims from this source
        new_src_id2 = result2["source_id"]
        scope_rows = store.conn.execute(
            "SELECT * FROM claims WHERE source_id = ? AND "
            "predicate = 'applies-to'",
            (new_src_id2,),
        ).fetchall()
        assert len(scope_rows) == 0, \
            f"T7.8: expected 0 scope claims, got {len(scope_rows)}"
        # No condition links for atomic claims from this source
        atomic_rows = store.conn.execute(
            "SELECT * FROM claims WHERE source_id = ? AND "
            "predicate != 'applies-to'",
            (new_src_id2,),
        ).fetchall()
        for ar in atomic_rows:
            conds = store.get_claim_conditions(ar["id"])
            assert len(conds) == 0, \
                (f"T7.8: atomic claim {ar['id']} should have "
                 f"0 conditions, got {len(conds)}")
        print("  Tier 7.8 OK: ingest with extract_conditions=False "
              "creates zero scope claims, zero condition links")

        store.close()

    print("  ALL TIER 7 ASSERTIONS PASSED")


def tier8_query_integration_regression():
    """Tier 8: Query Integration — cross-workstream behavior with context
    filtering, dispositions, concept citations, and cache keying."""
    import json as _json

    print()
    print("=" * 60)
    print("TIER 8 QUERY INTEGRATION REGRESSION CHECKS")
    print("=" * 60)

    from aleph.query import query as run_query
    from aleph import authority
    from aleph import concepts
    from aleph import contradictions

    # -- FakeLLM that dispatches on prompt substrings for v2 pipeline --
    class T8FakeLLM:
        def __init__(self, claim_ids=None, concept_ids=None, gap_ids=None,
                     replicate_ids=None, reconcile_ids=None, reconcile_concept=None,
                     ungrounded_concept_id=None):
            self.calls = 0
            self.claim_ids = claim_ids or []
            self.concept_ids = concept_ids or []
            self.gap_ids = gap_ids or ()
            self.replicate_ids = replicate_ids or []
            self.reconcile_ids = reconcile_ids or ()
            self.reconcile_concept = reconcile_concept
            self.ungrounded_concept_id = ungrounded_concept_id

        def complete(self, system, user, max_tokens=2048):
            self.calls += 1
            # SYNTHESIZE_V2: dispatch on "Dispositions handling"
            if "You answer a question" in system and "Dispositions handling" in system:
                parts = []
                # replicate: one sentence citing all replicate ids
                if self.replicate_ids:
                    ids_str = ",".join(str(i) for i in self.replicate_ids)
                    parts.append(
                        f"This finding is well-established [claim:{ids_str}]."
                    )
                # reconcile: both findings + concept
                if self.reconcile_ids and self.reconcile_concept:
                    a, b = self.reconcile_ids
                    parts.append(
                        f"Under young adults, dopamine enhances memory [claim:{a}]."
                    )
                    parts.append(
                        f"Under older adults, dopamine impairs memory [claim:{b}]."
                    )
                    parts.append(
                        f"The effect depends on age [concept:{self.reconcile_concept}]."
                    )
                # gap
                if self.gap_ids:
                    a, b = self.gap_ids
                    parts.append(
                        f"Findings conflict [claim:{a}, claim:{b}]. "
                        f"No scope difference was identified; an unstated premise "
                        f"may explain the divergence. Flagged for expert review."
                    )
                # plain claims
                for cid in self.claim_ids:
                    if cid not in self.replicate_ids and cid not in (self.reconcile_ids or ()):
                        if self.gap_ids and cid in self.gap_ids:
                            continue
                        parts.append(
                            f"The evidence shows this [claim:{cid}]."
                        )
                if not parts:
                    parts.append("No relevant findings.")
                return "\n".join(parts)
            # old synthesis (no "Dispositions handling")
            if "You answer a question" in system:
                ids_str = ",".join(str(i) for i in self.claim_ids)
                return f"Result [claim:{ids_str}]."
            return "UNKNOWN"

        def complete_json(self, system, user, max_tokens=4096):
            self.calls += 1
            # VERIFY_CONCEPT: "for concept citations"
            if "You are a verifier" in system and "for concept citations" in system:
                # If the ungrounded concept is being verified and the sentence
                # contains overreaching language
                if self.ungrounded_concept_id is not None and "overreaching" in user.lower():
                    return {"verdict": "UNGROUNDED", "reason": "sentence overreaches beyond spans"}
                return {"verdict": "GROUNDED", "reason": "concept spans support sentence"}
            # VERIFY (claim verifier): "You are a verifier" (without concept)
            if "You are a verifier" in system:
                return {"verdict": "GROUNDED", "reason": "span supports sentence"}
            # DETECT: "You judge whether two claims"
            if "You judge whether two claims" in system:
                return {
                    "contradicts": True,
                    "kind": "categorical",
                    "candidate_disposition": "reconcile",
                    "reason": "different conditions",
                }
            # extract
            if "You extract atomic claims" in system:
                return []
            return {}

    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t8.db"
        store = Store(db)

        # --- Setup: source with content ---
        content = (
            "Tesla battery packs retain about 90% of original capacity "
            "after 200,000 miles. "
            "Fleet data from California shows 92% retention. "
            "Fleet data from New York shows 88% retention. "
            "Older studies found only 80% retention but are now retracted."
        )
        sid = store.add_source("/test/t8.md", content)
        sid2 = store.add_source("/test/t8b.md", "Duplicate source with same finding about battery retention at 90%.")
        sid3 = store.add_source("/test/t8ny.md", "New York fleet data shows 88% battery retention.")

        c1 = store.add_claim(sid, "tesla battery", "retain",
                             "90% capacity after 200k miles",
                             content.find("retain about 90%"),
                             content.find("retain about 90%") + len("retain about 90% of original capacity"),
                             0.9)
        c2 = store.add_claim(sid, "tesla battery", "retain",
                             "92% retention in california",
                             content.find("92% retention"),
                             content.find("92% retention") + len("92% retention"),
                             0.85)
        c3 = store.add_claim(sid3, "tesla battery", "retain",
                             "88% retention in new york",
                             0, 40, 0.8)
        c4 = store.add_claim(sid, "tesla battery", "retain",
                             "80% retention (older studies)",
                             content.find("80% retention"),
                             content.find("80% retention") + len("80% retention"),
                             0.7)

        # ---- T8.1: Retracted claims filtered ----
        # Set scientific metadata, then retract
        authority.set_metadata(store, sid, "scientific",
                               {"peer_reviewed": True, "venue": "J.Energy"})
        authority.retract_source(store, sid, "data fabrication")

        # c1, c2, c4 are from sid -> retracted
        fake81 = T8FakeLLM(claim_ids=[c3])
        qr = run_query(store, fake81, "battery retention", use_cache=False)
        assert c1 not in qr.claim_ids_used, \
            f"T8.1: retracted claim {c1} should not be in claim_ids_used: {qr.claim_ids_used}"
        assert c2 not in qr.claim_ids_used, \
            f"T8.1: retracted claim {c2} should not be in claim_ids_used: {qr.claim_ids_used}"
        assert c4 not in qr.claim_ids_used, \
            f"T8.1: retracted claim {c4} should not be in claim_ids_used: {qr.claim_ids_used}"
        print("  Tier 8.1 OK: retracted claims filtered from query results")

        # ---- T8.2: context.include_retracted=True -> retracted included ----
        # Unretract so claims become active again, then re-retract cleanly
        authority.unretract_source(store, sid, "correction")
        authority.retract_source(store, sid, "data issues")

        fake82_ids = [c1, c3]
        fake82 = T8FakeLLM(claim_ids=fake82_ids)
        qr2 = run_query(store, fake82, "battery retention",
                         context={"include_retracted": True}, use_cache=False)
        # With include_retracted, the FakeLLM can cite retracted claims
        # The query pipeline should NOT filter them out
        # We just need to verify the pipeline ran to completion without filtering
        assert qr2.answer != "", "T8.2: answer should not be empty"
        assert c1 in qr2.claim_ids_used, \
            f"T8.2: retracted claim {c1} should be retrievable: {qr2.claim_ids_used}"
        print("  Tier 8.2 OK: include_retracted=True allows retracted claims through")

        # ---- T8.3: Jurisdiction prefix match ----
        # Un-retract sid so c1,c2 are active again for jurisdiction test
        authority.unretract_source(store, sid, "testing")
        # Set jurisdiction metadata
        authority.set_metadata(store, sid, "scientific",
                               {"peer_reviewed": True, "jurisdiction": "US-CA-LA"})
        authority.set_metadata(store, sid3, "scientific",
                               {"peer_reviewed": True, "jurisdiction": "US-NY"})

        fake83 = T8FakeLLM(claim_ids=[c1, c2])
        qr3 = run_query(store, fake83, "battery retention",
                         context={"jurisdiction": "US-CA"}, use_cache=False)
        # c1, c2 from sid (US-CA-LA) should match prefix US-CA
        # c3 from sid3 (US-NY) should NOT match
        assert c3 not in qr3.claim_ids_used, \
            f"T8.3: US-NY claim {c3} should not appear under jurisdiction=US-CA: {qr3.claim_ids_used}"
        print("  Tier 8.3 OK: jurisdiction prefix match filters correctly")

        # ---- T8.4: Date filter ----
        # Set expires_at in the past for sid
        import time as _time
        now = _time.time()
        authority.set_metadata(store, sid, "scientific",
                               {"peer_reviewed": True, "expires_at": now - 1000})
        authority.set_metadata(store, sid3, "scientific",
                               {"peer_reviewed": True})

        fake84 = T8FakeLLM(claim_ids=[c3])
        qr4 = run_query(store, fake84, "battery retention",
                         context={"date": now}, use_cache=False)
        assert c1 not in qr4.claim_ids_used, \
            f"T8.4: expired claim {c1} should be dropped: {qr4.claim_ids_used}"
        print("  Tier 8.4 OK: claims with expires_at < context.date are dropped")

        # Restore sid metadata for remaining tests
        authority.set_metadata(store, sid, "scientific",
                               {"peer_reviewed": True})

        # ---- T8.5: Cache keyed by context ----
        fake85 = T8FakeLLM(claim_ids=[c1])
        q_text = "tesla battery retention rate"
        run_query(store, fake85, q_text, context={"jurisdiction": "US-CA"}, use_cache=True)
        run_query(store, fake85, q_text, context={"jurisdiction": "US-NY"}, use_cache=True)
        # Count cache rows
        cache_count = store.conn.execute("SELECT COUNT(*) FROM view_cache").fetchone()[0]
        assert cache_count >= 2, \
            f"T8.5: same query with different contexts should produce 2 cache rows, got {cache_count}"
        print("  Tier 8.5 OK: cache keyed by context produces distinct rows")

        # ---- T8.6: Replicate surfaces as single finding ----
        # Clear cache and set up replicate scenario
        store.clear_cache()
        # Create two claims with same (S,P,O)
        r1 = store.add_claim(sid, "battery pack", "retain", "90% capacity",
                             content.find("90%"), content.find("90%") + 3, 0.85)
        r2_content = "Duplicate source with same finding about battery retention at 90%."
        r2 = store.add_claim(sid2, "battery pack", "retain", "90% capacity",
                             0, 20, 0.8)
        ct_rep = store.add_contradiction_with_kind(r1, r2, "categorical",
                                                    candidate_disposition="replicate")
        assert ct_rep is not None
        store.update_contradiction_disposition(ct_rep, "replicate")

        fake86 = T8FakeLLM(replicate_ids=[r1, r2])
        qr6 = run_query(store, fake86, "battery pack capacity", use_cache=False)
        # Check answer contains [claim:r1,r2] form
        combined_cite = f"[claim:{r1},{r2}]"
        assert combined_cite in qr6.answer, \
            f"T8.6: answer should contain {combined_cite}, got: {qr6.answer[:200]}"
        print("  Tier 8.6 OK: replicate surfaces as single finding with combined citation")

        # ---- T8.7: Reconcile surfaces both + rule + concept ----
        store.clear_cache()
        # Create concept
        concept_id = store.add_concept(
            "tesla battery", "Battery retention depends on age of driver", "synthesis", 0.8,
            [(c1, "premise"), (c2, "corroborating")],
            status="active",
        )
        # Create contradiction disposed as reconcile with rationale concept
        ct_rec = store.add_contradiction_with_kind(c1, c2, "categorical")
        if ct_rec is None:
            # Already exists; find it
            ct_rec = store.conn.execute(
                "SELECT id FROM contradictions WHERE "
                "(claim_a_id = ? AND claim_b_id = ?) OR (claim_a_id = ? AND claim_b_id = ?)",
                (c1, c2, c2, c1),
            ).fetchone()["id"]
        store.update_contradiction_disposition(
            ct_rec, "reconcile",
            rule="age-dependent effect",
            rationale_concept_id=concept_id,
        )

        fake87 = T8FakeLLM(reconcile_ids=(c1, c2), reconcile_concept=concept_id)
        qr7 = run_query(store, fake87, "tesla battery retention", use_cache=False)
        assert f"[concept:{concept_id}]" in qr7.answer, \
            f"T8.7: answer should contain [concept:{concept_id}], got: {qr7.answer[:300]}"
        assert concept_id in qr7.concept_ids_used, \
            f"T8.7: concept_ids_used should contain {concept_id}: {qr7.concept_ids_used}"
        print("  Tier 8.7 OK: reconcile surfaces both claims + concept citation")

        # ---- T8.8: Gap surfaces as open question ----
        store.clear_cache()
        g1 = store.add_claim(sid, "battery gap", "shows", "result A",
                             0, 10, 0.8)
        g2 = store.add_claim(sid, "battery gap", "shows", "result B",
                             11, 20, 0.75)
        ct_gap = store.add_contradiction_with_kind(g1, g2, "categorical")
        assert ct_gap is not None
        store.update_contradiction_disposition(ct_gap, "gap")

        fake88 = T8FakeLLM(gap_ids=(g1, g2))
        qr8 = run_query(store, fake88, "battery gap results", use_cache=False)
        assert "Flagged for expert review" in qr8.answer, \
            f"T8.8: answer should contain 'Flagged for expert review', got: {qr8.answer[:300]}"
        print("  Tier 8.8 OK: gap surfaces as open question with expert review language")

        # ---- T8.9: Concept citation verified (GROUNDED + UNGROUNDED) ----
        store.clear_cache()
        # Create a concept with support claims
        v_concept = store.add_concept(
            "battery gap", "Battery gap depends on testing conditions", "synthesis", 0.8,
            [(g1, "premise"), (g2, "corroborating")],
            status="active",
        )

        class T89FakeLLM:
            """Returns two sentences: one GROUNDED by concept spans, one UNGROUNDED."""
            def __init__(self, concept_id, claim_id):
                self.calls = 0
                self.concept_id = concept_id
                self.claim_id = claim_id

            def complete(self, system, user, max_tokens=2048):
                self.calls += 1
                if "Dispositions handling" in system:
                    return (
                        f"Battery gap depends on testing conditions [concept:{self.concept_id}].\n"
                        f"Overreaching claim about quantum effects [concept:{self.concept_id}]."
                    )
                return "UNKNOWN"

            def complete_json(self, system, user, max_tokens=4096):
                self.calls += 1
                # VERIFY_CONCEPT
                if "for concept citations" in system:
                    if "Overreaching" in user or "overreaching" in user or "quantum" in user.lower():
                        return {"verdict": "UNGROUNDED", "reason": "sentence overreaches beyond spans"}
                    return {"verdict": "GROUNDED", "reason": "concept spans support sentence"}
                # VERIFY (claim)
                if "You are a verifier" in system:
                    return {"verdict": "GROUNDED", "reason": "ok"}
                return {}

        fake89 = T89FakeLLM(v_concept, g1)
        qr9 = run_query(store, fake89, "battery gap results", use_cache=False, verify=True)
        # Should have verifier flags for the UNGROUNDED concept sentence
        assert "Verifier flags" in qr9.answer, \
            f"T8.9: answer should have Verifier flags, got: {qr9.answer}"
        assert "UNGROUNDED" in qr9.answer, \
            f"T8.9: answer should flag UNGROUNDED concept citation: {qr9.answer}"
        print("  Tier 8.9 OK: concept citation verified; UNGROUNDED sentence flagged")

        # ---- T8.10: Invalidate rationale concept -> cached view drops ----
        store.clear_cache()
        # Cache a view citing a concept
        inv_concept = store.add_concept(
            "tesla battery", "Tesla retention is consistent", "summary", 0.85,
            [(c1, "premise")],
            status="active",
        )
        fake810 = T8FakeLLM(claim_ids=[c1], concept_ids=[inv_concept],
                             reconcile_ids=(c1, c2), reconcile_concept=inv_concept)
        # Override complete to cite the concept
        _orig_complete = fake810.complete
        def _custom_complete(system, user, max_tokens=2048):
            if "Dispositions handling" in system:
                return f"Tesla retention is consistent [claim:{c1}] [concept:{inv_concept}]."
            return _orig_complete(system, user, max_tokens)
        fake810.complete = _custom_complete

        # verify=True: unverified views are not cached (Phase 3).
        run_query(store, fake810, "tesla retention consistency", use_cache=True, verify=True)
        assert store.stats()["cached_views"] >= 1, \
            "T8.10 precondition: at least 1 cached view"

        concepts.invalidate_concept(store, inv_concept, "testing cache invalidation")
        assert store.stats()["cached_views"] == 0, \
            f"T8.10: cached views should be 0 after concept invalidation, got {store.stats()['cached_views']}"
        print("  Tier 8.10 OK: invalidating rationale concept drops cached view")

        store.close()

    print("  ALL TIER 8 ASSERTIONS PASSED")


def scenario_end_to_end():
    """End-to-end narrative: ingest scientific paper -> conditions -> detect
    contradictions -> derive concept -> reconcile -> query with concept citations."""
    import json as _json

    print()
    print("=" * 60)
    print("SCENARIO END-TO-END")
    print("=" * 60)

    from aleph.query import query as run_query
    from aleph import authority
    from aleph import concepts
    from aleph import contradictions

    # -- FakeLLM that covers the full pipeline --
    class ScenarioFakeLLM:
        """Dispatches on system prompt substrings for the full pipeline."""
        def __init__(self):
            self.calls = 0
            # Will be set after claims are created
            self.finding_a_id = None
            self.finding_b_id = None
            self.scope_ids = []
            self.concept_id = None

        def complete(self, system, user, max_tokens=2048):
            self.calls += 1
            # SYNTHESIZE_V2
            if "Dispositions handling" in system:
                a = self.finding_a_id
                b = self.finding_b_id
                c = self.concept_id
                return (
                    f"In young adults (ages 18-25), dopamine enhances working memory [claim:{a}].\n"
                    f"In older adults (ages 65-75), dopamine impairs working memory [claim:{b}].\n"
                    f"The effect of dopamine on working memory depends on age [concept:{c}]."
                )
            return "UNKNOWN"

        def complete_json(self, system, user, max_tokens=4096):
            self.calls += 1

            # EXTRACT_SCOPE: "You extract SCOPE claims"
            if "You extract SCOPE claims" in system:
                return [
                    {
                        "subject": "the study",
                        "predicate": "used",
                        "object": "40 young adults (ages 18-25)",
                        "span": "Study used 40 young adults (ages 18-25)",
                        "kind": "sample",
                        "confidence": 0.95,
                    },
                    {
                        "subject": "the study",
                        "predicate": "used",
                        "object": "35 older adults (ages 65-75)",
                        "span": "Study used 35 older adults (ages 65-75)",
                        "kind": "sample",
                        "confidence": 0.9,
                    },
                ]

            # Atomic extraction: "You extract atomic claims"
            if "You extract atomic claims" in system:
                return [
                    {
                        "subject": "dopamine effect on working memory",
                        "predicate": "enhances",
                        "object": "working memory in young adults",
                        "span": "dopamine enhances working memory in young adults",
                        "confidence": 0.9,
                    },
                    {
                        "subject": "dopamine effect on working memory",
                        "predicate": "impairs",
                        "object": "working memory in older adults",
                        "span": "dopamine impairs working memory in older adults",
                        "confidence": 0.85,
                    },
                ]

            # DETECT: "You judge whether two claims"
            if "You judge whether two claims" in system:
                return {
                    "contradicts": True,
                    "kind": "categorical",
                    "candidate_disposition": "reconcile",
                    "reason": "opposite effects under different age conditions",
                }

            # CONCEPT_DERIVE: "You propose concepts"
            if "You propose concepts" in system:
                import re
                ids = [int(m) for m in re.findall(r"\[claim:(\d+)\]", user)]
                return [{
                    "statement": "Dopamine's effect on working memory depends on age",
                    "inference_type": "synthesis",
                    "confidence": 0.85,
                    "supports": [{"claim_id": cid, "role": "premise"} for cid in ids],
                }]

            # CONCEPT_VALIDATE: "You are a validator"
            if "You are a validator" in system:
                return {"verdict": "GROUNDED", "reason": "spans collectively support the concept"}

            # VERIFY_CONCEPT: "for concept citations"
            if "You are a verifier" in system and "for concept citations" in system:
                return {"verdict": "GROUNDED", "reason": "concept spans support sentence"}

            # VERIFY (claim): "You are a verifier"
            if "You are a verifier" in system:
                return {"verdict": "GROUNDED", "reason": "span supports sentence"}

            return {}

    fake = ScenarioFakeLLM()

    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "scenario.db"
        store = Store(db)

        # Step 2: Write mock scientific paper
        paper = (
            "Study used 40 young adults (ages 18-25) and "
            "Study used 35 older adults (ages 65-75) in a double-blind trial. "
            "Results show that dopamine enhances working memory in young adults "
            "but dopamine impairs working memory in older adults."
        )
        paper_path = Path(d) / "dopamine_study.md"
        paper_path.write_text(paper)

        # Step 4: Ingest with extract_conditions
        from aleph.ingest import ingest_file
        result = ingest_file(store, fake, paper_path, extract_conditions=True)
        assert result["status"] == "ingested", f"ingest failed: {result}"
        assert result["claims_added"] >= 2, f"expected >= 2 atomic claims: {result}"
        assert result["scope_claims_added"] >= 1, f"expected >= 1 scope claims: {result}"
        print(f"  Step 4: ingested {result['claims_added']} atomic claims, "
              f"{result['scope_claims_added']} scope claims")

        source_id = result["source_id"]

        # Step 5: Set scientific metadata
        authority.set_metadata(store, source_id, "scientific", {
            "peer_reviewed": True,
            "venue": "J. Neuroscience",
            "published_at": 1700000000,
        })
        meta = store.get_source_metadata(source_id)
        assert meta is not None
        assert meta["domain"] == "scientific"
        print("  Step 5: scientific metadata set")

        # Find the two finding claims (non-scope, non-applies-to)
        finding_claims = store.conn.execute(
            "SELECT * FROM claims WHERE source_id = ? AND status = 'active' "
            "AND predicate != 'applies-to' ORDER BY id",
            (source_id,),
        ).fetchall()
        assert len(finding_claims) >= 2, \
            f"expected >= 2 finding claims, got {len(finding_claims)}"
        finding_a = finding_claims[0]
        finding_b = finding_claims[1]
        fake.finding_a_id = finding_a["id"]
        fake.finding_b_id = finding_b["id"]
        print(f"  Finding claims: A={finding_a['id']} ({finding_a['predicate']}), "
              f"B={finding_b['id']} ({finding_b['predicate']})")

        # Step 7: Detect contradictions
        det_result = contradictions.detect_all(store, fake)
        assert det_result["contradictions_confirmed"] >= 1, \
            f"expected >= 1 contradiction, got {det_result}"
        print(f"  Step 7: detected {det_result['contradictions_confirmed']} contradiction(s)")

        # Step 8: Derive concepts
        new_concept_ids = concepts.derive_concepts(
            store, fake, "dopamine effect on working memory",
        )
        assert len(new_concept_ids) >= 1, \
            f"expected >= 1 concept, got {new_concept_ids}"
        concept_id = new_concept_ids[0]
        concept_row = store.get_concept(concept_id)
        assert concept_row is not None
        assert concept_row["status"] == "active", \
            f"expected active concept, got {concept_row['status']}"
        fake.concept_id = concept_id
        print(f"  Step 8: derived concept {concept_id}: {concept_row['statement']!r}")

        # Step 9: Dispose as reconcile
        # Find the contradiction
        all_cts = store.conn.execute(
            "SELECT * FROM contradictions WHERE status != 'resolved' LIMIT 1"
        ).fetchone()
        assert all_cts is not None, "no open contradiction found"
        ct_id = all_cts["id"]

        contradictions.dispose(
            store, ct_id, "reconcile",
            rule="age-dependent effect",
            rationale_concept_id=concept_id,
        )
        ct_after = store.conn.execute(
            "SELECT * FROM contradictions WHERE id = ?", (ct_id,)
        ).fetchone()
        assert ct_after["disposition"] == "reconcile", \
            f"expected disposition=reconcile, got {ct_after['disposition']}"
        assert ct_after["status"] == "resolved", \
            f"expected status=resolved, got {ct_after['status']}"
        print(f"  Step 9: contradiction {ct_id} disposed as reconcile")

        # Step 10: Query
        qr = run_query(
            store, fake,
            "What does dopamine do to working memory?",
            use_cache=False,
        )
        print(f"  Step 10: query returned answer ({len(qr.answer)} chars)")
        print(f"    claim_ids_used: {qr.claim_ids_used}")
        print(f"    concept_ids_used: {qr.concept_ids_used}")

        # Step 11: Assertions
        # Answer contains both claim citations
        assert f"[claim:{finding_a['id']}]" in qr.answer, \
            f"answer should cite claim A: {qr.answer[:300]}"
        assert f"[claim:{finding_b['id']}]" in qr.answer, \
            f"answer should cite claim B: {qr.answer[:300]}"
        # Answer contains concept citation
        assert f"[concept:{concept_id}]" in qr.answer, \
            f"answer should cite concept: {qr.answer[:300]}"
        # concept_ids_used contains the concept
        assert concept_id in qr.concept_ids_used, \
            f"concept_ids_used should contain {concept_id}: {qr.concept_ids_used}"
        # All citations are GROUNDED
        for c in qr.citations:
            assert c.verdict == "GROUNDED", \
                f"citation should be GROUNDED, got {c.verdict}: {c.sentence[:80]}"
        # Answer contains condition-aware phrasing (young vs older adults)
        answer_lower = qr.answer.lower()
        assert "young" in answer_lower or "18-25" in answer_lower, \
            f"answer should mention young adults condition: {qr.answer[:300]}"
        assert "older" in answer_lower or "65-75" in answer_lower, \
            f"answer should mention older adults condition: {qr.answer[:300]}"

        store.close()

    print()
    print("SCENARIO END-TO-END PASSED")


if __name__ == "__main__":
    main()
    tier1_regression()
    tier2_regression()
    tier23_regression()
    tier3_regression()
    phase0_schema_regression()
    tier4_concepts_regression()
    tier5_contradictions_regression()
    tier6_authority_regression()
    tier7_conditions_regression()
    tier8_query_integration_regression()
    scenario_end_to_end()
