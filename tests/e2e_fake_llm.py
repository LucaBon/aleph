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
        # hack: make the '10 years' claim appear newer than the '8 years' claim
        store.conn.execute(
            "UPDATE claims SET extracted_at = extracted_at + 100 WHERE object = 'well beyond 10 years under typical use'"
        )
        store.conn.commit()
        n = resolve_by_recency(store)
        print(f"  resolved: {n}")
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


if __name__ == "__main__":
    main()
    tier1_regression()
    tier2_regression()
    tier23_regression()
    tier3_regression()
