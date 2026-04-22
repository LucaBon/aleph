"""End-to-end test of the agent-mode CLI — no LLM anywhere.

Simulates what Claude Code would do: source-add, claim-add (many), subjects,
alias-add, contradiction-add/resolve, view-cache, stats. Exercises the
subject-normalization + alias fixes and the Tier 4 agent-surface hygiene:
unified envelope, contradiction/alias validation, source-replace.
"""
import json
import subprocess
import tempfile
from pathlib import Path


def _exec(db, *args):
    cmd = ["aleph", "--db", str(db), *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    parsed = json.loads(result.stdout) if result.stdout.strip() else None
    return result.returncode, parsed, result.stderr


def run(db, *args):
    """Run `aleph ...` expecting success. Returns the unwrapped `data` dict."""
    rc, parsed, stderr = _exec(db, *args)
    if parsed is None:
        if rc != 0:
            raise RuntimeError(f"command failed: {' '.join(args)}\n{stderr}")
        return None
    assert "ok" in parsed, f"malformed envelope: {parsed}"
    if not parsed["ok"]:
        raise RuntimeError(
            f"command returned error envelope: {parsed['error']} (args: {args})"
        )
    return parsed["data"]


def run_err(db, *args):
    """Run `aleph ...` expecting an error envelope. Returns the error dict."""
    rc, parsed, stderr = _exec(db, *args)
    assert parsed is not None, f"no stdout (rc={rc}, stderr={stderr})"
    assert parsed.get("ok") is False, f"expected error envelope, got: {parsed}"
    return parsed["error"]


def main():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "agent.db"
        src = Path(d) / "tesla.md"
        src.write_text(
            "Tesla batteries retain about 90% of original capacity after 200,000 miles.\n"
            "Tesla batteries last 8 years on average in early reports.\n"
            "Most packs last well beyond 10 years under typical use.\n"
            "NCA cells have energy density of 260 Wh/kg.\n"
        )

        print("1. source-add")
        r = run(db, "source-add", str(src))
        assert r["status"] == "ingested"
        sid = r["source_id"]
        print(f"   source_id={sid}")

        print("2. claim-add with varied subject forms (tests normalization)")
        claim_specs = [
            ("Tesla batteries", "retain", "about 90% of original capacity",
             "Tesla batteries retain about 90% of original capacity", 0.9),
            ("tesla battery", "lasts on average", "8 years",
             "Tesla batteries last 8 years on average", 0.7),
            ("Tesla Battery", "lasts", "well beyond 10 years",
             "Most packs last well beyond 10 years under typical use", 0.9),
            ("NCA cells", "have energy density of", "260 Wh/kg",
             "NCA cells have energy density of 260 Wh/kg", 0.95),
        ]
        added = []
        for subj, pred, obj, span, conf in claim_specs:
            r = run(db, "claim-add", "--source-id", str(sid),
                    "--subject", subj, "--predicate", pred, "--object", obj,
                    "--span", span, "--confidence", str(conf))
            assert "claim_id" in r, r
            added.append(r)
            print(f"   #{r['claim_id']}: subject={r['subject']!r} (was {subj!r})")

        print("3. subjects (all three Tesla variants should collapse to 'tesla battery')")
        r = run(db, "subjects")
        subs = {s["subject"]: s["count"] for s in r["subjects"]}
        print(f"   {subs}")
        assert subs.get("tesla battery") == 3, f"normalization failed: {subs}"
        assert "nca cell" in subs
        assert "ncas cell" not in subs  # singular check

        print("4. claim-by-subject 'Tesla Batteries' (alias-resolution path)")
        r = run(db, "claim-by-subject", "Tesla Batteries")
        assert r["subject_canonical"] == "tesla battery"
        assert len(r["claims"]) == 3

        print("5. alias-add 'tesla batteries' -> 'tesla battery pack'")
        r = run(db, "alias-add", "tesla batteries", "tesla battery pack")
        print(f"   rewritten: {r['claims_rewritten']} claims (all three tesla ones)")
        assert r["claims_rewritten"] == 3

        print("6. subjects again")
        r = run(db, "subjects")
        subs = {s["subject"]: s["count"] for s in r["subjects"]}
        print(f"   {subs}")
        assert subs.get("tesla battery pack") == 3, subs

        print("7. contradiction-add + resolve")
        # claim 2 (8 years) vs claim 3 (10 years), same resolved subject, differing objects
        c = run(db, "contradiction-add",
                str(added[1]["claim_id"]), str(added[2]["claim_id"]))
        cid = c["contradiction_id"]
        assert cid is not None
        r = run(db, "contradiction-list")
        assert len(r["contradictions"]) == 1
        r = run(db, "contradiction-resolve", str(cid),
                "--keep", str(added[2]["claim_id"]),
                "--drop", str(added[1]["claim_id"]))
        assert r["resolved"] == cid

        print("8. view-cache + view-get")
        run(db, "view-cache", "how long do tesla batteries last?",
            "Tesla packs last well beyond 10 years [claim:3].",
            "--claim-ids", str(added[2]["claim_id"]))
        r = run(db, "view-get", "how long do tesla batteries last?")
        assert r["cached"] is True
        assert "claim:3" in r["response"] or f"claim:{added[2]['claim_id']}" in r["response"]

        print("9. stats")
        r = run(db, "stats-json")
        print(f"   {r}")
        assert r["claims_active"] == 3  # one got superseded
        assert r["claims_superseded"] == 1
        assert r["cached_views"] == 1

        print("10. source-remove (cascade)")
        r = run(db, "source-remove", str(sid))
        assert r["removed"] == 1
        r = run(db, "stats-json")
        print(f"   stats after remove: {r}")
        assert r["sources"] == 0
        assert r["claims_active"] == 0
        assert r["cached_views"] == 0  # cache invalidated

        print()
        print("=" * 60)
        print("AGENT-MODE E2E PASSED (no API key, pure CLI)")
        print("=" * 60)


def tier4_regression():
    """Tier 4: envelope shape, contradiction validation, alias cycles, source-replace."""
    print()
    print("=" * 60)
    print("TIER 4 REGRESSION CHECKS")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t4.db"
        src = Path(d) / "src.md"
        src.write_text(
            "Alpha beta gamma.\n"
            "The red fox jumps over the lazy dog.\n"
            "Widgets have weight 10kg.\n"
            "Widgets have weight 20kg.\n"
        )

        # --- 4.2 envelope shape: success + error both wrap in {"ok": ...} ---
        rc, parsed, _ = _exec(db, "source-add", str(src))
        assert parsed["ok"] is True, parsed
        assert "data" in parsed and "source_id" in parsed["data"], parsed
        sid = parsed["data"]["source_id"]

        rc, parsed, _ = _exec(db, "source-get", "99999")
        assert parsed["ok"] is False, parsed
        assert parsed["error"]["code"] == "source_not_found", parsed
        print("  Tier 4.2 OK: unified envelope (ok/data/error) with error codes")

        # Seed a couple of claims for later steps
        c1 = run(db, "claim-add", "--source-id", str(sid),
                 "--subject", "widget", "--predicate", "weighs",
                 "--object", "10kg", "--span", "Widgets have weight 10kg",
                 "--confidence", "0.9")
        c2 = run(db, "claim-add", "--source-id", str(sid),
                 "--subject", "widget", "--predicate", "weighs",
                 "--object", "20kg", "--span", "Widgets have weight 20kg",
                 "--confidence", "0.9")
        c3 = run(db, "claim-add", "--source-id", str(sid),
                 "--subject", "fox", "--predicate", "is",
                 "--object", "red", "--span", "The red fox jumps over the lazy dog",
                 "--confidence", "0.9")

        # --- 4.3: contradiction-add rejects different subjects ---
        err = run_err(db, "contradiction-add",
                      str(c1["claim_id"]), str(c3["claim_id"]))
        assert err["code"] == "contradiction_invalid", err
        assert "different subjects" in err["message"], err
        print("  Tier 4.3 OK: contradiction-add rejects different-subject pair")

        # --- 4.3: contradiction-add rejects identical objects ---
        c_dup = run(db, "claim-add", "--source-id", str(sid),
                    "--subject", "widget", "--predicate", "weighs",
                    "--object", "10kg", "--span", "Widgets have weight 10kg",
                    "--confidence", "0.9")
        err = run_err(db, "contradiction-add",
                      str(c1["claim_id"]), str(c_dup["claim_id"]))
        assert err["code"] == "contradiction_invalid", err
        assert "identical objects" in err["message"], err
        print("  Tier 4.3 OK: contradiction-add rejects identical objects")

        # --- 4.3: contradiction-add rejects unknown claim id ---
        err = run_err(db, "contradiction-add", str(c1["claim_id"]), "99999")
        assert err["code"] == "claim_not_found", err
        print("  Tier 4.3 OK: contradiction-add rejects unknown claim id")

        # --- 4.3: legitimate contradiction (same subject, different objects) ---
        ok_c = run(db, "contradiction-add",
                   str(c1["claim_id"]), str(c2["claim_id"]))
        assert ok_c["created"] is True
        cid = ok_c["contradiction_id"]

        # --- 4.3: contradiction-resolve rejects --keep not in the pair ---
        err = run_err(db, "contradiction-resolve", str(cid),
                      "--keep", str(c3["claim_id"]))
        assert err["code"] == "contradiction_invalid", err
        assert "--keep" in err["message"], err
        print("  Tier 4.3 OK: contradiction-resolve rejects --keep not in pair")

        # --- 4.4: alias cycle detection ---
        # chain: alpha -> beta -> gamma, then try gamma -> alpha (would cycle)
        run(db, "alias-add", "alpha", "beta")
        run(db, "alias-add", "beta", "gamma")
        err = run_err(db, "alias-add", "gamma", "alpha")
        assert err["code"] == "alias_would_create_cycle", err
        print("  Tier 4.4 OK: alias cycle rejected before insert")
        # non-cycle still works
        ok = run(db, "alias-add", "delta", "epsilon")
        assert "from_canonical" in ok, ok

        # --- 4.6: source-replace swaps content atomically ---
        first_sid = sid
        src.write_text("Widgets have weight 30kg.\n")
        r = run(db, "source-replace", str(src))
        assert r["status"] == "replaced", r
        assert first_sid in r["old_source_ids"], r
        assert r["old_claims_removed"] >= 3, r
        assert r["source_id"] != first_sid, r
        # old source gone; old claims gone
        stats = run(db, "stats-json")
        assert stats["sources"] == 1, stats
        assert stats["claims_active"] == 0, stats
        print("  Tier 4.6 OK: source-replace swaps source atomically (old claims cascaded)")

        # source-replace on a missing file still errors cleanly
        err = run_err(db, "source-replace", "/nonexistent/path/file.md")
        assert err["code"] == "not_a_file", err
        print("  Tier 4.6 OK: source-replace on missing path returns error envelope")

    print("  ALL TIER 4 ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
    tier4_regression()
