"""Phase 3: `lint --resolve-by-recency` orders by source date, not by when
a claim was extracted; legal pairs are ranked by authority first
(lex superior, then lex specialis, then lex posterior)."""
from __future__ import annotations

import time

import pytest

from aleph import authority
from aleph.db import Store
from aleph.lint import resolve_by_recency, resolve_by_source_date

DAY = 86400.0


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def _claim(store, text, domain=None, **meta):
    sid = store.add_source(f"{text}.txt", text)
    if domain:
        authority.set_metadata(store, sid, domain, meta)
    return store.add_claim(sid, "battery", "lasts", text, 0, len(text), 0.8)


def _pair(store, a, b):
    return store.add_contradiction(a, b)


def test_newer_source_date_wins_even_if_extracted_earlier(store):
    new = _claim(store, "10 years", "scientific", published_at=time.time())
    old = _claim(store, "8 years", "scientific", published_at=time.time() - 400 * DAY)
    # `old` was extracted last; extraction order must not matter.
    assert store.get_claim(old)["extracted_at"] > store.get_claim(new)["extracted_at"]
    _pair(store, new, old)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 1
    assert store.get_claim(old)["status"] == "superseded"
    assert store.get_claim(new)["status"] == "active"
    assert r["decisions"][0]["rule"] == "source_date"


def test_undated_pairs_are_left_open(store):
    a = _claim(store, "10 years")
    b = _claim(store, "8 years", "scientific", published_at=time.time())
    _pair(store, a, b)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 0 and r["skipped"][0]["reason"] == "undated"
    assert store.get_claim(a)["status"] == store.get_claim(b)["status"] == "active"


def test_same_date_is_a_tie_and_left_open(store):
    t = time.time()
    a = _claim(store, "10 years", "policy", effective_at=t)
    b = _claim(store, "8 years", "policy", effective_at=t)
    _pair(store, a, b)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 0 and r["skipped"][0]["reason"] == "tie"


@pytest.mark.parametrize("domain,field", [
    ("scientific", "published_at"), ("policy", "effective_at"),
    ("corporate", "reviewed_at"), ("legal", "issued_at"),
])
def test_source_date_per_domain(domain, field):
    assert authority.source_date({"domain": domain, "metadata": {field: 5.0}}) == 5.0


def test_legal_authority_outranks_recency(store):
    # A newer regulation doesn't override an older statute (lex superior).
    statute = _claim(store, "statute says 10 years", "legal",
                     authority_level=3, specificity=1, issued_at=time.time() - 900 * DAY)
    regulation = _claim(store, "regulation says 8 years", "legal",
                        authority_level=2, specificity=1, issued_at=time.time())
    _pair(store, statute, regulation)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 1 and r["decisions"][0]["rule"] == "lex_superior"
    assert store.get_claim(regulation)["status"] == "superseded"


def test_lex_specialis_is_left_open_for_distinguish(store):
    # A special rule displaces the general one only inside its own scope:
    # that is `distinguish`, not `supersede`, so auto-resolution leaves it.
    general = _claim(store, "general rule 10 years", "legal",
                     authority_level=2, specificity=1, issued_at=time.time())
    special = _claim(store, "special rule 8 years", "legal",
                     authority_level=2, specificity=2, issued_at=time.time() - 900 * DAY)
    _pair(store, general, special)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 0 and r["skipped"][0]["reason"] == "lex_specialis"
    assert store.get_claim(general)["status"] == "active"


def test_legal_same_level_and_specificity_newer_wins(store):
    old = _claim(store, "old rule 10 years", "legal",
                 authority_level=2, specificity=1, issued_at=time.time() - 900 * DAY)
    new = _claim(store, "new rule 8 years", "legal",
                 authority_level=2, specificity=1, issued_at=time.time())
    _pair(store, old, new)
    r = resolve_by_source_date(store)
    assert r["decisions"][0]["rule"] == "lex_posterior"
    assert store.get_claim(old)["status"] == "superseded"


@pytest.mark.parametrize("meta_a,meta_b,reason", [
    # missing authority level: unknown is not "lowest"
    ({"specificity": 1, "issued_at": 1.0}, {"authority_level": 1, "issued_at": 2.0}, "unranked"),
    # same level and specificity, one undated
    ({"authority_level": 3, "specificity": 1}, {"authority_level": 3, "specificity": 1,
                                                "issued_at": 2.0}, "undated"),
    # different jurisdictions: levels aren't comparable
    ({"authority_level": 3, "jurisdiction": "IT", "issued_at": 1.0},
     {"authority_level": 2, "jurisdiction": "US", "issued_at": 2.0}, "cross_jurisdiction"),
])
def test_legal_pairs_are_never_decided_by_missing_fields(store, meta_a, meta_b, reason):
    a = _claim(store, "rule a", "legal", **meta_a)
    b = _claim(store, "rule b", "legal", **meta_b)
    _pair(store, a, b)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 0 and r["skipped"][0]["reason"] == reason


def test_legal_effective_at_serves_as_the_date(store):
    a = _claim(store, "rule a", "legal", authority_level=2, specificity=1, effective_at=1.0)
    b = _claim(store, "rule b", "legal", authority_level=2, specificity=1, effective_at=2.0)
    _pair(store, a, b)
    resolve_by_source_date(store)
    assert store.get_claim(a)["status"] == "superseded"


def test_nested_jurisdiction_levels_compare(store):
    a = _claim(store, "state rule", "legal", authority_level=2, jurisdiction="US-CA", issued_at=1.0)
    b = _claim(store, "federal rule", "legal", authority_level=3, jurisdiction="US", issued_at=1.0)
    _pair(store, a, b)
    assert resolve_by_source_date(store)["resolved"] == 1
    assert store.get_claim(a)["status"] == "superseded"


def test_nested_jurisdiction_same_level_is_not_lex_posterior(store):
    # A newer state statute doesn't supersede an older federal one of the
    # same level: that's preemption/competence, not lex posterior.
    fed = _claim(store, "federal rule", "legal", authority_level=3, jurisdiction="US", issued_at=1.0)
    state = _claim(store, "state rule", "legal", authority_level=3, jurisdiction="US-CA", issued_at=2.0)
    _pair(store, fed, state)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 0 and r["skipped"][0]["reason"] == "nested_jurisdiction"
    assert store.get_claim(fed)["status"] == "active"


def test_one_missing_jurisdiction_is_unknown(store):
    it = _claim(store, "IT rule", "legal", authority_level=3, jurisdiction="IT", issued_at=1.0)
    other = _claim(store, "unknown rule", "legal", authority_level=4, issued_at=1.0)
    _pair(store, it, other)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 0 and r["skipped"][0]["reason"] == "unknown_jurisdiction"


def test_specificity_missing_on_one_side_is_unranked(store):
    a = _claim(store, "rule a", "legal", authority_level=2, specificity=0, issued_at=1.0)
    b = _claim(store, "rule b", "legal", authority_level=2, issued_at=2.0)
    _pair(store, a, b)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 0 and r["skipped"][0]["reason"] == "unranked"


def test_date_decided_legal_pair_is_labelled_lex_posterior(store):
    a = _claim(store, "old", "legal", authority_level=2, specificity=1, issued_at=1.0)
    b = _claim(store, "new", "legal", authority_level=2, specificity=1, issued_at=2.0)
    _pair(store, a, b)
    assert resolve_by_source_date(store)["decisions"][0]["rule"] == "lex_posterior"


def test_chains_resolve_to_the_newest_claim(store):
    # A newer than B newer than C; pairs (B,C) and (A,B). Order of the open
    # list must not strand C as the survivor of B-C.
    c = _claim(store, "c 6 years", "scientific", published_at=1.0)
    b = _claim(store, "b 8 years", "scientific", published_at=2.0)
    a = _claim(store, "a 10 years", "scientific", published_at=3.0)
    _pair(store, b, c)
    _pair(store, a, b)
    resolve_by_source_date(store)
    assert [store.get_claim(x)["status"] for x in (a, b, c)] == ["active", "superseded", "superseded"]
    # (B,C) reopens when its keeper B is superseded (the cascade's rule);
    # both of its members are inactive, so nothing live rests on it.
    assert store.check_invariants() == []


def test_mixed_domain_pair_is_left_open(store):
    law = _claim(store, "law 10 years", "legal", authority_level=3, issued_at=time.time() - 900 * DAY)
    paper = _claim(store, "paper 8 years", "scientific", published_at=time.time())
    _pair(store, law, paper)
    r = resolve_by_source_date(store)
    assert r["resolved"] == 0 and r["skipped"][0]["reason"] == "mixed_domain"


def test_cross_subject_pairs_are_never_auto_resolved(store):
    general = _claim(store, "self-defence excuses", "scientific", published_at=1.0)
    limit = _claim(store, "excess limits the defence", "scientific", published_at=2.0)
    x = store.add_contradiction_cross_subject(general, limit, relation_kind="rule-limits-rule",
                                              justification="art. 55 limits art. 52")
    assert x is not None
    r = resolve_by_source_date(store)
    assert r["resolved"] == 0 and r["skipped"][0]["reason"] == "cross_subject"
    assert store.get_claim(general)["status"] == "active"


def test_resolve_by_recency_returns_the_resolved_count(store):
    a = _claim(store, "10 years", "scientific", published_at=time.time())
    b = _claim(store, "8 years", "scientific", published_at=time.time() - DAY)
    _pair(store, a, b)
    assert resolve_by_recency(store) == 1
    assert store.check_invariants() == []


def test_lint_cli_prints_skip_reasons(tmp_path, capsys, monkeypatch):
    import json as _json
    from aleph.cli import main
    db = tmp_path / "cli.db"
    s = Store(db)
    a = _claim(s, "10 years", "scientific", published_at=2.0)
    b = _claim(s, "8 years", "scientific", published_at=1.0)
    c = _claim(s, "undated 5 years")
    _pair(s, a, b)
    _pair(s, a, c)
    s.close()
    fx = tmp_path / "fx.json"
    fx.write_text(_json.dumps([{"match": {"default": True}, "response": {"contradicts": False}}]))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--db", str(db), "--mock-llm", str(fx), "lint", "--resolve-by-recency"]) == 0
    out = capsys.readouterr().out
    assert "resolved 1 contradictions by source date" in out
    assert "left open: 1 undated" in out
