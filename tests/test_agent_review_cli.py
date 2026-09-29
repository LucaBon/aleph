"""Phase 2 agent-mode surface: propositions and fidelity on claim-add,
claim-fidelity-check, and the review-* commands. Every call emits one JSON
envelope."""
from __future__ import annotations

import json

import pytest

from aleph.cli import main

TEXT = (
    "Tesla was founded in 2003. Model S packs retain about 90% of their "
    "original capacity after 200,000 miles. Degradation is not linear."
)
SPAN = "Model S packs retain about 90% of their original capacity after 200,000 miles."


@pytest.fixture
def aleph(tmp_path, capsys):
    db = str(tmp_path / "a.db")

    def run(*args):
        capsys.readouterr()
        rc = main(["--db", db, *map(str, args)])
        env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert (rc == 0) == env["ok"], env
        return env

    src = tmp_path / "tesla.txt"
    src.write_text(TEXT, encoding="utf-8")
    env = run("source-add", src)
    run.source_id = env["data"]["source_id"]
    return run


def _claim(aleph, obj, **kw):
    args = ["claim-add", "--source-id", aleph.source_id, "--subject", "Model S pack",
            "--predicate", "retains", "--object", obj, "--span", SPAN]
    for k, v in kw.items():
        args += [f"--{k}", v]
    return aleph(*args)


def test_claim_add_reports_no_issues_for_a_faithful_claim(aleph):
    d = _claim(aleph, "about 90% capacity after 200,000 miles")["data"]
    assert d["fidelity_issues"] == [] and d["review_id"] is None


def test_claim_add_reports_issues_and_queues_the_claim(aleph):
    d = _claim(aleph, "about 80% capacity after 200,000 miles")["data"]
    assert [i["kind"] for i in d["fidelity_issues"]] == ["number"]
    assert d["review_id"] is not None
    [item] = aleph("review-list")["data"]["items"]
    assert item["review_id"] == d["review_id"]
    assert item["target"] == {"claim_id": d["claim_id"]}
    assert item["details"]["issues"][0]["value"] == "80"


def test_claim_get_shows_proposition_and_context(aleph):
    d = _claim(aleph, "about 90% capacity",
               proposition="Model S packs keep about 90% of their capacity.")["data"]
    got = aleph("claim-get", d["claim_id"])["data"]
    assert got["proposition"] == "Model S packs keep about 90% of their capacity."
    assert got["context_text"] == TEXT
    assert got["span_text"] == SPAN


def test_claim_fidelity_check_single_and_all(aleph):
    ok = _claim(aleph, "about 90% capacity")["data"]["claim_id"]
    bad = _claim(aleph, "about 80% capacity")["data"]["claim_id"]
    one = aleph("claim-fidelity-check", ok)["data"]
    assert one["checked"] == 1 and one["flagged"] == []
    everything = aleph("claim-fidelity-check", "--all")["data"]
    assert everything["checked"] == 2
    assert [f["claim_id"] for f in everything["flagged"]] == [bad]


def test_claim_fidelity_check_enqueue_is_idempotent(aleph):
    _claim(aleph, "about 80% capacity")
    aleph("claim-fidelity-check", "--all", "--enqueue")
    assert len(aleph("review-list")["data"]["items"]) == 1


def test_review_add_and_resolve(aleph):
    cid = _claim(aleph, "about 90% capacity")["data"]["claim_id"]
    rid = aleph("review-add", "--type", "claim", "--id", cid,
                "--reason", "manual", "--note", "double-check")["data"]["review_id"]
    err = aleph("review-resolve", rid, "--decision", "maybe", "--by", "luca")
    assert err["error"]["code"] == "invalid_decision"
    d = aleph("review-resolve", rid, "--decision", "rejected", "--by", "luca",
              "--note", "object is wrong")["data"]
    assert d["status"] == "rejected"
    assert "claim-supersede" in d["next_step"]
    assert aleph("review-list")["data"]["items"] == []
    [closed] = aleph("review-list", "--status", "rejected")["data"]["items"]
    assert closed["resolved_by"] == "luca"
    again = aleph("review-resolve", rid, "--decision", "accepted", "--by", "luca")
    assert again["error"]["code"] == "review_not_open"


def test_review_add_alias_and_errors(aleph):
    _claim(aleph, "about 90% capacity")
    aleph("alias-add", "model s battery", "model s pack")
    d = aleph("review-add", "--type", "alias", "--alias", "model s battery",
              "--reason", "manual")["data"]
    [item] = aleph("review-list", "--type", "alias")["data"]["items"]
    assert item["target"] == {"alias_from": "model s battery"}
    assert item["review_id"] == d["review_id"]
    assert aleph("review-add", "--type", "claim", "--id", 999,
                 "--reason", "manual")["error"]["code"] == "review_target_not_found"
    assert aleph("review-add", "--type", "alias", "--id", 1,
                 "--reason", "manual")["error"]["code"] == "invalid_review_target"
    assert aleph("review-resolve", 999, "--decision", "accepted",
                 "--by", "x")["error"]["code"] == "review_not_found"


def test_review_add_alias_rejects_a_stray_id(aleph):
    _claim(aleph, "about 90% capacity")
    aleph("alias-add", "model s battery", "model s pack")
    err = aleph("review-add", "--type", "alias", "--alias", "model s battery", "--id", 3)
    assert err["error"]["code"] == "invalid_review_target"
