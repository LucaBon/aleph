# Roadmap

Aleph's promise: **every answer can be checked quickly and fixed cleanly.**

That promise has two halves with different kinds of evidence:

- **"Fixed cleanly"** is a property of the code. It can be tested exhaustively (invariant tests, canary injection) and can come close to a guarantee.
- **"Checked quickly"** is a property of people using the tool. It can only be measured with user studies, never guaranteed.

This roadmap came out of an external review (September 2026). Phases are ordered so that cheap, deterministic, credibility-building work lands first.

## Claims we can make today, and claims we can't yet

| Claim | Status | Evidence |
|---|---|---|
| Every stored claim's span is a verbatim substring of its source | **Yes** | Enforced at write time (`claim-add`, `ingest._locate_span`) |
| Removing a source invalidates cached views that cited it | **Yes** | e2e tests; smoke benchmark |
| No view, concept or disposition survives a retraction (transitively) | **Yes** | `Store.check_invariants()` + `invariant-check` exist; every deactivation path (remove, supersede, retract, `retracted` disposition) runs one cascade that reopens dependent contradictions. `tests/test_invariants.py` (hypothesis, `pip install -e .[dev]`) passes 1000 examples × 60 steps, plus targeted scenarios. Mutation-checked: disabling revival, reopening or view invalidation makes it fail. `benchmark/canary.py` reports 0 leaks, 0 invariant violations and 0% over-invalidation. Both run in CI on every push (Python 3.10/3.12/3.14) |
| Alias merges are reversible | **Yes** | `alias_events` merge log, `alias-undo`, and `alias-add` refuses to overwrite without `--force` and rejects cycles along the whole chain. Covered by property and scenario tests, run in CI on every push |
| A claim faithfully represents its span (numbers, negation, scope) | **Partly** | Numbers, dates, units, negation and entities are checked deterministically against the span and its context window on every claim write, and flagged claims go to the review queue (`aleph.fidelity`, `review-list`). The checker flags every seeded mismatch in `tests/fixtures/fidelity_seeded.json` in CI. Scope and paraphrase errors are not checked, and flags don't block writes |
| The verifier's false-accept / false-reject rates are known | **No** | The eval harness and a layered verifier exist (Phase 2), but the pair set has only draft labels, and neither verifier has been run on human labels |
| Answers surface contradicting evidence they didn't use | **Partly** | `ask` pulls in claims that conflict with, or are conditions of, retrieved claims. It shows unresolved conflicts to the synthesizer, and it reports uncited counter-evidence and evidence considered but not cited on every read, including cache hits. The synthesizer can still ignore what it's shown; nothing measures how often it does |
| Reviewers check Aleph answers faster *and* more accurately than RAG answers | **Unmeasured** | Phase 6: seeded-error user study |

"Implemented, not yet verified" becomes **Yes** only when the phase's exit criteria below are met and running in CI. Status last checked against the code on 2026-09-28.

The "83%" figure in earlier READMEs came from a fake-LLM run on six sentences. It is a smoke test of the pipeline, not an accuracy result, and is no longer used as a headline.

## Where the review was already addressed

- Retrieval is not keyword-only: `FTSRetriever` and `EmbeddingRetriever` exist. What's missing is hybrid fusion (Phase 3).
- Recency auto-resolution is opt-in (`lint --resolve-by-recency`). It used to sort on claim extraction time (`claims.extracted_at`) rather than source date. Phase 3 fixed that.
- Contradiction detection already blocks by subject. The problem is that the pre-filter sends almost every same-subject pair to the LLM (Phase 3).

## Phases

### Phase 0: Honest framing
- README reframed around auditability and reversibility; benchmark labelled as a smoke test.
- **Test prerequisites** (Phase 1 needs these):
  - a `dev` extra in `pyproject.toml` with `pytest` and `hypothesis` (done);
  - remove the hardcoded `/home/claude/aleph/...` paths from `tests/e2e_fake_llm.py` (done: none remain);
  - one command that runs both e2e scripts plus the pytest suite (done: `pytest` runs the property suite, both e2e scripts and the canary benchmark via `tests/test_scripts.py`);
  - CI that runs that command (done: `.github/workflows/ci.yml` runs `pytest` under the `thorough` profile on Python 3.10/3.12/3.14 on every push and pull request).
- **Schema versioning** (done): a read-only `schema_version` config key and the numbered `MIGRATIONS` list in `db.py`. SCHEMA + `_run_alters` define version 1; an unversioned store is stamped 1, and a store from a newer aleph is refused (`schema_too_new`). `_run_alters` is still fine for adding columns, but Phase 2 (`proposition`) and Phase 4 (`features`) change what existing rows mean and go through `MIGRATIONS`. Phase 2's is migration 2.

**Exit criteria:** a fresh clone passes the full test command with no path edits, and CI runs it on every push.

**Status:** met. CI installs from a fresh checkout and passes on every push (first green run 2026-09-28, [LucaBon/aleph](https://github.com/LucaBon/aleph)).

### Phase 1: "Fixed cleanly" as a tested guarantee
- `Store.check_invariants()` and the `invariant-check` agent command. The invariant: no active view, active/attested concept or resolved contradiction depends on a retracted, superseded or removed claim.
- Property-based tests (hypothesis) over random operation sequences.
- Reopen dispositioned contradictions when a claim they rest on is retracted, superseded or removed.
- Alias merge log, `alias-undo`, and no silent alias overwrite.
- Canary leakage benchmark (`benchmark/canary.py`), including the over-invalidation rate.

**Status:** exit criteria met (2026-09-28). The thorough property profile and the canary benchmark (0 leaks, 0 invariant violations) pass in CI on every push. `invariant-check` returns no violations on the `corpus/` store (25 sources, 198 claims, 5 `distinguish` resolutions, 12 aliases, 8 draft concepts), and still none after removing each of its 25 sources in turn (checked 2026-09-28).

Two pre-existing alias bugs were found by the property test and fixed:
- merging into a subject that was itself an alias left claims under a non-canonical subject;
- overwriting an alias could create a resolution cycle.

`remove_source` no longer fails the `superseded_by` foreign key when another source's claim was superseded by one of its claims.

A code review of the Phase 0–1 branch (PR #1) found more channels by which a derived state could outlive its grounds. Each was fixed with a regression test in `tests/test_invariants.py`:
- re-disposing a `replicate` compounded or stranded its confidence bump;
- `lint --resolve-by-recency` force-resolved pairs onto inactive claims;
- `concept-validate` could revive superseded or invalidated concepts;
- `source-unretract` revived claims a `retracted` disposition had dropped, and `source-authority-set` could flip retraction without the cascade;
- `retract_source` and `source-replace` weren't atomic;
- the predicate-alias cycle check only looked at the end of the chain;
- `alias-undo` restored an overwritten alias target without a cycle check. The `thorough` profile found this during the Phase 2 work (A→B overwritten to A→C, then B→A, then undo). The undo is now refused with `alias_would_cycle`.

**Exit criteria:**
- `tests/test_invariants.py` passes under the `thorough` profile (1000 examples × 60 steps) in CI.
- The canary benchmark shows 0% leakage (no view, concept or disposition that reaches a canary claim survives its removal or retraction) and reports the over-invalidation rate.
- `invariant-check` returns no violations on the benchmark store and on a store built from `corpus/`.

### Phase 2: Claim fidelity and a graded verifier
- A `proposition` field and a context window on claims; the triple becomes an index.
- Deterministic fidelity checker: numbers, dates, units, negations and entities must appear in the span or its context.
- A review queue for claims, contradictions, aliases and concepts.
- A layered verifier (deterministic → entailment → LLM) with SUPPORTED / UNSUPPORTED / UNCERTAIN verdicts.
- A human-labeled verifier eval set (200–300 pairs) with published false-accept and false-reject rates.

**Order within the phase:** build the eval set first and measure the current verifier on it, so the layered verifier is judged against a real baseline.

**Status:** code in place (2026-09-28); the two eval exit criteria are open because they need human labels and a live-API run.

- **Proposition and context window** (schema migration 2): `claims.proposition` holds the claim as a sentence, and `context_start`/`context_end` hold the enclosing sentence plus one neighbour on each side, never crossing a paragraph break. Migration 2 backfills context windows for existing claims and leaves their proposition NULL rather than inventing one from the triple. `claim-add --proposition`, and ingest's extraction prompt now asks for one.
- **Fidelity checker** (`aleph/fidelity.py`): runs on every claim write. Flags go to the review queue; they never refuse a write. `claim-fidelity-check [ID | --all] [--enqueue]` re-checks existing claims. It was tried on a migrated copy of a local store built from `corpus/` (the store isn't in the repo). It flagged 49 of 198 claims:

- 30 number issues;
- 19 negation issues (9 added, 10 dropped);
- 10 entity issues;
- 4 date issues.

Many of the flags are objects that carry article numbers, years or citations the span doesn't contain. None of them has been triaged into real findings and false positives yet.
- **Review queue** (`review_queue`; `review-list`, `review-add`, `review-resolve`): holds claims, contradictions, concepts and aliases. Recording a decision changes nothing else. Reviews cascade with their target, and `alias-undo` closes alias reviews as `obsolete` (checked by `check_invariants`).
- **Layered verifier** (`aleph/verifier.py`) runs three layers, and each layer decides or passes. (1) Deterministic: a verbatim restatement of the whole span is SUPPORTED; an added number, date, unit or negation is UNSUPPORTED. (2) Optional NLI entailment (`ALEPH_ENTAILER=nli`). (3) LLM. `ask --verifier layered` opts in, and it bypasses the view cache. It stays opt-in until the eval shows it beats the baseline.
- **Eval harness** (`python -m aleph.verifier_eval`, [benchmark/verifier_eval/](../benchmark/verifier_eval/)): the pair format, a labeling rule, false-accept/false-reject/uncertain rates, and runners for `baseline`, `layered` and `deterministic`. `pairs.jsonl` has 49 **draft** pairs written and labeled by Claude. The draft set is a harness fixture, not the eval set. The 200–300 human-labeled pairs are still to do.
- **Ingest cost**: every ingest result reports token usage, list-price cost, and cost per 1k source tokens. Source tokens are estimated as chars / 4. A model with no known price reports cost as unknown.

A code review of this branch found several defects, each now fixed with a regression test:

- the deterministic layer accepted verbatim fragments of a negated span;
- decimal points were dropped when comparing numbers;
- the ingest cost report left out the `--extract-conditions` pass;
- accepted reviews were re-queued;
- alias reviews outlived the merge they were filed on;
- layered and default `ask` shared cached views;
- multi-citation sentences were rejected;
- migration 2 was not atomic.

Also fixed: `_init_fts` never backfilled a store created before FTS existed, because `SELECT rowid FROM claims_fts` reads the content table rather than the index. The next UPDATE on such a store corrupted the index. It now rebuilds when the index's docsize count differs from the claims count.

**Exit criteria:**
- Baseline and new false-accept / false-reject rates are published.
- The layered verifier's false-accept rate is lower than the baseline's, and its false-reject rate isn't worse.
- The fidelity checker flags every seeded number, date, negation or entity mismatch in a fixture set. **Met for the fixture:** 25 seeded cases and 12 faithful controls in `tests/test_fidelity.py` run in CI. The seeded cases include decimal shifts (9.0 vs 90), scale words (million vs billion), reordered ISO dates, and a negation that sits elsewhere in the span. The checks are lexical, so the fixture can't prove coverage beyond these patterns.
- Ingest reports its LLM cost per 1k source tokens, so the cost of extraction is known before Phase 6. **Reporting in place;** not yet run against the live API on `corpus/`.

### Phase 3: Omission-aware retrieval and scalable contradictions
- Hybrid FTS + embedding retrieval (RRF).
- Contradiction- and condition-aware expansion; unresolved conflicts shown to the synthesizer.
- `claim_ids_unused` ("evidence considered but not used").
- A counter-evidence check on the final answer.
- Predicate blocking, a stricter pre-filter, and review-queue routing for detected contradictions.
- Recency resolution by source date; legal-domain sources ranked by authority.

**Exit criteria:**
- Hybrid retrieval beats both FTS-only and embedding-only on claim recall@k over a labeled query set. **Open:** the harness and a draft query set exist; human labels and an embeddings run are needed.
- The contradiction pre-filter cuts LLM pair judgements by a measured factor on `corpus/` without losing any contradiction in a hand-labeled sample. **Deferred** until the Phase 4 core is decided.
- `lint --resolve-by-recency` orders by source date, with a test. **Met:** `tests/test_recency.py`, in CI.

The contradiction-scaling work is conditional on Phase 4 keeping contradictions in, or close to, the core (see Sequencing notes).

**Status (2026-09-29):** everything except the contradiction-scaling item is in place. The retrieval exit criterion is unmeasured.

- **Recency by source date** (`lint.resolve_by_source_date`; `resolve_by_recency` wraps it). The source's own date decides (`authority.source_date`: issued, published, effective or reviewed, depending on the domain), never extraction time. Two legal sources are compared only within one legal order:
  - authority level decides first (lex superior), including across nested jurisdictions (US over US-CA);
  - then the newer date decides (lex posterior), but only within the same jurisdiction.

  Each step needs its field on both sources. A missing field never counts as lowest.

  Pairs are resolved oldest first, so in a chain A > B > C the oldest claim C doesn't survive.

  These pairs stay open, each reported with its reason:
  - pairs from different domains (`mixed_domain`);
  - legal pairs where only one source has a jurisdiction (`unknown_jurisdiction`);
  - legal pairs from unrelated jurisdictions (`cross_jurisdiction`);
  - same-level legal pairs from nested jurisdictions (`nested_jurisdiction`): that's preemption, not lex posterior;
  - legal pairs with a missing authority level or a one-sided specificity (`unranked`);
  - legal pairs split by specificity (`lex_specialis`): a special rule displaces the general one only within its scope, which is `distinguish`, not `supersede`;
  - cross-subject pairs;
  - undated pairs and ties;
  - pairs that `dispose` refuses.
- **Expansion:** `ask` and `compose` add claims that conflict with, or are conditions of, retrieved claims, up to `k // 2` of them. Each goes through the context filter and is labeled "included because: …", which names the disposition (conflicts with, replicates, coexists with, …). Pairs with differing predicate senses are skipped. The claims block lists each claim's conditions. Open conflicts (never disposed, or reopened by the cascade) appear in the dispositions block as `unresolved`, with a prompt rule to present both sides. `--no-expand` turns this off. Adding a contradiction, disposing it, reopening it, or resolving it with `contradiction-resolve` now invalidates views citing either side, because their prose may describe the old conflict state.
- **`claim_ids_unused`:** the claims shown to the synthesizer but not cited. `view_cache.considered_claim_ids` stores them. It isn't an invalidation index: a cache hit filters it to claims that are still active.
- **Counter-evidence check** (`query.counter_evidence`; the agent-mode command is `counter-evidence --claim-ids`): the uncited side of any live conflict (open, `dispute` or `gap`) involving a cited claim. It is computed on every read, applying the query's context filter and skipping pairs with differing predicate senses. So it reflects later changes to the counter claims without invalidating the view.
- **Hybrid retrieval** (`HybridRetriever`, reciprocal rank fusion with k=60; `ALEPH_RETRIEVER=hybrid` fuses FTS with embeddings) and a **retrieval eval** (`python -m aleph.retrieval_eval`, [benchmark/retrieval_eval/](../benchmark/retrieval_eval/)).
  - The query sets carry **draft** labels.
  - On a local store built from `corpus/` (agent-extracted claims; the store isn't in the repo), recall@10 is 0.79 for FTS and 0.73 for keyword. The labels are draft.
  - The embedding and hybrid retrievers haven't been run: they need the `embeddings` extra.
- **Deferred:** predicate blocking, the stricter pre-filter, and review-queue routing for detected contradictions. They wait on the Phase 4 core decision, per the sequencing notes.

Also fixed: a non-default view was being written to the view cache under the default key, where it overwrote the default view. This covered the layered verifier, expansion off, and `--no-verify`. Such views now stay out of the cache.

A code review of this branch found more defects, each now fixed with a regression test:

- contradictions settled with the legacy `contradiction-resolve --keep` were treated as unresolved;
- auto-resolution superseded cross-subject pairs;
- legal ranking let missing fields and unrelated jurisdictions decide;
- views weren't invalidated when a contradiction changed;
- disposition grouping read only the newest 1000 contradictions in the store;
- `HybridRetriever` broke retrievers that implement only the minimal protocol;
- the eval's source matching matched `data.txt` for `a.txt`.

A second review found that:

- a newer state statute could supersede an older federal one;
- a legal source with no jurisdiction was compared with one that had a jurisdiction;
- counter-evidence ignored the query context;
- expansion called replications "conflicts";
- chains of pairs could leave the oldest claim active.

All of these are fixed with tests. Still open: every contradiction add scans all cached views, which costs about 12 ms per add at 5,000 views. This matters only when bulk `contradiction-scan` runs against a large cache.

### Phase 4: Minimal core with optional extensions
- A `features` store config. The core is sources, claims, views and the cascade; concepts, conditions, contradictions and authority become opt-in.
- Concepts are decomposed into atomic sub-statements, each grounded by a single span, labelled as synthesis, and not citable in core mode.

**Exit criteria:**
- With every extension off, the full test suite passes against the core schema.
- Turning an extension on for an existing store is a tested migration.
- `invariant-check` stays clean in both modes.

### Phase 5: Interfaces
- A curated Python API, an MCP server (`aleph mcp`), and a local review UI (click-to-span, mark-wrong, review queue).
- PDF ingestion that keeps the verbatim-span contract. Hyphenation, column order and running headers must not break span location. Real legal and scientific corpora are mostly PDFs; ingest reads only `.txt/.md` today.

**Exit criteria:**
- The review UI supports the Phase 6 seeded-error study end to end: show an answer, click to its span, mark a sentence wrong, log timing.
- The MCP server exposes the agent-mode surface with the same JSON envelope.

### Phase 6: Evaluation program
- **Code properties:** invariants, canary leakage, over-invalidation, regeneration churn, root-cause localization.
- **Measured results:** attribution on ALCE/QASPER against chunk-cited RAG; retraction leakage against a summary-caching wiki; contradiction discovery; cost and scale at 10k and 100k claims.
- **User studies:** seeded-error review (speed × detection), automation bias, span sufficiency, residual error among unflagged sentences.

**Exit criteria:** each measured result and user study has a pre-registered hypothesis and a published result, including null results. Those results decide the Deferred items.

### Deferred
- Lazy, per-chunk extraction; widening the concept, condition and disposition machinery. Only if Phase 6 shows these add value.

## Sequencing notes

Phases are mostly sequential. Two pieces of evaluation should run earlier than Phase 6, because later phases depend on what they find:

1. **Verifier baseline** runs at the start of Phase 2 (see above).
2. **A rough RAG attribution comparison**, using a small ALCE or QASPER slice against chunk-cited RAG, runs after Phase 2 and before Phase 5. If Aleph doesn't beat chunk-cited RAG on attribution, the interface work in Phase 5 should wait.

**Decide the Phase 4 core before the Phase 3 contradiction work.** Phase 3 scales contradiction detection. Phase 4 may make contradictions opt-in, and Deferred makes their expansion depend on Phase 6. Decide what the core is (the `features` split) before scaling any extension. Until then, limit Phase 3 contradiction work to what the evaluation needs.

**Planning docs.** This file is the authoritative plan. `docs/backlog.md` and `docs/improvement-proposals.md` are input to it. A backlog item isn't scheduled until it's mapped to a phase here.
