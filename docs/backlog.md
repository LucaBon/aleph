# Aleph backlog — prioritized

Session-grounded task list derived from a second end-to-end agent-mode run (2026-04-23) on the Italian criminal-law corpus (25 sources, 198 claims, 8 draft concepts, 5 resolved contradictions, 1 case evaluation cached via `view-cache`).

Cross-references:
- Prior retrospective: [agent-mode-evaluation.md](agent-mode-evaluation.md)
- Prior proposals (overlaps tagged below): [improvement-proposals.md](improvement-proposals.md)

Each item is tagged:
- **[NEW]** — not present in prior proposals
- **[CONFIRMS Pn.m]** — new field evidence for an existing proposal; upgrade priority if marked
- **type: feature | bug | improvement | docs**

Priority legend:
- **P0** — breaks a stated invariant or blocks a common workflow
- **P1** — real friction at normal use
- **P2** — polish / direction

---

## P0 — invariant-breaking or workflow-blocking

### P0.A — `view-cache` CLI silently drops concept_ids
**[NEW]** **type: bug**

The DB has a `view_cache.concept_ids` column (JSON default `'[]'`), invalidation helpers `_invalidate_cache_for_concepts` / `mark_concepts_stale_for_claims` already exist, and [query.py](../src/aleph/query.py) writes both claim_ids and concept_ids on cache insert — but the **CLI `view-cache` subcommand only accepts `--claim-ids CSV`**. In today's run I cited 5 concepts in a cached opinion and stored 0. If any cited draft concept is later invalidated, `_invalidate_cache_for_concepts` will not touch this cached entry → silent cache staleness in violation of the documented invariant.

Fix (≤ 30 min):
- Add `--concept-ids CSV` to `cmd_view_cache` in [agent_cli.py](../src/aleph/agent_cli.py).
- Thread it through to the insert (symmetric with how API-mode writes concept_ids in [query.py](../src/aleph/query.py)).
- Mirror in `view-get` output.

### P0.B — Early refusal for LLM-gated commands when `ANTHROPIC_API_KEY` is absent
**[NEW]** **type: feature**

Today the `LLM_COMMANDS` gate at [cli.py:20](../src/aleph/cli.py#L20) only checks *at subparser time* for a handful of command names. The refusal goes through, but:
- A caller can set `ANTHROPIC_API_KEY=dummy` and the command proceeds to a late `LLM.complete` failure deep in the pipeline, long after they've paid I/O cost.
- There is no user-facing **upgrade path hint** ("the agent-mode counterpart is X").

Fix:
- Validate the key at gate time (`_looks_valid_anthropic_key` — prefix/length check, not network).
- On refusal, print the LLM-mode command *and* the closest agent-mode substitute from a static table (`concept-validate` → "use `concept-add --skip-validation` + manual attestation; see P1.2"; `contradiction-scan` → "use `contradiction-add` pair-wise with `classify_pair` heuristic"; `ask` → "retrieve with `claim-search`/`concept-list`/`contradiction-list --all`; compose manually; persist via `view-cache`").

This closes the "silent LLM-only" footgun observed by both this session and the prior one.

### P0.C — Concept promotion path in agent mode
**[CONFIRMS P1.2 — upgrade to P0]** **type: feature**

Same finding, confirmed by a second independent run. All 8 concepts in the current DB are stuck at `draft`; they are citable in prose but not as `rationale_concept` in `contradiction-dispose` (which refuses with `rationale_concept_not_active`). This is the single workflow friction that forced me to annotate 5 concepts as `(concept draft — unvalidated)` in the cached opinion — an honesty workaround the user has to live with.

The prior proposal **P1.2** (`concept-attest`) is the right shape; this session adds two data points:
- A full KB of 8 concepts can be built and contradictions disposed **without ever reaching `active`** — the agent-mode path is production-feasible today except for this single gate.
- `concept-add --skip-validation` already returns a `"note"` field per [cli.py](../src/aleph/cli.py), but `contradiction-dispose` doesn't surface structured details (see [P0.2](improvement-proposals.md#p02--add-details-to-rationale_concept_not_active-error)). Both should land in the same patch series.

Recommended path: implement **P1.2** from [improvement-proposals.md](improvement-proposals.md#p12--add-an-attested-promotion-path-for-concepts) + **P0.2** together.

---

## P1 — real workflow friction

### P1.A — `claim-by-subject` has no `-k` / `--limit`
**[NEW]** **type: improvement**

`claim-search` accepts `-k N`; `claim-by-subject` does not. `claim-by-subject "omicidio"` returned 25+ rows in a single envelope — fine for my uses but forced me to fall back to `claim-search`. Trivial inconsistency; surface parity with the sibling command.

Note: [P1.5](improvement-proposals.md#p15---compact-mode-on-claim-by-subject-and-claim-search) already asks for `--compact` on this command — add `-k` in the same patch.

### P1.B — Near-duplicate claim clustering in retrieval
**[NEW]** **type: improvement**

`claim-search "coniuge" -k 15` returned claim 26 (from [corpus/art-577-cp.txt](../corpus/art-577-cp.txt)) and claim 174 (from [corpus/brocardi-577-commento.txt](../corpus/brocardi-577-commento.txt)) both asserting essentially the same fact ("ergastolo if omicidio against coniuge") from different sources, each at confidence 0.95. The agent has to notice the overlap; the retriever doesn't flag it.

Two options:
1. Cluster post-hoc on `(normalize(subject), normalize(predicate), shingles(object))` Jaccard ≥ threshold, return the representative + `duplicates: [ids]` so disposition-grouping ([query.py](../src/aleph/query.py)) can surface "this is the consensus across N sources" rather than counting it as N supporting votes.
2. Or: a `--dedup-threshold 0.8` flag on `claim-search`.

Option 1 is the better default because it also fixes a subtle downstream bias: the synthesizer currently sees two near-identical claims as two corroborating citations, inflating consensus strength.

### P1.C — Source-substitution provenance
**[CONFIRMS P1.7]** **type: feature**

`manifest.json` in this corpus openly flags 6 of 25 sources as **substitutions**: the file whose slug is `cass-pen-v-29527-2022.txt` actually contains Cass. I n. 12328/2017, because the former citation wasn't reachable. Aleph's grounding check is satisfied (the span *is* in the file), but the *citation in the file header* is misleading. A downstream reader of claim 111 learns `source_path="cass-pen-v-29527-2022.txt"` and has no signal that the case number on the slug is fictive.

Prior proposal [P1.7](improvement-proposals.md#p17---fetch_method--provenance_notes-on-source-metadata) asks for `fetch_method` + `provenance_notes`. Extend it with a third optional field:

- `substituted_from`: original citation/URL the slug promised; present when the file is a semantic substitute.

Validators in [authority.py](../src/aleph/authority.py) should warn when the source filename contains a citation pattern (e.g. `cass-pen-*-NNNNN-YYYY`) but `substituted_from` is unset and the file's verbatim content doesn't self-identify as that citation. Surface in the synthesizer's verifier footer: *"cited source is a substituted rendering — original citation was X"*.

This is the most serious honesty gap I observed in this session because it escapes the grounding invariant silently.

### P1.D — `stats-json` should report retrieval backend
**[NEW]** **type: improvement**

`_init_fts` in [db.py](../src/aleph/db.py) silently falls back to LIKE when FTS5 is unavailable. Two deployments with identical corpora will rank differently with no log signal. Add:
```json
{"retriever": "fts5" | "like", "fts_available": true|false}
```
to `stats-json`. Cheap, observable.

### P1.E — `view-cache --answer-file PATH` (or stdin)
**[NEW]** **type: improvement**

Today the answer is a positional argument. Long opinions need `$(cat file)` or heredocs, both of which choke on shell-metacharacter-laden content. Add `--answer-file PATH` (mutually exclusive with the positional) or accept stdin when answer is `-`.

---

## P2 — polish and direction

### P2.A — Skill catalogue split
**[NEW]** **type: docs**

The base skill [skills/aleph/SKILL.md](../skills/aleph/SKILL.md) covers only ingest/ask/lint. The codebase has five workstreams — WS-A (concepts), WS-B (typed contradictions), WS-C (source authority), WS-D (claim conditions), plus case-evaluation — and currently agent-mode callers have to derive workflows from CLAUDE.md + source reading.

This session added [skills/aleph-case/SKILL.md](../skills/aleph-case/SKILL.md) for case evaluation. The matching gap is:
- `skills/aleph-concepts/SKILL.md` — derive → validate → stale → rebuild lifecycle
- `skills/aleph-contradictions/SKILL.md` — scan → typed disposition → rule composition
- `skills/aleph-authority/SKILL.md` — metadata by domain, retraction, context filtering
- `skills/aleph-conditions/SKILL.md` — explicit vs inferred, overlap scoring

Each would be ≤ 200 lines and turns "read CLAUDE.md to figure it out" into "the skill discovery surface tells me exactly which workflow to run".

### P2.B — `aleph compose` / agent-mode `ask` counterpart
**[CONFIRMS P2.5]** **type: feature**

I manually built the compose flow for the second time in two sessions. The pattern is stable:

```
claim-search (batch by topic)  +
concept-list (filter by relevance)  +
contradiction-list --all (grouped by disposition) 
  → structured brief for the operator
```

`aleph compose --query "<q>" [--context JSON]` would promote this from ad-hoc scripting to a supported workflow. The prior proposal **P2.5** is the right shape; today's evidence is that the composition is *mechanical enough* that it could be scripted rather than redone by the operator every session.

### P2.C — Concept verdict provenance
**[NEW]** **type: feature**

Related to P0.C: when concepts are promoted, store who/what produced the verdict:
- `validation_verdict_source`: enum `llm | manual | inferred`
- `validation_verdict_at`: timestamp
- `validation_verdict_by`: string (model name for LLM; actor id for manual)

So a caller can audit "was this concept promoted by an LLM judging GROUNDED, or by a human attestation?" — same honesty contract as `claim_conditions.explicit`.

### P2.D — Context-template helper
**[NEW]** **type: feature**

`aleph ask --context '{...}'` is powerful but today requires hand-authored JSON. `aleph context-template --domain legal --jurisdiction IT` emitting a starter JSON with legal-specific fields (jurisdiction dotted-prefix, effective_at, include_retracted=false) lowers the barrier.

### P2.E — Locale migration
**[CONFIRMS P1.3]** **type: improvement**

Prior proposal P1.3 covers the missing Italian normalizer. Add to it: a `aleph locale-migrate --from en --to it` admin command that re-normalizes all subjects under a new locale, rewrites affected claims atomically (same mechanism `add_alias` uses), and invalidates cached views. Without migration, locale mistakes are permanent.

---

## Quick-wins — < 1 hour each

These are leaf patches I'd pick up first for immediate ROI:
1. **P0.A** — add `--concept-ids` to `view-cache` CLI (≤ 30 min).
2. **P1.A** — add `-k` to `claim-by-subject` (≤ 15 min).
3. **P1.D** — add `retriever` field to `stats-json` output (≤ 20 min).
4. **P1.E** — add `--answer-file` to `view-cache` (≤ 30 min).
5. [**P0.2** from existing doc](improvement-proposals.md#p02--add-details-to-rationale_concept_not_active-error) — add structured `details` to `rationale_concept_not_active` error (≤ 15 min).

Combined: ~2 hours of work that closes the most visible agent-mode footguns.

---

## Validated-working (no change proposed)

Things that worked well enough in this session that I do **not** recommend changing — repeated from the prior doc's "Not recommended" section with new supporting evidence:

- **Verbatim-span strictness.** 198 claims, zero false-grounded-pass observed.
- **Disposition vocabulary.** All 5 existing contradictions were cleanly expressible with the current set (`distinguish`, `reconcile`, `supersede`). No gap observed.
- **Agent-mode JSON envelope discipline.** `{"ok": true, "data": {...}}` / `{"ok": false, "error": {...}}` held across every command I ran today.
- **Concept lifecycle.** `draft → active → stale → (new concept via rebuild)` is the right shape. The gap is the promotion *mechanism* (P0.C), not the lifecycle.
- **Cache invalidation topology.** `_invalidate_cache_for_claims` and `_invalidate_cache_for_concepts` cover the right seams; the CLI exposure gap (P0.A) is the only hole.
