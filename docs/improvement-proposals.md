# Improvement proposals — from one end-to-end agent-mode run

Prioritized list of concrete fixes to Aleph based on friction observed during a real run (Italian criminal-law corpus; 25 sources, 198 claims, 8 concepts, 5 contradictions; full retrospective in [agent-mode-evaluation.md](agent-mode-evaluation.md)).

Priority legend:
- **P0** — blocks a common workflow; fixable in a day; high ROI
- **P1** — friction that real users will hit; 1–3 days of work
- **P2** — nice-to-have, architectural direction, or multi-day work

Scope estimates assume one focused engineer familiar with the codebase.

---

## P0 — surface fixes

### P0.1 — Normalize `contradiction-add` to flag-based args

**Problem.** Every other write uses `--flag` form (`claim-add --source-id N --subject … --span …`). Only `contradiction-add` takes positional args (`contradiction-add CLAIM_A CLAIM_B`). I wrote the wrapper assuming flags and had to fix it. New agent users will hit the same inconsistency.

**Proposed change.** Accept both forms — keep positionals for back-compat, add `--claim-a` / `--claim-b` — or cut over to flags with a one-release deprecation warning. Code is in [`cmd_contradiction_add`](../src/aleph/agent_cli.py) around line 338.

**Scope.** 30 minutes, including tests.

---

### P0.2 — Add `details` to `rationale_concept_not_active` error

**Problem.** When `contradiction-dispose --rationale-concept <N>` refuses, the error body is just `{"code": "rationale_concept_not_active", "message": "rationale_concept_not_active"}`. Other errors in the same module include structured `details` (`source_id`, `span_preview`, `contradiction_claims`, `subject_a` / `subject_b`). I had to separately call `concept-get` to learn the concept was `draft`.

**Proposed change.** In `cmd_contradiction_dispose`, surface `{"concept_id": N, "current_status": "draft", "required_status": "active"}` in the error's `details` field.

**Scope.** 15 minutes.

---

### P0.3 — Document the `--skip-validation` dead-end

**Problem.** `concept-add --skip-validation` creates a concept in `draft`. The documented path to promote it to `active` is `concept-validate`, which calls the LLM. If you're in a fully-agent-mode run (no API key), there is no promotion path and no error telling you so. Your drafts accumulate; any attempt to cite them via `contradiction-dispose --rationale-concept` fails. This isn't obvious from the handler docstrings.

**Proposed change.** Near-term: add an explicit note in the output of `concept-add --skip-validation` — `{"status": "draft", "note": "promotion to active requires concept-validate (LLM)"}`. In [CLAUDE.md](../CLAUDE.md) under the concept section, document that drafts cannot be rationale concepts. Long-term: see P1.2.

**Scope.** 1 hour for the note; CLAUDE.md edit is a small rewrite.

---

## P1 — ergonomics and completeness

### P1.1 — Safer `--support` / `--conditions` syntax

**Problem.** Both `concept-add --support "26:premise,174:premise,…"` and `claim-add --conditions "12:sample,17:method"` are colon-and-comma-separated strings inside a single argv slot. A stray space, a missed colon, an unquoted shell argument — all produce unhelpful errors. On `concept-add` I hand-built the string in Python and it worked, but it's a footgun.

**Proposed change.** Accept JSON in addition to the colon syntax: `--support '[[26,"premise"],[174,"premise"]]'` or `--support-json '{"26":"premise","174":"premise"}'`. Keep the legacy syntax as a sugar. Same for `--conditions`. The parsers in `cmd_concept_add` and `cmd_claim_add` become a two-liner with fallback.

**Scope.** Half a day for both, including tests.

---

### P1.2 — Add an attested-promotion path for concepts

**Problem.** When Aleph is driven fully in agent-mode (operator *is* the LLM), `concept-add --skip-validation` is a one-way ticket to `draft`. There's no way to express "operator attests this is grounded in the support spans". Result: my 8 concepts stayed draft and could not be used as `contradiction-dispose --rationale-concept`.

**Proposed change.** Split the lifecycle:

```
draft → attested → active
draft → rejected (already exists)
active → stale → new concept (already exists)
```

- Add `concept-attest <CID> --attested-by "<string>" --rationale "<string>"`: sets status `attested`, records the attestor.
- Extend `contradiction-dispose` rationale-concept check: accept `active` unconditionally; accept `attested` with an `attested_by` reference written into the contradiction row; reject `draft`.
- Surface `attested` in `concept-list --status` and in `concept-get` output.

This keeps the honesty contract (you can't silently promote drafts) while giving agent-mode a principled escape hatch.

**Scope.** 1–2 days including tests and schema migration (new column on concepts; new optional columns on contradictions to record attestation trail).

---

### P1.3 — Locale-aware subject normalization

**Problem.** `normalize_subject` in [db.py](../src/aleph/db.py) applies English singularization rules (`-ies → -y`, `-sses → -ss`, trailing `-s` unless `-ss`/`-us`). For my Italian corpus this meant `articoli`, `coniugi`, `sentenze`, `aggravanti`, `attenuanti` were not collapsed — I had to seed 14 aliases by hand, and because claims were already in the canonical form when I ran alias-add, zero existing claims got rewritten. The alias mechanism is there to patch non-English normalization, but the gap is real.

**Proposed change.** Introduce a `Normalizer` protocol with swappable implementations. Default stays `EnglishNormalizer`; add `ItalianNormalizer` with the common endings (`-i → -o` / `-a` for adjectives and nouns by gender, `-zioni → -zione`, `-ità → -ità`, apostrophe and dot handling).

Either (a) select via `Store(db_path, locale="it")`, or (b) set at init time via a `aleph init --locale it` one-time store config stored in a new `store_config` row, or (c) let users register custom normalizers as Python callables via a config hook.

**Scope.** 2–3 days. The English rules are 5 lines; writing correct Italian rules is about 20 lines. The plumbing (protocol, config, tests against multilingual fixtures) is the real work.

---

### P1.4 — Add a `source-yield` diagnostic

**Problem.** After ingesting 25 sources and 198 claims, I had no quick way to ask "which source produced few claims per KB of content?" I only learned the 198-vs-300-target delta because the subagent self-reported. `stats-json` returns only totals.

**Proposed change.** New agent command `source-yield`:

```json
{
  "ok": true,
  "data": {
    "sources": [
      {
        "source_id": 5,
        "path": "/…/art-577-cp.txt",
        "size_bytes": 1781,
        "active_claims": 8,
        "claims_per_kb": 4.5,
        "superseded_claims": 0,
        "flagged_thin": false
      }
    ],
    "summary": {
      "median_claims_per_kb": 3.2,
      "thin_threshold": 1.0,
      "thin_sources_count": 2
    }
  }
}
```

Useful for triaging before phases that depend on claim density (concept derivation, contradiction scan).

**Scope.** Half a day.

---

### P1.5 — `--compact` mode on `claim-by-subject` and `claim-search`

**Problem.** One shell loop over 16 subjects dumped 39 KB of JSON into an agent's context — mostly full object strings. Agent consumers typically need the shape (IDs + subject + predicate + confidence), not the whole object text.

**Proposed change.** Add `--compact` / `--fields <list>` to `claim-by-subject`, `claim-search`, `claim-by-subject`, `claim-list`:

```bash
aleph claim-by-subject "omicidio" --fields id,predicate,confidence
# or
aleph claim-by-subject "omicidio" --compact  # IDs + predicate + first 60 chars of object
```

**Scope.** 2–3 hours.

---

### P1.6 — Escape valve for cross-subject contradictions

**Problem.** [`cmd_contradiction_add`](../src/aleph/agent_cli.py) rejects claim pairs whose subjects differ. The reasoning in the docstring ("almost certainly a bug in the caller's reasoning") is defensible as the default. But legitimate cross-subject regime contradictions exist: in my run, claim_31 (art. 577 c.3 pre-reform, subject=`omicidio`) and claim_148 (Corte Cost. 197/2023, subject=`femminicidio`) are in genuine doctrinal tension — the Constitutional decision struck down the art. 577 rule. I had to drop the pair.

**Proposed change.** Add `--cross-subject --relation-kind <regime-supersedes | rule-limits-rule | doctrinal-cross-ref> --justification "<string>"` to `contradiction-add`. The cross-subject pair is recorded with a distinct flag in the contradiction row; `contradiction-list` can filter on it; `_group_by_disposition` in the query pipeline can choose whether to surface them (default: yes, but visually distinguished).

**Scope.** 1 day including tests and query-pipeline integration.

---

### P1.7 — `fetch_method` / `provenance_notes` on source metadata

**Problem.** Three of my 25 sources are WebFetch-distilled renderings of the original HTML, not the Gazzetta Ufficiale PDF. This is epistemically relevant downstream — a verifier checking a claim against a *distilled* rendering is different from checking against the authoritative original. Aleph correctly trusts the file it was given, but there's no in-store place to record the epistemic chain. I recorded it in `corpus/manifest.json`, outside Aleph.

**Proposed change.** Extend `source_metadata` schema with two optional fields:

- `fetch_method`: enum `direct | distilled | ocr | transcription | pasted`
- `provenance_notes`: free-text string

Both optional. Validators in [authority.py](../src/aleph/authority.py) warn (not error) if `fetch_method != "direct"` and there's no `provenance_notes`. The query pipeline can surface "cited source is a distilled rendering" in the verifier-flags footer.

**Scope.** Half a day for schema + validator extension; another half day to surface it in the synthesize/verify output.

---

## P2 — longer-term / architectural

### P2.1 — `--mock-llm` / fixture mode for LLM-gated commands

**Problem.** `ingest`, `ask`, `concept-validate`, `concept-rebuild`, `contradiction-scan`, `claim-condition-extract` all either consume real API tokens or fail. Developers can't smoke-test the full pipeline — including `SYNTHESIZE_V2` and the per-sentence verifier — without incurring cost or getting a key. Plus, CI for the end-to-end `ask` path currently requires a live API credential.

**Proposed change.** Introduce a `MockLLM` adapter in [llm.py](../src/aleph/llm.py) that reads a fixtures file (YAML or JSON) mapping `(system_prompt_hash, user_prompt_hash) → canned_response`. Enable via `ALEPH_LLM_FIXTURES=path/to/fixtures.yaml` env or `--mock-llm path`. Add a small set of fixtures for the V2 prompts and the verifier prompts so CI can exercise the golden paths.

**Scope.** 2–3 days including fixture set. Big payoff for CI and developer experience.

---

### P2.2 — Dashboard or `--report` command consolidating diagnostics

**Problem.** During the run I ran `stats-json`, `source-list`, `alias-list`, `contradiction-list`, `concept-list` separately to understand the store's state. This is fine for scripting but there's no one-shot "give me everything" view.

**Proposed change.** `aleph report [--format text|markdown|json]` that produces a single structured snapshot:

- Source counts by authority_type, authority_level bucket
- Claim counts by subject (top 20), by status
- Concept counts by status, by subject
- Contradiction counts by disposition, open vs. resolved
- Alias count, with top-5 aliases by claims-rewritten
- Cache state
- Top-10 claim subjects without supporting concepts ("candidates for concept-derive")
- Top-10 claim pairs with same subject and opposing predicates ("candidates for contradiction-scan")

The last two become gateway hints for the LLM-gated commands.

**Scope.** 2 days. Mostly composition over existing handlers.

---

### P2.3 — Retrieval pluggability, documented

**Problem.** [`retrieval.py`](../src/aleph/retrieval.py) already supports FTSRetriever and KeywordRetriever, and CLAUDE.md flags this as "the one swap point". But there's no documented recipe for how a user would plug in BM25, embeddings, or a hybrid. This matters because the P1.3 normalizer is a smaller swap-point hammer; users who want embeddings-first retrieval are more common than users who want custom subject normalization.

**Proposed change.** Three improvements:

1. A `Retriever` protocol documented in the module docstring with the exact method signature the rest of the pipeline calls (`search(keywords: list[str], limit: int) -> list[ClaimRow]`).
2. A reference implementation: `EmbeddingRetriever(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")` that computes and caches claim embeddings in a new `claim_embeddings` table. Optional dependency on `sentence-transformers`.
3. A config hook so `query.py` picks the retriever via `ALEPH_RETRIEVER=embedding` or `Store.default_retriever` without monkey-patching.

**Scope.** 3–5 days for a proper embedding backend with caching; the protocol + docs alone are a day.

---

### P2.4 — Provenance chain visualization

**Problem.** Given a `[claim:N]` citation, the chain is `claim → source → authority_metadata → fetch_method/provenance`. There's no CLI that walks this chain in one call.

**Proposed change.** `aleph provenance <claim_id>`:

```json
{
  "claim_id": 148,
  "subject": "femminicidio",
  "predicate": "prevede",
  "object": "…",
  "span": "…",
  "span_start": 1234,
  "span_end": 1412,
  "source": {
    "source_id": 22,
    "path": "corpus/dpc-femminicidio-giuridico.txt",
    "sha256": "…",
    "authority": {
      "domain": "legal",
      "authority_type": "guidance",
      "authority_level": 30,
      "jurisdiction": "IT"
    },
    "fetch_method": "direct",
    "provenance_notes": null
  },
  "conditions": [
    {"condition_claim_id": 5, "kind": "scope", "explicit": true}
  ],
  "concepts_citing_this_claim": [8],
  "contradictions_involving_this_claim": []
}
```

One call gets you the whole picture for an auditor. Useful when a client or reviewer challenges a specific cited claim.

**Scope.** 1 day. Mostly stitching existing helpers.

---

### P2.5 — Agent-mode synthesis helper (optional counterpart to `ask`)

**Problem.** Without the LLM, there is no agent-mode equivalent of `aleph ask`. The operator has to retrieve claims + concepts manually, compose the answer, and hand-verify sentence-by-sentence. I did this in my Phase H — it works, but it's strictly worse than the pipeline's automated output.

**Proposed change.** `aleph compose --query "<question>" [--context JSON]` that:

1. Runs retrieval + context filter + disposition grouping (all already exists, no LLM).
2. Emits a **structured brief** for the operator:
   ```
   [TOP-K CLAIMS]
     claim:148 — "Corte Cost. 197/2023 …"
     claim:170 — "…"
   [RELEVANT CONCEPTS — ACTIVE]
     concept:3 (legittima difesa) — "…"
   [RELEVANT CONCEPTS — ATTESTED]
     concept:7 (donna maltrattata) — "…" [attested by: agent-2026-04-23]
   [DISPOSITION GROUPINGS]
     distinguish: contradiction:3 (regola generale vs. domicilio regime)
     dispute: contradiction:2 (foreign divergence)
     gap: {claim:166 vs claim:172 — reconcile downgraded to dispute, no shared conditions}
   [GAPS DETECTED]
     Subject "omicidio preterintenzionale" has no active concept.
     Subject "eccesso colposo" has no active concept.
   ```
3. The operator writes the answer themselves, citing claim/concept IDs. No LLM, but full pipeline structure is exposed.

This is what I manually did in Phase H. Making it a command turns it from ad-hoc scripting into a first-class agent-mode workflow.

**Scope.** 2–3 days. Most components exist; the new code is the assembly and the structured output format.

---

## Meta-observation: document the agent-mode-first path

The runbook I followed (ingest → authority → aliases → concepts-as-drafts → contradictions → manual case query) is **more of the product than the docs currently emphasize**. [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md) lean on the API-mode demo; an explicit "Agent-mode quickstart — running Aleph without an API key" section would:

1. Make clear which commands require the LLM (the `LLM_COMMANDS` set at [cli.py:20](../src/aleph/cli.py#L20))
2. Show the manual counterpart workflow for each LLM-gated command
3. Set expectations on what's *lost* without the LLM (drafts, no per-sentence verifier, no automated contradiction scan)
4. Set expectations on what's *not* lost (grounding, citation integrity, disposition discipline, provenance)

Most "optional LLM" tools collapse to static CRUD without the key. Aleph is unusual in that it doesn't — the honesty contracts survive. That's a selling point worth articulating.

---

## Not recommended

Things I considered and *don't* think should change:

- **Verbatim-span strictness at the agent boundary.** Several frustrations in my run (3 recovered span rejections, the 2 aliases that collapsed to no-op under normalization) tempted me to argue for fuzzy-match fallback. On reflection, the strictness is the feature. Any relaxation leaks.
- **Disposition vocabulary.** The set `supersede | coexist | distinguish | reconcile | replicate | dispute | retracted | gap | unresolved` covered every doctrinal relationship I found. Don't add more until a specific gap is observed.
- **Single-subject canonicalization via `normalize_subject`.** The alias mechanism handles the edge cases; normalize_subject doing *more* would hide more surprises. The locale gap (P1.3) is real, but the design of normalize-then-alias-if-needed is correct.
- **SQLite as the backend.** The honesty contracts don't depend on the backend. SQLite + WAL + 5-second busy_timeout was more than enough for a single-writer agent workflow. No reason to swap.
