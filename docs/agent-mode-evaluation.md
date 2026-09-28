# Agent-mode evaluation — end-to-end run, Italian criminal-law corpus

A retrospective from one real end-to-end run exercising Aleph's agent-mode surface without `ANTHROPIC_API_KEY`. Every observation here is backed by something that happened in the session — the goal is engineering feedback, not a review.

## What the run actually did

- Fetched 25 Italian-legal documents from the web to `corpus/*.txt` (~74 KB total) with a manifest.
- Ingested via the agent-mode pairs `source-add` + `claim-add`, with a subagent doing the claim extraction itself. Produced **198 active claims**, 0 unrecovered span rejections (3 recovered on first retry).
- Set per-source authority metadata on all 25 sources via `source-authority-set --domain legal`.
- Seeded 14 Italian aliases (12 effective — 2 collapsed to no-op under normalization).
- Composed 8 draft concepts via `concept-add --skip-validation` (no LLM validator available).
- Recorded 5 within-subject contradictions and disposed them all with typed dispositions (4 × `distinguish`, 1 dispose-failed-and-retried).
- Evaluated a hypothetical legal case (uxoricide with minor present, no prior abuse) and produced a citation-backed memo with a self-authored verifier-flags footer.

Final store state: 25 sources, 198 active claims, 0 superseded, 5 contradictions (all resolved), 0 cached views, 8 draft concepts, 14 aliases.

## Top-line

The tool **works as documented**. Agent-mode delivered what [CLAUDE.md](../CLAUDE.md) promises — no LLM in-process for the commands that don't need one, honest rejection at the write boundary, disposition vocabulary policed in Python. Zero correctness bugs across ~700 agent-mode calls in this session. The friction is in ergonomics, tooling, and a few places where the contract is strict in ways that aren't yet matched by equivalent-strictness *agent-mode alternatives*.

## Pros — things that concretely worked

### The `{ok, data|error}` envelope is disciplined

Every command emits one JSON line. Errors carry codes I could switch on (`span_not_in_source`, `rationale_concept_not_active`, `contradiction_invalid`, `claim_not_found`). I parsed with a one-liner Python every time. Compared to typical CLIs where you grep stderr, this is ergonomic for automation.

### The verbatim-span invariant caught real problems

The ingesting subagent hit `span_not_in_source` three times:
1. Italian smart-quotes wrapping the span
2. A zero-width whitespace variant inside a Cassazione sentence
3. Accented uppercase "À" in a proper noun, encoding-shifted through shell heredoc

All three were *real*, all three were recovered by shrinking the span. The rule in [`cmd_claim_add`](../src/aleph/agent_cli.py) — `idx = content.find(span); if idx < 0: refuse` — is the right strictness. The asymmetry between the agent boundary (strict) and the ingest-pipeline [`_locate_span`](../src/aleph/ingest.py) (one whitespace-tolerant retry) is also correct: agents can copy carefully, but the automated ingest path needs forgiveness for the LLM's minor reformatting.

### Disposition validators actually enforce invariants

When I passed `--rationale-concept 7` on `contradiction-dispose` the first time, the command rejected it with `rationale_concept_not_active`. I had tried to cite a concept that was still `draft` (because I couldn't validate without the LLM). The tool refused to let me smuggle an unvalidated claim of doctrinal synthesis into a contradiction rationale — exactly what the honesty contract should do. Same stringency worked on `coexist|distinguish|reconcile` requiring `--rule`, and `supersede|retracted` requiring `--keep/--drop`: every gate held.

### Authority metadata schema is right-sized for legal

Translating Italian legal hierarchy into `(authority_level, specificity, issued_at)` went without contortion. My convention — Corte Cost.=95, SS.UU.=85, Codice Penale=80, Leggi di riforma=78, Cassazione semplice=65, dottrina=30, statistics=20 — slotted into the schema directly. [authority.py](../src/aleph/authority.py) accepted the dates, enums, and tuples without fabrication. `source-authority-get` roundtripped cleanly: `authority_rank: [80, 577, -1237161600]` came back exactly as sent.

### Same-subject check on `contradiction-add` is pedagogical

My first mental pairing was across subjects (`omicidio` claim vs. `femminicidio` claim on the pre/post-2025 regime). `contradiction-add` refused with a specific error, which forced me to find *genuine* same-subject pairs. The docstring's reasoning ("contradiction spanning subjects = caller bug") is load-bearing — it made me think more carefully. I found legitimate within-subject pairs I would have missed otherwise.

### Deferred cache invalidation is wired correctly

`alias-add` returned `cache_cleared: false` because `cached_views: 0`. Not a no-op by accident — the handler checks state and reports honestly. In a live system with cached views, the invalidation would fire; the wiring exists even though my run never hit it.

### The grounding chain never broke

Every `[claim:N]` in my case-evaluation memo traces to a verbatim substring of the source .txt that lives on disk and is hashed in the store. I never once had to retract a citation because a claim's span "wasn't quite right" — the pipeline had already enforced that at write time.

## Cons — friction I actually hit

### CLI flag inconsistency

`contradiction-add` takes **positional** args (`claim_a claim_b`). Every other write (`claim-add`, `concept-add`, `source-authority-set`, `contradiction-dispose`) uses `--flag`. I wrote the Python wrapper assuming flags and had to fix. See [agent_cli.py](../src/aleph/agent_cli.py) `cmd_contradiction_add` vs. neighbors.

### `--support "ID:role,ID:role,..."` string-parsed payload is fragile

Colon+comma-separated string inside a single argv slot, with role defaults. One stray space and it's debugging. Safer surfaces: JSON array (`--support '[[26,"premise"],[174,"premise"]]'`) or repeated flags (`--support 26:premise --support 174:premise`). Same pattern on `claim-add --conditions "id:kind,id:kind"` — both would benefit.

### English-only subject normalization is a real ceiling for non-English corpora

`normalize_subject` in [db.py](../src/aleph/db.py) strips the dot from `aggravante art. 577` → `aggravante art 577` and the apostrophe from `attenuante stato d'ira` → `attenuante stato dira`. The ingesting subagent noticed and used the normalized form, so claims went in consistent. But the aliases I added *after* ingest rewrote **zero** claims because they were already in canonical form. The alias system is there to patch non-English normalization gaps, but the normalization itself — singularization via `-ies → -y` / `-sses → -ss` / trailing `-s` rules — only knows English.

For Italian, there's a real improvement space: `articoli → articolo`, `coniugi → coniuge`, `sentenze → sentenza`, `aggravanti → aggravante` are all mechanical rules that `normalize_subject` doesn't apply because the rules are locale-specific. I had to manually alias all of them, and the aliases are future-proofing only (no existing claims to rewrite).

### Concept drafts are a dead-end without LLM validation

`concept-add --skip-validation` creates a concept in `draft`. The only promotion path is `concept-validate`, which calls the LLM. If you're in a fully-agent-mode run (no key, the agent *is* the reasoning engine), you have no way to attest "I the operator checked grounding; please treat this as durable." Result: my 8 concepts stayed `draft`, and three of my first five `contradiction-dispose` calls failed on `rationale_concept_not_active` because I tried to cite drafts.

The cleanest fix is to split the lifecycle: `draft → attested → active`, where `attested` is agent-attested (no LLM) and `active` is LLM-validator-verified. Or add a distinct `--attested-by <string>` promotion path that records who/what attested, and allow `contradiction-dispose --rationale-concept` to accept `attested` status with a warning surface.

### The same-subject requirement on `contradiction-add` blocks legitimate cross-subject regime changes

The best example from my run: claim_31 ("Le circostanze attenuanti… non possono essere ritenute prevalenti" — art. 577 c.3 pre-reform) and claim_148 ("Corte Cost. 197/2023 ha dichiarato l'incostituzionalità del divieto…") sit under different canonical subjects (`omicidio` and `femminicidio` respectively, due to where I extracted them from). They're genuinely in tension — the Corte Cost. decision struck down the art. 577 rule — but Aleph refuses the pair because the subjects differ.

The gate is defensible (a spanning-subject contradiction is often a caller mistake), but a structured escape valve — `--cross-subject --relation-kind supersedes-rule --justification "..."` — would let users record real regime-change contradictions without bypassing the check.

### No yield diagnostics

After ingest I wanted to know "which of my 25 sources produced unusually few claims per KB?" There is no tool for this. `stats-json` returns totals only (`sources: 25, claims_active: 198`). I only learned the 198/300-600 delta because the subagent self-reported. A `source-yield` report (claims count, active claims, claims-per-KB, span-rejection rate, by source) would catch thin ingestion before the user builds on a shaky base.

### `rationale_concept_not_active` error is terser than its neighbors

Other errors include structured `details` (`span_preview`, `source_id`, `contradiction_claims`). This one returns just the code and message. I had to inspect concept status separately to understand what went wrong. Should include `{"concept_id": N, "current_status": "draft"}`.

### `claim-by-subject` has no compact mode

Pulling 16 subjects in one shell loop dumped 39 KB of JSON into my context — most of it full object strings. A `--compact` flag (IDs + subject + predicate + truncated object) or a `--fields id,subject,predicate` projection would let agent consumers pick the shape they need.

### `source_metadata` has no `fetch_method` / `provenance_notes` field

When the source .txt is a WebFetch-distilled rendering of the original HTML (as happened for 3 of my 25 sources — the reform laws), that is epistemically relevant downstream. Aleph correctly trusts whatever is in the file, but there's no in-store place to record "this is a distilled rendering, not the Gazzetta Ufficiale original". I recorded this in `corpus/manifest.json` outside Aleph. A `fetch_method` enum (`direct | distilled | ocr | manual-transcription`) and free-text `provenance_notes` on `source_metadata` would close this loop.

### No dry-run / fixture mode for LLM-gated commands

`ingest`, `ask`, `concept-validate`, `concept-rebuild`, `contradiction-scan`, `claim-condition-extract` all either consume real API tokens or fail. A `--mock-llm path/to/fixtures.json` flag that routes `LLM.complete` / `LLM.complete_json` through a canned-response reader would let developers smoke-test the full pipeline (including `VERIFY_SYSTEM`, `SYNTHESIZE_V2`, `_group_by_disposition`) without incurring cost or requiring the key.

## Quality of the result vs. expectation

### Above expectation: raw-material fidelity

198 verbatim-grounded Italian-legal claims from 25 sources is genuinely useful. Every citation in my case memo traces cleanly to a span-in-source-in-manifest. Writing the memo by consulting those claims was faster than reading the original .txt files, and the discipline of citing narrowly (`[claim:148]` specifically for Corte Cost. 197/2023, not "per the corpus") forced better reasoning.

### Below expectation: structural synthesis from contradictions

I expected `contradiction-list --disposition supersede` to drive "what changed" in my case memo. In practice, only one of my 5 recorded contradictions (C4: femminicidio pre/post-2025) mattered to the case analysis, and I had to record it as a subject-matching proxy because the original cross-subject pair was refused.

I suspect this is a corpus-size effect — at ~200 claims across 25 sources, there just aren't many same-subject claim pairs in genuine tension. The tool probably earns its contradiction structure around 1000+ claims. For my size, the 8 draft concepts carried more analytical weight than the 5 contradictions did.

### Below expectation: structural lift from the query pipeline

Because `aleph ask` was LLM-gated and I had no key, I never got to exercise `_group_by_disposition` → `SYNTHESIZE_V2` → `VERIFY_SYSTEM`. That pipeline is where Aleph earns its keep over "SQL + citation discipline". My manual Phase H memo has citations and a verifier-flags footer, but:

- The grouping of contradictions by disposition (`replicate | distinguish | dispute | gap`) never got surfaced to a synthesizer.
- The per-sentence verifier I hand-rolled is strictly worse than `VERIFY_SYSTEM` would be. I verified by memory; the real verifier reads each cited span against each sentence.

So the memo is honest but "not a full Aleph artifact". It's a research memo assembled from Aleph claims, with manual stand-ins for the two pipeline stages that require the LLM.

## Structural observation

The agent-mode surface is **more of the product than a casual reading of the codebase suggests**.

Typical "optional LLM" tools collapse to static CRUD schemas without the key. Aleph's agent-mode CLI is substantively more: the honesty contracts (verbatim spans, draft-concept lifecycle, typed dispositions, same-subject requirement, disposition-rule pairing, cache invalidation on mutation) *are the product*, and they operate independent of the LLM. The LLM adds automation *over* the same honesty contracts — it does not create them. An agent or a team of humans driving the CLI gets the same guarantees as the API-mode pipeline.

Concretely, this means the cost of using Aleph without the key is (a) more manual reasoning work, (b) concepts stuck in `draft`, (c) no per-sentence verifier. It is *not* loss of grounding, citation integrity, disposition discipline, or provenance. That's an unusual property for a tool that advertises itself as LLM-centric, and it's worth highlighting more explicitly in the README.

## Would I use it for real work?

**Yes, with conditions.**

For research/analysis workflows where citation integrity matters — legal, medical, regulatory, financial-compliance — the verbatim-span discipline pays off immediately. I would put the case memo in front of a supervising attorney, flagged as "grounded in a 25-source Aleph corpus with 198 atomic claims, no independent verification against the original Gazzetta Ufficiale PDFs." I would *not* send it to a client.

For fast-moving, poorly-sourced, or thesaurus-heavy domains, the friction of verbatim-span discipline exceeds the benefit. A product-docs corpus that changes weekly would grind against the tool; a hundred-year-old statutory corpus fits it like a glove.

## Quick summary table

| Dimension | Verdict | Evidence |
|---|---|---|
| Correctness | Clean | 0 bugs in ~700 agent-mode calls |
| Honesty contracts | Enforced | Span invariant caught 3 real issues; rationale-concept refused drafts; disposition validators blocked 3 attempts |
| Ergonomics | Mixed | Envelope is great; flag inconsistency + string-parsed support syntax are friction |
| Localization | Weak | English-only normalize_subject; alias escape hatch exists but doesn't backfill |
| Agent-mode completeness | ~90% | Concepts are stuck in draft without LLM validator; no attested-promotion path |
| Diagnostics | Thin | stats-json is coarse; no source-yield report |
| Pipeline lift w/o key | Partial | Persistence layer works; synthesis+verifier layer doesn't |

See [improvement-proposals.md](improvement-proposals.md) for prioritized fixes.
