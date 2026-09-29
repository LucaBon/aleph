# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Aleph

A stateless-view knowledge base for LLMs. The core invariant: **atomic claims with verbatim source spans are the only compounding primitive; every prose "view" is ephemeral, regenerated on demand, and verified sentence-by-sentence against cited source spans.** Never treat generated prose as a source of truth, and never edit it in place — fix the underlying claims and let views regenerate.

Four first-class node types sit on top of claims, each with its own persistence and lifecycle: **concepts** (higher-order propositions grounded by claim sets), **conditions** (claim→claim links for scope/method/sample/limitation/assumption), **contradictions with dispositions** (typed conflicts resolved by an explicit vocabulary), and **source metadata** (per-domain authority/provenance).

## Commands

### Install (editable, for development)
```bash
pip install -e .
```
Installs the `aleph` console script (entry point `aleph.cli:main`).

### Running tests
Two end-to-end scripts plus a pytest property suite (`pip install -e .[dev]`).

```bash
# Full pipeline (ingest → ask → cache → lint → resolve → cascade-delete)
# against a deterministic FakeLLM. No API key.
python tests/e2e_fake_llm.py

# Agent-mode CLI test: spawns `aleph` subprocesses, exercises every
# agent-facing subcommand, no LLM involved.
python tests/e2e_agent_mode.py

# Property-based test of the retraction-cascade invariant (hypothesis) plus
# targeted leakage scenarios. HYPOTHESIS_PROFILE=thorough for 1000 examples.
pytest tests/test_invariants.py

# Canary leakage benchmark: inject a fabricated source, let it influence
# views/concepts/dispositions, retract it, regenerate, search for traces.
python benchmark/canary.py

# Everything above in one command (tests/test_scripts.py wraps the scripts).
pytest
```

Both e2e scripts import the installed `aleph` package, so run `pip install -e .[dev]` first; `e2e_agent_mode.py` needs the `aleph` console script on `PATH`.

### API-mode commands (require `ANTHROPIC_API_KEY`)
```bash
aleph ingest PATH [PATH ...] [--extract-conditions]  # recurses; .txt/.md/.markdown only
aleph ask "question" [-k 30] [--no-verify] [--no-cache] [--json] [--context JSON] [--verifier llm|layered]
aleph lint [--resolve-by-recency]
aleph show [CLAIM_ID] [--limit 50]
aleph sources
aleph remove SOURCE_ID
aleph stats
aleph clear-cache
```
Global flags: `--db PATH` (default: `./aleph.db` if it exists, else `~/.aleph/aleph.db`), `--model MODEL` (default: `$ALEPH_MODEL` or `claude-opus-4-7`).

- `ingest` reports per file: claims added, dropped (ungrounded) and flagged for fidelity review, LLM token usage, list-price cost, and cost per 1k source tokens (source tokens estimated as chars / 4; unknown model price ⇒ cost `None`, never a guess). Prices live in `PRICES_PER_MTOK` in [llm.py](src/aleph/llm.py).
- `--verifier layered` swaps ask's claim-citation check for the layered verifier (see "Claim fidelity and the layered verifier" below). Default stays `llm` until the verifier eval shows the layered one is better.
- `--extract-conditions` runs a pre-pass that extracts scope/method/sample/limitation/assumption claims from the whole document, then links every atomic claim from the same source to the whole scope set as `explicit=True` conditions. Opt-in (off by default).
- `--context` accepts a JSON object with optional keys `{jurisdiction, date, domain, include_retracted}`. Claims are filtered by source metadata before synthesis: retracted claims are dropped unless `include_retracted`; jurisdiction matches exact or dotted-prefix (`US-CA` keeps `US-CA-LA`); `date` (ISO or epoch) is compared against `effective_at`/`expires_at` on the source; `domain` matches `source_metadata.domain`. The cache key is `sha256(question || json(context, sort_keys=True))`, so different contexts cache independently.

### Agent-mode commands (no API key; JSON stdout, one object per call)
Everything Claude Code drives. All commands are registered in [src/aleph/agent_cli.py](src/aleph/agent_cli.py) via `register_agent_commands`. Every agent-mode call emits a single JSON envelope: `{"ok": true, "data": {...}}` on success, `{"ok": false, "error": {"code", "message", "details"}}` on failure.

Grouped by workstream:

- **Sources** — `source-add`, `source-replace` (atomic remove+re-add at a given path), `source-get`, `source-list`, `source-remove`, `source-yield` (per-source claim density for triage).
- **Claims** — `claim-add` (optional `--proposition`, `--conditions id:kind,id:kind,...`; the response carries `fidelity_issues` and `review_id`), `claim-get` (includes `proposition` and `context_text`), `claim-fidelity-check [ID | --all] [--enqueue]`, `claim-search` (FTS5 if available, else LIKE; `--compact` / `--fields` for compact output), `claim-by-subject` (same projection flags), `claim-supersede`, `subjects`.
- **Review queue (Phase 2)** — `review-list [--status open|accepted|rejected|obsolete|all] [--type claim|contradiction|concept|alias]`, `review-add --type T (--id N | --alias FROM)`, `review-resolve ID --decision accepted|rejected --by WHO` (records the decision only; a rejection's `next_step` names the command that actually fixes the item).
- **Aliases** — `alias-add` (refuses to overwrite an existing alias without `--force`), `alias-list`, `alias-log` (merge events), `alias-undo FROM` (restores the subjects a merge rewrote).
- **Contradictions (manual)** — `contradiction-add` (takes `CLAIM_A CLAIM_B` or `--claim-a/--claim-b`; `--cross-subject --relation-kind X --justification Y` is the P1.6 escape valve for doctrinal tensions across subjects), `contradiction-list [--disposition D] [--kind K] [--cross-subject-filter any|only|exclude] [--all]`, `contradiction-resolve`.
- **Contradictions (typed/disposition-aware, WS-B)** — `contradiction-scan` (LLM), `contradiction-dispose` (`--rationale-concept` accepts both `active` and `attested` concepts), `contradiction-get`, `contradiction-rule-get`.
- **Concepts (WS-A)** — `concept-add` (accepts legacy colon OR JSON `--support`), `concept-derive` (LLM), `concept-get`, `concept-list`, `concept-validate` (LLM), `concept-rebuild` (LLM), `concept-attest` (P1.2: agent-driven `draft → attested` with attestor trail), `concept-supersede`, `concept-invalidate`.
- **Source authority (WS-C)** — `source-authority-set` (metadata includes optional `fetch_method` + `provenance_notes`), `source-authority-get`, `source-list-by-domain`, `source-retract` (scientific only), `source-unretract`.
- **Claim conditions (WS-D)** — `claim-condition-add`, `claim-condition-remove`, `claim-conditions-list`, `claim-condition-extract` (LLM).
- **Diagnostics / agent-mode synthesis** — `invariant-check` (reports every view/concept/resolution/supersession resting on an inactive claim; see "The retraction cascade" below), `provenance CLAIM_ID` (one-shot claim→source→authority→conditions→concepts→contradictions walk; P2.4), `compose --query "…"` (retrieval + disposition brief without an LLM, the agent-mode counterpart to `ask`; P2.5), `report-json` (full diagnostic snapshot; P2.2).
- **Views** — `view-get`, `view-cache`, `cache-clear`.
- **Stats** — `stats-json`.
- **Store config** — `config-get [KEY]`, `config-set KEY VALUE` (currently `locale`).

LLM-calling agent commands (`concept-derive`, `concept-validate`, `concept-rebuild`, `contradiction-scan`, `claim-condition-extract`) need `ANTHROPIC_API_KEY` even though they're agent-mode; the subparser-time check in [cli.py:20](src/aleph/cli.py#L20) `LLM_COMMANDS` gates them alongside the API-mode commands. They can also be satisfied in an offline run by pointing `--mock-llm PATH` or `ALEPH_LLM_FIXTURES=PATH` at a JSON/YAML fixtures file (P2.1); the [`MockLLM`](src/aleph/llm.py) adapter matches fixtures by system-prompt / user-prompt substrings or sha256 hashes and returns canned responses, so the same code path runs in CI against a deterministic adapter.

### Agent-mode-first workflow (no API key)

The full Aleph runbook survives without a live LLM. The honesty contracts — verbatim spans, deterministic disposition vocabulary, cached view invalidation — don't depend on the synthesizer being present. The agent-mode counterpart of each LLM-gated workflow:

| API-mode / LLM step | Agent-mode counterpart |
|---|---|
| `aleph ingest` (API extracts claims) | Agent reads files, calls `claim-add` per claim; `source-yield` surfaces thin extractions |
| `aleph ask` (synthesizes + verifies) | `compose --query "…"` emits the retrieval/disposition brief; the agent writes the prose and cites IDs itself |
| `concept-validate` (LLM GROUNDED check) | `concept-attest` records an attestor + rationale; `contradiction-dispose --rationale-concept` accepts attested concepts alongside active ones |
| `concept-derive` / `concept-rebuild` | `concept-add --skip-validation` then `concept-attest` once the agent has audited the support spans |
| `contradiction-scan` (LLM pair judgement) | `report-json` lists same-subject candidate pairs; agent runs `contradiction-add` by hand |
| Provenance audit | `provenance CLAIM_ID` walks the full chain in one call |

What's lost without the LLM: no auto extraction, no per-sentence verifier, no automated contradiction scan. What's not lost: grounding (verbatim-span check), citation integrity, disposition discipline, cache invalidation, and — with `concept-attest` — a principled promotion path for concepts.

See the [aleph-case skill](skills/aleph-case/) for the full agent-mode runbook used on the Italian criminal-law corpus that inspired these proposals.

Example:
```bash
aleph --db aleph.db claim-add --source-id 1 --subject "tesla battery" \
  --predicate "retains" --object "90% capacity at 200k mi" \
  --span "retain about 90% of their original capacity" --confidence 0.9 \
  --conditions 12:sample,17:method
```

### Verifier eval (Phase 2)
```bash
python -m aleph.verifier_eval --pairs benchmark/verifier_eval/pairs.jsonl \
    --verifier deterministic|baseline|layered [--mock-llm F] [--out R.json]
```
`deterministic` needs no key; `baseline` (ask's default check) and `layered` call the LLM. The shipped pairs carry draft labels (`labeler: draft:claude`), so reports say `labels_status: draft` and their rates are not publishable; see [benchmark/verifier_eval/README.md](benchmark/verifier_eval/README.md).

### Environment
- `ANTHROPIC_API_KEY` — required for the `LLM_COMMANDS` set (API-mode `ingest`/`ask`/`lint` plus the agent-mode LLM-calling commands listed above). Gated in [cli.py:263](src/aleph/cli.py#L263).
- `ALEPH_MODEL` — default model override ([llm.py](src/aleph/llm.py)).
- `ALEPH_ENTAILER=nli` (optional `ALEPH_ENTAILER_MODEL`) — turns on the layered verifier's NLI entailment layer (needs the `embeddings` extra).

## Architecture

### Two execution modes share one store
- **Agent mode** — no LLM in-process (with the five listed exceptions). Claude Code or another agent is the reasoning engine; these commands persist/retrieve. Handlers live in [src/aleph/agent_cli.py](src/aleph/agent_cli.py) and emit a single JSON object per invocation via `_ok`/`_err`.
- **API mode** — the same code path calls the LLM itself via [src/aleph/llm.py](src/aleph/llm.py). The set of LLM-requiring commands is the literal `LLM_COMMANDS` constant at [cli.py:20](src/aleph/cli.py#L20).

Both modes read and write the same SQLite schema, so an agent session and a batch script can share a knowledge base.

### The compounding asymmetry
Left side of the pipeline is permanent and deterministic; right side is ephemeral and probabilistic. Do not blur this line:

- **Sources** ([db.py](src/aleph/db.py), table `sources`) — immutable, sha256-hashed, re-ingest is a no-op. `source-replace` is the atomic primitive for in-place updates.
- **Source metadata** (`source_metadata`) — typed per-domain JSON blob (`legal | scientific | policy | corporate | generic`). Fields are validated (not fabricated) in [authority.py](src/aleph/authority.py). Authority ranking is deterministic: legal uses `(authority_level, specificity, issued_at)`; scientific uses `(peer_reviewed, citation_count, published_at)`. Retraction is scientific-domain only; it's atomic over metadata + claims + concept staleness + view cache. Retraction state changes only through `source-retract` / `source-unretract` (`source-authority-set` refuses with `retraction_state_change`). `claims.retracted_cause` (`source` | `disposition`) lets unretract revive only what the source retraction took out, never a claim a `retracted` disposition dropped.
- **Claims** (`claims`) — `(subject, predicate, object, span_start, span_end, confidence, status)` tuples with `status ∈ {active, superseded, retracted}`. The span is the canonical form; the triple is just the index card. `add_claim` normalizes subject + predicate through `resolve_subject` (alias-aware), so three users writing "Tesla Batteries" / "tesla battery" / "Tesla Battery" collapse to one subject. `subject_aliases` provides manual canonicalization — aliases rewrite existing claims atomically, invalidate cached views, and **mark dependent concepts stale**.
- **Claim conditions** (`claim_conditions`, WS-D) — a claim→claim link describing WHEN the target claim applies. `kind ∈ {sample, method, scope, limitation, assumption}`. `explicit=True` means stated verbatim; `explicit=False` means LLM-inferred. Never silently flip inferred to explicit — this is the honesty contract. `conditions_overlap(a, b)` (Jaccard) feeds contradiction disposition: if overlap is 0 or None, `reconcile` is downgraded to `dispute` (Invariant 3 in [contradictions.py](src/aleph/contradictions.py)).
- **Concepts** (`concepts`, WS-A) — higher-order propositions grounded by a set of supporting claims. `inference_type ∈ {summary, generalization, synthesis}`. Lifecycle: `draft → active` after an LLM validator returns `GROUNDED` against the union of supporting spans, OR `draft → attested` via `concept-attest` (agent records an `attested_by` identifier + free-text `attestation_rationale` — the P1.2 escape hatch for fully-agent-mode runs); `active → stale` / `attested → stale` when any support claim is superseded/retracted/alias-rewritten (via `mark_concepts_stale_for_claims`); `stale → new concept` via `concept-rebuild`, which always creates a new row and supersedes the old (concepts are never mutated in place). v1 constraint: concepts cannot support other concepts — supports must be claims. `contradiction-dispose --rationale-concept` accepts both `active` and `attested`; `draft` concepts cannot be cited (the `concept-add` response surfaces a `"note"` when status stays at `draft` to make this explicit).
- **Contradictions** (`contradictions` + `contradiction_rules`) — first-class nodes with:
  - `kind`: `numeric | categorical | negation | temporal | normative | unknown` (typed pre-filter classifier in `classify_pair`).
  - `disposition`: `supersede | coexist | distinguish | reconcile | replicate | dispute | retracted | gap | unresolved` — the explicit vocabulary for resolution. Rules:
    - `coexist | distinguish | reconcile` require a `rule` string (and optionally `applies_when` JSON predicate and a `rationale_concept_id` pointing to an `active` or `attested` concept).
    - `supersede | retracted` require `keep`/`drop` claim IDs.
    - `replicate` bumps both claims' confidence by +0.05 (capped at 1.0).
  - `candidate_disposition` + `overlap_score` are written by the detector and preserved as the LLM's recommendation separate from the human/agent decision.
  - `cross_subject` + `relation_kind` + `justification` (P1.6) carry the same-subject-check escape valve. Set by `contradiction-add --cross-subject --relation-kind X --justification Y` where `X ∈ {regime-supersedes, rule-limits-rule, doctrinal-cross-ref}`. Cross-subject rows are surfaced distinctly in `contradiction-list` (via `--cross-subject-filter`) and in `contradiction-get`.
  - `list_contradictions_full` LEFT JOINs with `contradiction_rules` for the combined view.
- **Views** (`view_cache`) — generated prose answers keyed by `sha256(question || json(context, sort_keys=True))`. Both `claim_ids` and `concept_ids` are stored. Invalidation is automatic in all the places that mutate the claim/concept graph (`add_claim`, `supersede_claim`, `retract_source`, `add_alias`, `contradiction` dispose side-effects, `update_concept_status` to `superseded|invalidated`, `remove_source`). Cache helpers: `_invalidate_cache_for_claims` and `_invalidate_cache_for_concepts` in [db.py](src/aleph/db.py).

### The grounding invariant
A claim that cannot be located as a verbatim substring of its source is refused at write time:
- [ingest.py:_locate_span](src/aleph/ingest.py) drops ungrounded claims from LLM extraction. A whitespace-tolerant fallback recovers legitimate spans where the LLM collapsed whitespace, but never paraphrases.
- [agent_cli.py:cmd_claim_add](src/aleph/agent_cli.py) uses `content.find(span)` — there is no fuzzy match at the agent boundary. If your agent-composed span doesn't appear verbatim in the source, the write fails.

Concepts have a looser analog: the validator prompt requires every factual element of a concept's statement to be supported by the *union* of its support spans, returning `GROUNDED | PARTIAL | UNGROUNDED`. Only `GROUNDED` promotes a concept from `draft` to `active`.

### Claim fidelity and the layered verifier (Phase 2)
- Every claim stores an optional `proposition` (the claim as one sentence; the triple is its index) and a context window (`context_start`/`context_end`: enclosing sentence ± one, never across a paragraph break, ≤ 600 chars) computed at write time by `fidelity.context_window`.
- `Store.add_claim` runs `fidelity.check_fidelity` on the claim's proposition, else predicate + object (the subject is an index label, so it isn't checked), against span + context: numbers, dates, units, negation (added or dropped) and entities must appear there. Issues queue the claim in `review_queue` (reason `fidelity`) in the same transaction. **Flags never refuse a write** — the verbatim-span check stays the only hard gate.
- `review_queue` rows target exactly one claim/contradiction/concept (FK `ON DELETE CASCADE`) or alias (`alias_from`, with the merge's `canonical_to` recorded in `details`; overwriting or undoing the alias closes its reviews as `obsolete`, and `check_invariants` reports `open_review_without_target` otherwise). At most one open row per (target, reason) (unique partial index). Re-enqueueing a finding identical to an accepted/rejected one returns that decision instead of re-queuing. `resolve_review` only records `accepted | rejected`; don't make it mutate the target.
- `verifier.LayeredVerifier`: deterministic (verbatim restatement of the *whole* span ⇒ SUPPORTED, never a fragment; added number/date/unit/negation ⇒ UNSUPPORTED; entity flags and dropped negations are only hints) → optional entailer → LLM (`SUPPORTED | UNSUPPORTED | UNCERTAIN`; a failed or off-vocabulary call is UNCERTAIN, never UNSUPPORTED). In `ask` it maps SUPPORTED→GROUNDED, UNSUPPORTED→UNGROUNDED, UNCERTAIN→UNCERTAIN (ranked between PARTIAL and UNGROUNDED); a multi-citation sentence is checked against each cited span plus the other cited spans; a layered `ask` bypasses the view cache. `verifier.baseline_verdict` wraps `query._verify_span` — the exact check ask uses by default — so the eval compares like with like.

### The query pipeline ([src/aleph/query.py](src/aleph/query.py))
The pipeline is always `SYNTHESIZE_V2` — v1 prompts are kept in the file for reference only.

1. **Cache check** — hash is `sha256(question.lower().strip() || json(context, sort_keys=True))`. Hit returns immediately, including both `claim_ids` and `concept_ids`.
2. **Retrieve claims** — `Retriever.search(keywords, limit)` from [retrieval.py](src/aleph/retrieval.py). Defaults to `FTSRetriever` (SQLite FTS5 + BM25) when the store has `claims_fts` available, else `KeywordRetriever` (LIKE-based scoring). Plug in a different backend by passing an explicit `retriever=` — the rest of the pipeline is agnostic.
3. **Context filter** — `_filter_claims_by_context` drops retracted (unless `include_retracted`) and filters by jurisdiction/date/domain using `source_metadata`. Drops are logged.
4. **Retrieve concepts** — keyword match on `subject` OR `statement`, filtered to `status='active'`, ordered by confidence then `last_validated_at`. Default cap is `max(5, retrieve_k // 3)`.
5. **Disposition grouping** — `_group_by_disposition` looks up non-`unresolved` contradictions involving retrieved claims and bundles them into `replications | reconciled | coexisting | distinguished | disputed | gaps`. This block is shown to the synthesizer so it can present conflicts honestly instead of picking a side.
6. **Synthesize** — `SYNTHESIZE_V2_SYSTEM` + template. Each factual sentence ends with `[claim:ID]` and/or `[concept:ID]` citations. The prompt has per-disposition handling instructions (e.g. `replicate` ⇒ one sentence citing all member IDs; `gap` ⇒ use the exact "Flagged for expert review" phrasing).
7. **Verify** — per-sentence, per-reference. `[claim:ID]` goes through `VERIFY_SYSTEM` against the claim's span; `[concept:ID]` goes through `VERIFY_CONCEPT_SYSTEM` against the *union* of the concept's support spans. Verdicts are `GROUNDED | PARTIAL | UNGROUNDED | ERROR` (ERROR is a verifier-itself failure, distinct from UNGROUNDED). Worst verdict wins at the sentence level; `_annotate_answer` lists every non-GROUNDED ref individually in the "**Verifier flags**" footer.
8. **Cache** — write `{claim_ids, concept_ids}` intersected with what was actually retrieved (hallucinated IDs never enter the cache index).

### LLM adapter boundary ([src/aleph/llm.py](src/aleph/llm.py))
`LLM.complete` and `LLM.complete_json` are the only two methods any other module calls. `complete_json` strips ```json fences and retries once with a "re-emit as strict JSON" nudge on parse failure. To swap Anthropic for another provider, rewrite this file — nothing else imports the SDK.

### Subject normalization ([src/aleph/normalization.py](src/aleph/normalization.py))
Deterministic and locale-aware. The `Normalizer` protocol exposes a single `normalize(s: str) -> str` method; two implementations ship:

- **`EnglishNormalizer`** (locale `en`, default): lowercase → collapse whitespace → strip non-`\w\s-` → singularize the last word with simple rules (`-ies → -y`, `-sses → -ss`, trailing `-s` unless `-ss`/`-us`).
- **`ItalianNormalizer`** (locale `it`, P1.3): adds Italian suffix rules (`-zioni → -zione`, `-nti → -nte`, `-nze → -nza`, `-che → -ca`, fallback `-i → -o`); preserves invariants ending in a stressed vowel (`-ità`, `-tù`, …).

Every `add_claim` and both sides of `resolve_subject` dispatch through `Store.normalize_subject`, which calls the store's configured normalizer. Pass `--locale it` the first time you open a store (or `config-set locale it` later) to switch. Existing aliases were normalized under the old locale's rules, so switching mid-corpus is a deliberate migration — `Store.__init__` refuses to switch implicitly.

### Retrieval pluggability ([src/aleph/retrieval.py](src/aleph/retrieval.py))
`query.query()` calls the `Retriever` protocol, not `Store.search_claims` directly. Three implementations ship: `KeywordRetriever` (LIKE), `FTSRetriever` (FTS5 + BM25), `EmbeddingRetriever` (sentence-transformers + cached `claim_embeddings` table; optional dep). `default_retriever(store)` honours `ALEPH_RETRIEVER=keyword|fts|embedding` and otherwise picks FTS when available. Plug in a new backend by implementing the single-method protocol and passing `retriever=` to `query()` — the rest of the pipeline is agnostic.

### Storage layout
`Store` is the only class that knows SQLite ([src/aleph/db.py](src/aleph/db.py)). The `SCHEMA` constant is applied with `CREATE TABLE IF NOT EXISTS`, plus `_run_alters` for idempotent column additions (each `ALTER` swallows SQLite's "duplicate column" error). `_init_fts` sets up the `claims_fts` virtual table, triggers, and a backfill; if FTS5 isn't available (ancient/stripped SQLite build) the store silently falls back to LIKE search. `PRAGMA foreign_keys = ON`, `journal_mode = WAL`, `busy_timeout = 5000` — the minimum viable multi-writer story for SQLite.

Tables:
- `sources`, `claims`, `subject_aliases`, `contradictions`, `view_cache` — the original core.
- `concepts`, `concept_supports` — WS-A.
- `claim_conditions` — WS-D.
- `source_metadata` — WS-C.
- `contradiction_rules` — WS-B (rule + applies_when + rationale_concept_id).
- `claims_fts` — FTS5 virtual table with triggers mirroring INSERT/DELETE/UPDATE on `claims`. `_init_fts` rebuilds it when its docsize count differs from `claims` (stores created before FTS).
- `review_queue` — Phase 2 (schema migration 2).

Schema version 2 (`MIGRATIONS` in db.py; each migration runs inside an explicit `BEGIN`, since sqlite3 would otherwise autocommit DDL) adds `claims.{proposition, context_start, context_end}` (context backfilled with the frozen `fidelity._context_window_v2` — never edit it; change `context_window` instead — proposition left NULL) and `review_queue`. New stores run it too.

Additional columns added via `_run_alters`:
- `contradictions.{kind, disposition, disposition_at, candidate_disposition, overlap_score}`
- `view_cache.concept_ids` (JSON list, default `'[]'`)

### The retraction cascade
Every path that takes claims out of the active set (`supersede_claim`, `retract_source`, `remove_source`, the `retracted` disposition) calls `Store._on_claims_deactivated_tx`, which:
1. drops cached views citing the claims;
2. marks dependent concepts stale, which also drops views citing those concepts and reopens contradictions using them as rationale;
3. revives claims whose supersession chain no longer ends in an active claim (`_revive_orphaned_supersessions_tx`);
4. reopens resolved contradictions whose resolution rested on the claims.

A reopen sets `disposition='unresolved'`, records `reopened_from` and `reopened_cause`, and reverts replicate confidence bumps (stored per claim in `confidence_delta_a` and `confidence_delta_b`). For keep/drop resolutions, only the kept claim (`resolved_to`) leaving triggers a reopen. `check_invariants` in `db.py` checks exactly the states this cascade rules out; keep the two in step, and extend `tests/test_invariants.py` when adding a mutation.

Alias merges are logged in `alias_events` and `alias_rewrites` (one row per rewritten claim, with its old subject), so `undo_alias` can restore them.

## When modifying this codebase
- **Changing retrieval**: prefer a new `Retriever` subclass in [retrieval.py](src/aleph/retrieval.py) and a `default_retriever` tweak. `Store.search_claims` and `Store.search_claims_fts` are the existing backends; don't push ranking decisions into `query()`.
- **Changing the LLM provider**: rewrite `src/aleph/llm.py` only. Preserve `complete`/`complete_json`.
- **Changing the storage backend** (e.g. Postgres): the target is `Store`; the schema ports over. Keep `clear_cache` / cascade semantics and the two `_invalidate_cache_for_*` helpers intact.
- **Adding an agent command**: add the handler to [agent_cli.py](src/aleph/agent_cli.py), register it inside `register_agent_commands` (it returns the set of registered command names, so the dispatcher in [cli.py](src/aleph/cli.py) picks it up without a hand-maintained list). Agent commands take `(args, store)`; API-mode ones take `(args)`. If the new command needs an LLM, add its name to the `LLM_COMMANDS` set at [cli.py:20](src/aleph/cli.py#L20).
- **Any mutation that could make a cached view stale** must invalidate in the same transaction. For claims leaving the active set, call `self._on_claims_deactivated_tx(cx, claim_ids, cause=...)`. For other claim mutations (e.g. subject rewrites), use `_invalidate_cache_for_claims(cx, claim_ids, cause=...)` and `_mark_concepts_stale_for_claims_tx(cx, claim_ids, cause=...)` together; for concept mutations use `_invalidate_cache_for_concepts`. The existing patterns live in `add_claim`, `supersede_claim`, `add_alias`, `remove_source`, `retract_source`, `update_concept_status`, and `contradictions.dispose`.
- **The `--span` contract on `claim-add` is verbatim substring match**. Don't add fuzzy-match fallbacks at the agent boundary — ungrounded claims must fail loudly, not silently drift. The ingest path has a whitespace-tolerant fallback but never paraphrases.
- **Never silently flip `explicit` on a claim_condition** from False to True. Inferred conditions stay inferred until a human/agent upgrades them.
- **Never silently override a disposition**. `coexist | distinguish | reconcile` require a `rule`; `supersede | retracted` require `keep/drop`. The validators in `contradictions.dispose` are the authoritative gate.
- **Concepts are never mutated in place** once active. Use `concept-rebuild` (creates a new row and supersedes the old) or `concept-invalidate`. `update_concept_status` to `superseded | invalidated` invalidates cached views citing the concept.
