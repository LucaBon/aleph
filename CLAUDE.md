# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Aleph

A stateless-view knowledge base for LLMs. The core invariant: **atomic claims with verbatim source spans are the only compounding primitive; every prose "view" is ephemeral, regenerated on demand, and verified sentence-by-sentence against cited source spans.** Never treat generated prose as a source of truth, and never edit it in place — fix the underlying claims and let views regenerate.

## Commands

### Install (editable, for development)
```bash
pip install -e .
```
Installs the `aleph` console script (entry point `aleph.cli:main`).

### Running tests
The repo ships two end-to-end tests; there is no pytest suite.

```bash
# Full pipeline (ingest → ask → cache → lint → resolve → cascade-delete)
# against a deterministic FakeLLM. No API key.
python tests/e2e_fake_llm.py

# Agent-mode CLI test: spawns `aleph` subprocesses, exercises every
# agent-facing subcommand, no LLM involved.
python tests/e2e_agent_mode.py
```

Note: [tests/e2e_fake_llm.py:10](tests/e2e_fake_llm.py#L10) hardcodes `sys.path.insert(0, "/home/claude/aleph/src")` and [line 97](tests/e2e_fake_llm.py#L97) reads `/home/claude/aleph/examples/tesla_batteries.md`. Before running locally, swap those to this repo's absolute paths (or `pip install -e .` and delete the sys.path line).

### API-mode commands (require `ANTHROPIC_API_KEY`)
```bash
aleph ingest PATH [PATH ...]          # recurses; .txt/.md/.markdown only
aleph ask "question" [-k 30] [--no-verify] [--no-cache] [--json]
aleph lint [--resolve-by-recency]
aleph show [CLAIM_ID] [--limit 50]
aleph sources
aleph remove SOURCE_ID
aleph stats
aleph clear-cache
```
Global flags: `--db PATH` (default: `./aleph.db` if it exists, else `~/.aleph/aleph.db`), `--model MODEL` (default: `$ALEPH_MODEL` or `claude-opus-4-7`).

### Agent-mode commands (no API key; JSON stdout, one object per call)
Everything Claude Code drives. See [README.md](README.md) for the full list; all commands are registered in [src/aleph/agent_cli.py](src/aleph/agent_cli.py). Example:
```bash
aleph --db aleph.db claim-add --source-id 1 --subject "tesla battery" \
  --predicate "retains" --object "90% capacity at 200k mi" \
  --span "retain about 90% of their original capacity" --confidence 0.9
```

### Environment
- `ANTHROPIC_API_KEY` — required for `ingest`, `ask`, `lint` (the LLM-mode subcommands are gated in [cli.py:235](src/aleph/cli.py#L235)).
- `ALEPH_MODEL` — default model override ([llm.py:19](src/aleph/llm.py#L19)).

## Architecture

### Two execution modes share one store
- **Agent mode** (`source-add`, `claim-*`, `alias-*`, `contradiction-*`, `view-*`, `subjects`, `stats-json`) — no LLM in-process. Claude Code or another agent is the LLM; these commands just persist/retrieve. Handlers live in [src/aleph/agent_cli.py](src/aleph/agent_cli.py) and emit a single JSON object per invocation via `_emit`.
- **API mode** (`ingest`, `ask`, `lint`) — same code path calls the LLM itself via [src/aleph/llm.py](src/aleph/llm.py). The set of LLM-requiring commands is the literal `LLM_COMMANDS` constant at [cli.py:20](src/aleph/cli.py#L20).

Both modes read and write the same SQLite schema, so an agent session and a batch script can share a knowledge base.

### The compounding asymmetry
Left side of the pipeline is permanent and deterministic; right side is ephemeral and probabilistic. Do not blur this line:

- **Sources** ([db.py](src/aleph/db.py), table `sources`) — immutable, sha256-hashed, re-ingest is a no-op.
- **Claims** (`claims`) — `(subject, predicate, object, span_start, span_end, confidence)` tuples. The span is the canonical form; the triple is just the index card. `add_claim` normalizes subject + predicate through `resolve_subject` (alias-aware), so three users writing "Tesla Batteries" / "tesla battery" / "Tesla Battery" collapse to one subject. `subject_aliases` provides manual canonicalization for cases the normalizer can't catch — **aliases rewrite existing claims atomically and clear the view cache**.
- **Contradictions** (`contradictions`) — first-class nodes, not a post-hoc report. `lint` groups active claims by subject, proposes pairs with differing objects, and asks the LLM only whether each pair genuinely conflicts (not what the answer is). Resolutions either supersede the losing claim or leave both and mark the contradiction resolved.
- **Views** (`view_cache`) — generated prose answers keyed by sha256 of the normalized query. The cache is **invalidated automatically**: any `claim-add`, `claim-supersede`, `alias-add` that rewrites, `contradiction-resolve`, or `source-remove` calls `clear_cache()`. Never assume a cached view is still valid after any mutation.

### The grounding invariant
A claim that cannot be located as a verbatim substring of its source is refused at write time:
- [ingest.py:_locate_span](src/aleph/ingest.py#L80) drops ungrounded claims from LLM extraction.
- [agent_cli.py:cmd_claim_add](src/aleph/agent_cli.py#L81) returns an error to the agent instead of inserting.

This is why `claim-add --span` must be an **exact** substring. The span is located by `content.find(span)`; there is no fuzzy match at the agent boundary. If your agent-composed span doesn't appear verbatim in the source, the write fails — that's the design.

### The query pipeline ([src/aleph/query.py](src/aleph/query.py))
1. **Cache check** — sha256 of normalized query → `view_cache`. Hit returns immediately.
2. **Retrieve** — `Store.search_claims` is keyword-only (SQLite `LIKE`-based scoring on `subject||predicate||object`). Ranks by hit count, confidence, recency. To swap in BM25 or embeddings, rewrite this one method; the rest of the pipeline takes a ranked list of claim rows.
3. **Synthesize** — LLM call with `SYNTHESIZE_SYSTEM` prompt. Contract: every factual sentence must end with `[claim:ID]` citations referencing only the supplied IDs.
4. **Verify** — per-sentence loop: parse `[claim:ID]` from the first citation, fetch that claim's span via `get_span_text`, ask the LLM `GROUNDED | PARTIAL | UNGROUNDED`. Non-`GROUNDED` verdicts are appended as a "**Verifier flags**" footer by `_annotate_answer`. The verifier **never reads earlier generated prose** — only source spans.
5. **Cache** — final annotated answer + `claim_ids_used` are written to `view_cache`.

### LLM adapter boundary ([src/aleph/llm.py](src/aleph/llm.py))
`LLM.complete` and `LLM.complete_json` are the only two methods any other module calls. `complete_json` strips ```json fences and retries once with a "re-emit as strict JSON" nudge on parse failure. To swap Anthropic for another provider, rewrite this file — nothing else imports the SDK.

### Subject normalization ([src/aleph/db.py:normalize_subject](src/aleph/db.py#L22))
Deterministic: lowercase → collapse whitespace → strip non-`\w\s-` → singularize the last word with simple English rules (`-ies → -y`, `-sses → -ss`, trailing `-s` unless `-ss`/`-us`). This runs on every `add_claim` **and** on both sides of `resolve_subject` lookups, so aliases match post-normalization. English-only by design; non-English corpora should lean on the alias table.

### Storage layout
`Store` is the only class that knows SQLite ([src/aleph/db.py](src/aleph/db.py)). Schema is declared inline as the `SCHEMA` constant and applied in the constructor with `CREATE TABLE IF NOT EXISTS`, so opening an existing DB is a no-op migration. `PRAGMA foreign_keys = ON` is set; `ON DELETE CASCADE` on `claims.source_id` handles `source-remove` for claims, and `remove_source` additionally walks `view_cache` and deletes any cached view whose `claim_ids` JSON intersects the removed claims' IDs.

## When modifying this codebase
- Changing retrieval: rewrite `Store.search_claims` only. Don't push ranking decisions into `query()`.
- Changing the LLM provider: rewrite `src/aleph/llm.py` only. Preserve the `complete` / `complete_json` signatures.
- Changing the storage backend (e.g. Postgres): the target is `Store`; the schema ports over directly. Keep `clear_cache` / cascade semantics intact.
- Adding an agent command: add the handler to [agent_cli.py](src/aleph/agent_cli.py), register it in `register_agent_commands`, and add it to the set at [cli.py:241-245](src/aleph/cli.py#L241-L245) (agent commands receive `(args, store)`; API-mode ones receive `(args)`).
- Any mutation that could make a cached view stale must call `store.clear_cache()` in the same transaction path (see `add_claim`, `supersede_claim`, `add_alias`, `resolve_by_recency` for the existing pattern).
- The `--span` contract on `claim-add` is verbatim substring match. Don't add fuzzy-match fallbacks at the agent boundary — ungrounded claims must fail loudly, not silently drift.
