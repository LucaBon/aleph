# Aleph

**Every answer can be checked quickly and fixed cleanly.**

Aleph is a knowledge base built around auditability and reversibility. Each answer sentence cites atomic claims, and each claim points to an exact, verbatim span of an immutable source. When a claim or source turns out to be wrong, you retract it: every cached answer, concept and contradiction resolution that depended on it is invalidated, and answers regenerate without it.

What we can claim today, and what we can't yet, is tracked in [docs/roadmap.md](docs/roadmap.md). In short, the retraction cascade is deterministic code that we are putting under invariant and canary-leakage tests. The verifier's accuracy has **not** been measured against human labels yet, so treat its verdicts as a triage signal, not a guarantee.

This is an implementation of the "Claim Layer" — a response to Karpathy's [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern that keeps the compounding benefit while eliminating the main failure modes (hallucination propagation, lost provenance, silent drift, deletion pain).

## The idea in 30 seconds

You built a RAG pipeline. It hallucinates. When an answer is wrong, you can't tell which sentence is wrong, or where it came from, or how to fix it. When a source turns out to be wrong, you can't cleanly remove its influence — it's already been mixed into cached prose, summaries, and downstream indexes.

Aleph sits one layer deeper. The primitives, strictly separated:

- **Sources** — immutable, hashed, trusted. The source of truth. Optional typed metadata per domain (`legal | scientific | policy | corporate | generic`) drives authority ranking, retraction, jurisdiction/date filtering.
- **Claims** — atomic `(subject, predicate, object, span, confidence)` tuples, each pointing back to an exact substring of a source. Index cards, not summaries.
- **Conditions** — claim-to-claim links that say WHEN a claim applies (sample, method, scope, limitation, assumption). Explicit vs inferred is tracked — never silently flipped.
- **Concepts** — higher-order propositions (summary, generalization, synthesis) grounded by a set of supporting claims. Persistent, cite-able, validated against the union of supporting spans before going active; auto-marked stale when their supports change.
- **Contradictions with dispositions** — typed first-class nodes (`numeric | categorical | negation | temporal | normative`) resolved by an explicit vocabulary (`supersede | coexist | distinguish | reconcile | replicate | dispute | retracted | gap | unresolved`). Rules and rationale concepts travel with the disposition.
- **Views** — human-readable answers to questions. Generated on demand from claims *and* concepts, cited inline, verified sentence-by-sentence against the source spans they cite (for claim cites) or the union of supporting spans (for concept cites). Never authoritative. Never edited.

Nothing synthetic is ever the source of truth. Any wrong claim can be removed and every downstream view — and every concept that depended on it — is marked stale and regenerates clean.

## Smoke test

> **Smoke test, not an evaluation.** These numbers come from a deterministic *fake* LLM on a 3-source toy corpus. They show that the pipeline's mechanics work end to end; they say nothing about accuracy on real data. The real evaluation plan is in [docs/roadmap.md](docs/roadmap.md).

The numbers below come from [`benchmark/run.py`](benchmark/run.py) against a 3-source corpus under [`benchmark/corpus/`](benchmark/corpus/) with four seeded cross-source contradictions. The fake LLM is deterministic; running the script reproduces the numbers exactly. A live version using the Anthropic API is at [`benchmark/run_live.py`](benchmark/run_live.py).

<!-- BENCHMARK:START -->
| Metric | With Aleph | Without Aleph |
|---|---|---|
| Sentences with source-verified citations | **83%** (5/6) | 0% (0/3) |
| Seeded contradictions caught by `lint` | **4/4** (100% recall, 100% precision) | n/a |
| Downstream views invalidated on `source-remove` | **1/3** in 5.2 ms | stale prose persists |
| LLM calls on cache hit | **0** (cold: 3) | n/a |
| Surface subject forms → canonical | **12 → 7** | n/a |
<!-- BENCHMARK:END -->

The one failing sentence is the verifier flagging a sentence that mentions "2022 policy" while the cited claim's span only contains the retention-window fact: a paraphrase not supported by the span it cites, even though the date appears elsewhere in the same source.

To regenerate these numbers, run:

```bash
python benchmark/run.py                 # offline, deterministic, no API key
python benchmark/run.py --check-readme  # fails CI if README numbers have drifted
ANTHROPIC_API_KEY=... python benchmark/run_live.py  # against a real model
```

## Who this is for

- **Research teams and analysts** — you maintain a body of findings drawn from many papers, notes, and interviews. You need to be able to delete a paper cleanly when a preprint gets retracted, and to answer "what do my sources say about X?" with an inline-cited summary you can audit back to the original text.
- **Regulated or audit-sensitive orgs** (legal, medical, financial, compliance) — you need answers that carry provenance by construction, not retrofitted via "click to view source". You need to be able to prove, after the fact, that no statement in an answer was fabricated beyond what a specific span of a specific source supports.
- **Developers building agent workflows** — you have an LLM loop that needs durable memory without the drift failure mode of summarization. Aleph gives you a write-API your agent can call (`source-add`, `claim-add`, `claim-search`) with a hard grounding invariant enforced at the boundary.

## Use cases

**Literature review under changing evidence.** Ingest 20 papers on a topic; ask "what's the consensus on X?"; get a paragraph with sentence-level citations. When a paper is retracted, `aleph remove <id>` cascades — every cached view that cited it is invalidated, every downstream answer regenerates without the retracted claim bleeding through.

**Policy Q&A inside a regulated org.** Ingest the current policy set. When policies update, ingest the new version too; `aleph lint` flags the conflicting clauses. A resolution workflow (`--resolve-by-recency` or manual `contradiction-resolve`) marks the old clause as superseded. Recency means the policy's own `effective_at` date (set with `source-authority-set`), not when it was ingested; undated pairs stay open. Answers to "what's our data retention policy?" always cite the current source — with the exact span quoted inline.

**Durable agent memory with no drift.** Your agent loop extracts claims as it reads material, via the `claim-add` JSON command. Each claim enters the store only if its span is a verbatim substring of the source — ungrounded claims are refused at the write boundary. The agent can later retrieve claims by subject or keyword and compose answers without ever paraphrasing its own prior prose.

## How it works

```
  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐
  │   sources    │──▶│    claims    │◀─▶│ contradictions           │
  │  (immutable, │   │ (atomic,     │   │  (typed: numeric, etc.;  │
  │   hashed;    │   │  pointers to │   │   disposition: supersede,│
  │   domain     │   │  source      │   │   coexist, reconcile,    │
  │   metadata)  │   │  spans)      │   │   distinguish, ...)      │
  └──────────────┘   └──┬──────┬────┘   └──────────┬───────────────┘
                        │      │                   │
                        │      │ support           │ rule /
                        │      ▼                   │ rationale
                        │  ┌──────────┐            │
                        │  │ concepts │◀───────────┘
                        │  │ (summary,│
                        │  │  gen.,   │
                        │  │  synth.) │
                        │  └────┬─────┘
                        │       │
                   ┌────┴───────┴──┐
                   │  conditions   │  (sample/method/scope/
                   │  (claim→claim)│   limitation/assumption;
                   │               │   explicit vs inferred)
                   └────┬──────────┘
                        │
                        │  retrieve (FTS5/keyword) + context filter
                        ▼                     (jurisdiction/date/domain)
                 ┌──────────────┐         ┌──────────────────────┐
                 │  synthesize  │────────▶│      verifier        │
                 │  view (v2:   │         │ per sentence & per   │
                 │  claims +    │         │ cite: claim → span;  │
                 │  concepts +  │         │ concept → union of   │
                 │ dispositions)│         │ supporting spans     │
                 └──────┬───────┘         └──────────┬───────────┘
                        │                            │
                        ▼                            ▼
                 ┌──────────────────────────────────────────────┐
                 │               view (ephemeral)               │
                 │  prose + [claim:N] / [concept:N] + flags     │
                 └──────────────────────────────────────────────┘
```

The key asymmetry: **the left side is permanent and deterministic; the right side is ephemeral and probabilistic.** Scoping is a graph/FTS5 query with context filtering. Synthesis is an LLM call (or an agent turn) that receives pre-computed disposition groupings so it can't silently pick a side of a conflict. Verification is an LLM/agent check against specific source spans (or, for concepts, the union of supporting spans), never against earlier generated prose.

## Two modes

### Mode 1: Claude Code (recommended)

No API key. The agent — Claude Code or any comparable coding agent — *is* the LLM. Aleph becomes a set of deterministic CLI commands the agent drives. The agent reads files, decides on claims, persists them; later reads claims and composes cited answers.

```bash
pip install aleph-kb          # no ANTHROPIC_API_KEY needed for agent mode
cp -r skills/aleph ~/.claude/skills/   # or wherever your agent looks for skills
```

Then inside Claude Code:

```
> ingest ~/papers/*.pdf into a knowledge base
> what does my research say about attention head pruning?
> find any contradictions between the papers I've added
```

Claude Code reads `SKILL.md`, drives the CLI, and returns cited answers. The skill file documents three workflows (ingest, ask, lint) that the agent follows.

#### Agent-mode-first quickstart (no API key, nothing gated)

The full Aleph runbook runs without a live LLM. The honesty contracts — verbatim spans, deterministic disposition vocabulary, cached view invalidation — don't depend on the synthesizer being present. A typical flow:

```bash
# 1. Register sources. The agent reads each file, picks out claims,
#    records them with verbatim spans. source-yield flags sources with
#    suspiciously thin extraction (claims/KB) for a second pass.
aleph source-add  path/to/doc.md
aleph claim-add   --source-id 1 --subject "…" --predicate "…" \
                  --object "…" --span "…"      # span must be verbatim
aleph source-yield --thin-threshold 1.0

# 2. Tag sources with provenance + authority so downstream readers can
#    audit the chain. fetch_method and provenance_notes are optional but
#    recommended for anything that isn't a first-party original.
aleph source-authority-set 1 --domain legal --metadata '{
  "jurisdiction":"IT", "authority_type":"statute",
  "authority_level":5, "specificity":2,
  "fetch_method":"distilled",
  "provenance_notes":"WebFetch rendering of HTML"
}'

# 3. Build concepts. The agent audits each proposed concept against the
#    union of its support spans and attests it — the agent-mode
#    counterpart to concept-validate (LLM).
aleph concept-add    --subject "…" --statement "…" \
                     --support '[[12,"premise"],[17,"corroborating"]]' \
                     --confidence 0.8 --skip-validation
aleph concept-attest 1 --attested-by "agent-2026-04" \
                       --rationale "both spans stated verbatim"

# 4. Record contradictions + dispositions explicitly. attested concepts
#    are accepted as --rationale-concept alongside active ones. The
#    cross-subject escape valve covers legitimate doctrinal tensions.
aleph contradiction-add 3 4        # same subject, different objects
aleph contradiction-dispose 1 --disposition coexist \
                              --rule "jurisdiction-scoped" \
                              --rationale-concept 1
aleph contradiction-add  --cross-subject \
                         --relation-kind regime-supersedes \
                         --justification "Constitutional ruling 197/2023 …" \
                         7 11

# 5. Compose answers. `compose` is the agent-mode counterpart to `ask`:
#    it runs retrieval + context filter + disposition grouping and emits
#    the structured evidence brief, and the agent writes the prose itself.
aleph compose --query "how does Italian law treat femicide?" -k 20 \
              --context '{"jurisdiction":"IT"}'
aleph provenance 148                  # full chain for one claim
aleph report --format markdown        # one-shot diagnostic snapshot
```

For CI, smoke tests, or offline work against the full LLM-gated pipeline (ingest, ask, contradiction-scan, concept-validate, …), set `ALEPH_LLM_FIXTURES=path/to/fixtures.json` or pass `--mock-llm path/to/fixtures.json`. The [`MockLLM`](src/aleph/llm.py) adapter reads canned responses keyed by system/user prompt substrings or sha256 hashes.

### Mode 2: Direct API

Set `ANTHROPIC_API_KEY` and the original three high-level commands (`aleph ingest`, `aleph ask`, `aleph lint`) will call the API internally. Useful for batch processing or scripting without an agent in the loop.

```bash
export ANTHROPIC_API_KEY=sk-...
aleph ingest ~/my-notes/
aleph ask "how long do Tesla batteries actually last?"
aleph lint --resolve-by-recency
```

## Verification

No API key needed:

```bash
python tests/e2e_fake_llm.py
python tests/e2e_agent_mode.py
python benchmark/run.py
```

The first two run the existing end-to-end pipeline and CLI tests. The third runs the benchmark that produces the numbers in the [smoke test](#smoke-test).

The verifier eval ([benchmark/verifier_eval/](benchmark/verifier_eval/)) measures false-accept and false-reject rates on labeled sentence/span pairs. Its current pairs are draft labels awaiting human review, so it produces no publishable rates yet.

## Commands

**Agent-mode** (no API key, JSON in/out — what Claude Code calls). A handful are LLM-calling and do need `ANTHROPIC_API_KEY`; those are marked `(LLM)`.

```
# sources
source-add PATH                  register a source
source-replace PATH              atomic remove+re-add when the file changed on disk
source-get ID                    fetch source content
source-list                      list sources
source-remove ID                 cascade-delete
source-yield [--thin-threshold F] per-source claims/KB diagnostic (triage)

# claims
claim-add --source-id N --subject S --predicate P --object O --span TEXT \
          --confidence C [--proposition SENTENCE] [--conditions id:kind,id:kind,...]
                                 fidelity issues are reported and queued for review
claim-get ID                     claim + its source span, proposition and context window
claim-fidelity-check [ID | --all] [--enqueue]
                                 numbers/dates/units/negations/entities vs span + context
claim-search QUERY [-k N] [--compact|--fields id,predicate,...]   FTS5/LIKE
claim-by-subject SUBJECT [--compact|--fields ...]                 alias-resolved
claim-supersede OLD NEW          mark OLD as superseded
subjects                         distinct subjects by count

# answer checks
counter-evidence --claim-ids 1,2    uncited sides of live conflicts with the claims an answer cites

# review queue (claims, contradictions, concepts, aliases)
review-list [--status open|accepted|rejected|obsolete|all] [--type T]
review-add --type T (--id N | --alias FROM) [--reason R] [--note TEXT]
review-resolve ID --decision accepted|rejected --by WHO [--note TEXT]
                                 records the decision; fixing a rejected item is a separate step

# aliases + store config
alias-add FROM TO                declare FROM = TO; rewrites existing claims
alias-list                       show aliases
config-get [KEY]                 read store config (currently: locale)
config-set KEY VALUE             write store config (locale = en | it)

# contradictions — manual
contradiction-add A B            record contradiction (same subject)
contradiction-add --cross-subject --relation-kind regime-supersedes|rule-limits-rule|doctrinal-cross-ref \
                  --justification TEXT  A B                      cross-subject escape valve
contradiction-list [--all] [--disposition D] [--kind K] [--cross-subject-filter any|only|exclude]
contradiction-resolve ID --keep K [--drop D]

# contradictions — typed / disposition-aware (WS-B)
contradiction-scan [--subject S] [--kind K] [--since T]        (LLM)
contradiction-dispose ID --disposition D [--rule ...] [--applies-when JSON]
                         [--rationale-concept N] [--keep K] [--drop D]
contradiction-get ID             full contradiction + rule + spans + cross-subject fields
contradiction-rule-get ID        the resolution rule attached to a disposition

# concepts (WS-A)
concept-add --subject S --statement TEXT --support id:role,...    (legacy colon or JSON)
            [--inference-type summary|generalization|synthesis]
            [--confidence C] [--skip-validation]                (LLM unless --skip-validation)
concept-derive --subject S [--claims IDS] [--limit N]           (LLM)
concept-attest ID --attested-by HANDLE --rationale TEXT          draft -> attested (agent-mode)
concept-get ID [--with-spans]
concept-list [--subject S] [--status draft|attested|active|stale|superseded|invalidated]
concept-validate ID                                             (LLM)
concept-rebuild ID                                              (LLM)
concept-supersede OLD NEW
concept-invalidate ID --reason TEXT

# source authority / retraction (WS-C)
source-authority-set ID --domain legal|scientific|policy|corporate|generic \
                         --metadata JSON [--strict]             fetch_method + provenance_notes supported
source-authority-get ID                                          surfaces provenance_warnings
source-list-by-domain DOMAIN
source-retract ID --reason TEXT         (scientific domain only; cascades)
source-unretract ID --reason TEXT

# claim conditions (WS-D)
claim-condition-add --claim N --condition M --kind sample|method|scope|limitation|assumption
                    [--explicit|--no-explicit] [--confidence C]
claim-condition-remove --claim N --condition M
claim-conditions-list CLAIM_ID
claim-condition-extract CLAIM_ID [--same-source-only|--no-same-source-only]  (LLM)

# agent-mode synthesis / audit
provenance CLAIM_ID              one-shot claim -> source -> authority -> conditions -> concepts walk
compose --query "Q" [-k N]       retrieval + dispositions brief; the agent composes prose itself
          [--concept-k N] [--context JSON]
report-json                      full diagnostic snapshot (JSON)

# views + stats
view-get QUERY                   cache lookup
view-cache QUERY RESPONSE --claim-ids CSV
cache-clear
stats-json
```

**API-mode** (requires `ANTHROPIC_API_KEY` *or* `--mock-llm PATH`):

```
aleph ingest PATH [PATH ...] [--extract-conditions]
aleph ask "question" [-k N] [--no-verify] [--no-cache] [--json]
                     [--context '{"jurisdiction":"US-CA","date":"2026-04-22"}']
aleph lint [--resolve-by-recency]
aleph show [ID]                  show a claim or list them
aleph sources                    list sources (human-readable)
aleph remove SOURCE_ID           remove + cascade
aleph stats                      counts
aleph report [--format text|markdown|json]   one-shot diagnostic snapshot
aleph clear-cache
```

`--extract-conditions` runs a pre-pass that pulls scope/method/sample/limitation/assumption claims from the whole document and links every atomic claim from that source to them. `--context` on `ask` filters the retrieved claim set by `source_metadata` (jurisdiction prefix-match, date vs `effective_at`/`expires_at`, domain, `include_retracted`). Different contexts cache independently.

Global flags: `--db PATH`, `--model MODEL`, `--locale en|it`, `--mock-llm FIXTURES_PATH` (or set `ALEPH_LLM_FIXTURES=path`). `ALEPH_RETRIEVER=keyword|fts|embedding` selects the retriever backend (embeddings require `pip install sentence-transformers`).

## But isn't this just…?

The three most common objections, one sentence each. Full treatment in [docs/FAQ.md](docs/FAQ.md).

- **"…just RAG with extra steps?"** → RAG re-derives per query; Aleph has a persistent, inspectable, deletable claim graph. The difference shows up the moment a source turns out to be wrong (see: deletion-propagation row in the [smoke test](#smoke-test)).
- **"…just a database?"** → Aleph IS a database with one invariant enforced at the write boundary: every claim's span must be a verbatim substring of its source. Strip that invariant and you have a database. Keep it and you have provenance-by-construction.
- **"…the verbatim-span requirement is too strict for real text."** → That strictness is the feature. A fuzzy-match fallback at the write boundary defeats the guarantee that every claim maps back to specific source characters — which is what makes the verifier meaningful.

Four more objections (claim extraction itself hallucinates, keyword retrieval won't scale, ingest is too expensive, why not embeddings) are answered in the FAQ.

## How this addresses each LLM Wiki limitation

| LLM Wiki problem | Aleph's fix |
|---|---|
| Hallucinations get baked into pages and compound | Pages (views) are ephemeral. Every read regenerates from current claims and concepts. Nothing compounds because nothing is permanent. |
| Lossy summarization — edge cases vanish | Claims are atomic. Each edge case is its own claim. Scope/method/limitations are first-class conditions linked to the claims they qualify, not buried in prose. |
| Provenance collapses | Every sentence cites specific claim IDs and/or concept IDs; every claim points to an exact span; every concept is validated against the union of its support spans. One hop back to the original. |
| `index.md` breaks at ~200–500 pages | Retrieval is FTS5 (or LIKE) over indexed claim fields. No "read index into context" step. |
| Retrieval-as-reasoning is silently lossy | Scoping is keyword/FTS5/graph + context filter (jurisdiction/date/domain), all deterministic and auditable. The agent only reasons over pre-filtered claims. |
| No deletion, contradiction, supersession | `source-remove` cascades. `source-retract` is atomic over claims + concepts + view cache. Contradictions carry an explicit disposition (`supersede`, `coexist`, `reconcile`, …) with rules and rationale concepts. |
| Subject synonyms silently split knowledge | Subjects are normalized at write time (case, whitespace, simple plurals). For the hard cases, `alias-add` rewrites all claims atomically and marks dependent concepts stale. |
| Generalizations drift away from the evidence | Concepts must be `GROUNDED` against the union of their support spans before going active. Any support change demotes them to `stale` and forces a rebuild. |
| Apparent conflicts get "resolved" by picking a side | Dispositions are explicit: `coexist`, `distinguish`, `reconcile` require a rule; `supersede`/`retracted` require `keep`/`drop`; `gap` surfaces the conflict with the exact phrasing "Flagged for expert review." No silent winners. |
| One source being authoritative for every question | Source metadata carries `authority_level`, `jurisdiction`, `effective_at`/`expires_at`, `peer_reviewed`, `retracted`, `citation_count` — ranking is domain-specific, never fabricated. |

## When to use this vs. LLM Wiki vs. RAG

- **LLM Wiki** (Karpathy's pattern) — best for personal research where you want to *browse* an evolving vault in Obsidian. The page-as-artifact is the point.
- **RAG** — best at massive scale (tens of thousands of documents, frequently updated) where in-context knowledge isn't possible.
- **Aleph** — best when correctness and auditability matter: research teams, anything regulatory or medical, enterprise KBs, any case where a source can turn out to be wrong and you need to pull its influence cleanly. Also good as a substrate under a chat UI where users want citations that actually check out.

## Extending it

- **Different LLM provider (API mode)** — rewrite `src/aleph/llm.py`. Keep `complete` and `complete_json`.
- **Different agent** — any tool-calling agent that can run CLI commands can drive aleph. The SKILL.md is just markdown documentation; translate it to any agent's skill/memory format.
- **Better retrieval** — subclass `Retriever` in `src/aleph/retrieval.py` and pass it into `query()` via the `retriever=` parameter, or make your class the default in `default_retriever(store)`. FTS5 (BM25) is the built-in default; plug in embeddings (`sentence-transformers` + FAISS) the same way. The rest of the pipeline doesn't care.
- **PDFs and HTML** — have the agent convert to text before calling `source-add`, or extend `ingest_paths` for API mode. The only requirement is UTF-8 text with stable character offsets.
- **Postgres backend** — `Store` is the only class that knows about SQLite; the schema ports over directly for multi-user deployments. Keep `clear_cache` / cascade semantics and the two `_invalidate_cache_for_*` helpers intact.

## Distribution plan

- **Individuals** — `pip install aleph-kb`, drop the skill folder in your agent's skills directory, point at a folder of notes. Under five minutes.
- **Small teams** — check `aleph.db` into git-lfs, or share via rsync. Because claims carry provenance and views are ephemeral, merge conflicts reduce to last-write-wins on claims; views regenerate.
- **Enterprise** — port `Store` to Postgres, put the CLI behind an API, integrate with existing auth. The claim model is the contribution; the storage substrate is interchangeable.
- **Offline / private** — agent mode with a local model (Claude Code with a local backend, or any OSS agent). Nothing in the core depends on the network.

## Known limitations of this MVP

- **Retrieval** defaults to FTS5 (BM25) with a LIKE fallback. An `EmbeddingRetriever` is available (`pip install sentence-transformers`) with per-claim vector caching in the `claim_embeddings` table; select it via `ALEPH_RETRIEVER=embedding` or by passing `retriever=` to `query()`. New backends only need the single-method `Retriever` protocol.
- **Subject normalization** ships English (default) and Italian (`--locale it`). Additional locales slot in by implementing the `Normalizer` protocol in [`src/aleph/normalization.py`](src/aleph/normalization.py) and registering them in `NORMALIZERS`. Switching the locale of an existing store is a deliberate migration (existing aliases were normalized under the old rules).
- **Aliases are not undoable** by a single command; to reverse an alias, re-alias manually or edit the `subject_aliases` table. This is a design choice: aliasing is a commitment.
- **Entity resolution is lexical** by default. For cases normalization can't catch, aliases are the tool. A fuller embedding-based entity resolution sits on top of `EmbeddingRetriever`.
- **Concepts cannot cite concepts** (v1). Supports must be claims; concept-of-concept requires a schema extension.
- **Condition inference is opt-in and best-effort.** `--extract-conditions` at ingest time and `claim-condition-extract` later emit `explicit=False` links; never silently upgraded to explicit.
- **Retraction is scientific-domain only.** Other domains use `source-remove` (hard delete) or `claim-supersede` — retraction is a concept specific to peer-reviewed literature.
- **No built-in web UI.** The CLI + JSON output is designed to be wrapped.
- **No streaming ingest for huge files.** A 500-page book works; the agent just makes many `claim-add` calls. Break it into sections for throughput.

## Why "Aleph"

Borges's story: a single point containing all points. A knowledge base is the inverse — many points, one structure that lets you look through any of them back at the whole. Also short and not taken.

## License

MIT.
