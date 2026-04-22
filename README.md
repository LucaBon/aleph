# Aleph

**A knowledge base where every answer sentence ends in a citation that checks out — against the exact source characters, one sentence at a time.**

In Aleph's offline benchmark, **83% of generated answer sentences pass per-span verification**. The same LLM answering the same questions freehand — no claims, no citations — passes **0%**. Full numbers in the [proof block](#proof) below; `python benchmark/run.py` reproduces them without an API key.

This is an implementation of the "Claim Layer" — a response to Karpathy's [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern that keeps the compounding benefit while eliminating the main failure modes (hallucination propagation, lost provenance, silent drift, deletion pain).

## The idea in 30 seconds

You built a RAG pipeline. It hallucinates. When an answer is wrong, you can't tell which sentence is wrong, or where it came from, or how to fix it. When a source turns out to be wrong, you can't cleanly remove its influence — it's already been mixed into cached prose, summaries, and downstream indexes.

Aleph sits one layer deeper. Three things, strictly separated:

- **Sources** — immutable, hashed, trusted. The source of truth.
- **Claims** — atomic `(subject, predicate, object, span, confidence)` tuples, each pointing back to an exact substring of a source. Index cards, not summaries.
- **Views** — human-readable answers to questions. Generated on demand from claims, cited inline, verified sentence-by-sentence against the source spans they cite. Never authoritative. Never edited.

Nothing synthetic is ever the source of truth. Any wrong claim can be removed and every downstream view regenerates clean.

## Proof

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

The 83% figure is honest: the verifier flags a sentence that mentions "2022 policy" while the cited claim's span only contains the retention-window fact. That's the verifier working — catching a paraphrase that isn't supported by the specific span it cites, even when the date is technically in the same source file. A naïve pipeline has no way to flag that.

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

**Policy Q&A inside a regulated org.** Ingest the current policy set. When policies update, ingest the new version too; `aleph lint` flags the conflicting clauses. A resolution workflow (`--resolve-by-recency` or manual `contradiction-resolve`) marks the old clause as superseded. Answers to "what's our data retention policy?" always cite the current source — with the exact span quoted inline.

**Durable agent memory with no drift.** Your agent loop extracts claims as it reads material, via the `claim-add` JSON command. Each claim enters the store only if its span is a verbatim substring of the source — ungrounded claims are refused at the write boundary. The agent can later retrieve claims by subject or keyword and compose answers without ever paraphrasing its own prior prose.

## How it works

```
  ┌──────────────┐       ┌──────────────┐      ┌──────────────────┐
  │   sources    │──────▶│    claims    │◀────▶│ contradictions   │
  │  (immutable) │       │ (pointers to │      │  (first-class    │
  │              │       │  source      │      │   nodes)         │
  └──────────────┘       │  spans)      │      └──────────────────┘
                         └──────┬───────┘             ▲
                                │                     │
                         ┌──────┴───────┐             │
                         │   subject    │─────────────┘
                         │   aliases    │  (canonicalize before
                         │              │   grouping)
                         └──────┬───────┘
                                │  retrieve (deterministic)
                                ▼
                         ┌──────────────┐       ┌──────────────────┐
                         │   synthesize │──────▶│    verifier      │
                         │   view       │       │  (per sentence   │
                         │ (LLM / agent)│       │   vs source span)│
                         └──────┬───────┘       └────────┬─────────┘
                                │                        │
                                ▼                        ▼
                         ┌───────────────────────────────────────┐
                         │          view (ephemeral)             │
                         │  prose + [claim:N] citations + flags  │
                         └───────────────────────────────────────┘
```

The key asymmetry: **the left side is permanent and deterministic; the right side is ephemeral and probabilistic.** Scoping is a graph/keyword query. Synthesis is an LLM call (or an agent turn). Verification is an LLM/agent check against specific source spans, never against earlier generated prose.

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

The first two run the existing end-to-end pipeline and CLI tests. The third runs the benchmark that produces the numbers in the [proof block](#proof).

## Commands

**Agent-mode** (no API key, JSON in/out — what Claude Code calls):

```
source-add PATH                  register a source
source-get ID                    fetch source content
source-list                      list sources
source-remove ID                 cascade-delete
claim-add --source-id N --subject S --predicate P --object O --span TEXT --confidence C
claim-get ID                     claim + its source span
claim-search QUERY [-k N]        keyword search over claims
claim-by-subject SUBJECT         all claims for a subject (alias-resolved)
claim-supersede OLD NEW          mark OLD as superseded
subjects                         distinct subjects by count
alias-add FROM TO                declare FROM = TO; rewrites existing claims
alias-list                       show aliases
contradiction-add A B            record contradiction
contradiction-list [--all]       open (or all)
contradiction-resolve ID --keep K [--drop D]
view-get QUERY                   cache lookup
view-cache QUERY RESPONSE --claim-ids CSV
cache-clear
stats-json
```

**API-mode** (requires `ANTHROPIC_API_KEY`):

```
aleph ingest PATH [PATH ...]     ingest files or directories
aleph ask "question"             generate + verify
aleph lint [--resolve-by-recency]
aleph show [ID]                  show a claim or list them
aleph sources                    list sources (human-readable)
aleph remove SOURCE_ID           remove + cascade
aleph stats                      counts
aleph clear-cache
```

Global flags: `--db PATH` and `--model MODEL`.

## But isn't this just…?

The three most common objections, one sentence each. Full treatment in [docs/FAQ.md](docs/FAQ.md).

- **"…just RAG with extra steps?"** → RAG re-derives per query; Aleph has a persistent, inspectable, deletable claim graph. The difference shows up the moment a source turns out to be wrong (see: deletion-propagation row in the [proof block](#proof)).
- **"…just a database?"** → Aleph IS a database with one invariant enforced at the write boundary: every claim's span must be a verbatim substring of its source. Strip that invariant and you have a database. Keep it and you have provenance-by-construction.
- **"…the verbatim-span requirement is too strict for real text."** → That strictness is the feature. A fuzzy-match fallback at the write boundary defeats the guarantee that every claim maps back to specific source characters — which is what makes the verifier meaningful.

Four more objections (claim extraction itself hallucinates, keyword retrieval won't scale, ingest is too expensive, why not embeddings) are answered in the FAQ.

## How this addresses each LLM Wiki limitation

| LLM Wiki problem | Aleph's fix |
|---|---|
| Hallucinations get baked into pages and compound | Pages (views) are ephemeral. Every read regenerates from current claims. Nothing compounds because nothing is permanent. |
| Lossy summarization — edge cases vanish | Claims are atomic. Each edge case is its own claim. The agent reads actual source spans at synthesis time, not earlier summaries. |
| Provenance collapses | Every sentence cites specific claim IDs; every claim points to an exact span. One hop back to the original. |
| `index.md` breaks at ~200–500 pages | Retrieval is deterministic SQL over indexed claim fields. No "read index into context" step. |
| Retrieval-as-reasoning is silently lossy | Scoping is keyword/graph (deterministic, auditable). The agent only reasons over pre-filtered claims. |
| No deletion, contradiction, supersession | `source-remove` cascades. Contradictions are first-class nodes with resolution policies. View cache invalidates automatically. |
| Subject synonyms silently split knowledge | Subjects are normalized at write time (case, whitespace, simple plurals). For the hard cases, `alias-add` rewrites all claims atomically. |

## When to use this vs. LLM Wiki vs. RAG

- **LLM Wiki** (Karpathy's pattern) — best for personal research where you want to *browse* an evolving vault in Obsidian. The page-as-artifact is the point.
- **RAG** — best at massive scale (tens of thousands of documents, frequently updated) where in-context knowledge isn't possible.
- **Aleph** — best when correctness and auditability matter: research teams, anything regulatory or medical, enterprise KBs, any case where a source can turn out to be wrong and you need to pull its influence cleanly. Also good as a substrate under a chat UI where users want citations that actually check out.

## Extending it

- **Different LLM provider (API mode)** — rewrite `src/aleph/llm.py`. Keep `complete` and `complete_json`.
- **Different agent** — any tool-calling agent that can run CLI commands can drive aleph. The SKILL.md is just markdown documentation; translate it to any agent's skill/memory format.
- **Better retrieval** — rewrite `Store.search_claims` in `db.py`. Drop in BM25 (`rank-bm25`) or embeddings (`sentence-transformers` + FAISS). The rest of the system doesn't care.
- **PDFs and HTML** — have the agent convert to text before calling `source-add`, or extend `ingest_paths` for API mode. The only requirement is UTF-8 text with stable character offsets.
- **Postgres backend** — `Store` is the only class that knows about SQLite; the schema ports over directly for multi-user deployments.

## Distribution plan

- **Individuals** — `pip install aleph-kb`, drop the skill folder in your agent's skills directory, point at a folder of notes. Under five minutes.
- **Small teams** — check `aleph.db` into git-lfs, or share via rsync. Because claims carry provenance and views are ephemeral, merge conflicts reduce to last-write-wins on claims; views regenerate.
- **Enterprise** — port `Store` to Postgres, put the CLI behind an API, integrate with existing auth. The claim model is the contribution; the storage substrate is interchangeable.
- **Offline / private** — agent mode with a local model (Claude Code with a local backend, or any OSS agent). Nothing in the core depends on the network.

## Known limitations of this MVP

- **Keyword retrieval** over claims. Works fine for proper nouns and specific terms; for thesaurus-style queries add embeddings. One-method swap in `db.py`.
- **Subject normalization is English-only** (handles basic plurals via `-s`, `-ies`, `-sses`). Non-English corpora need either a language-aware normalizer or heavier reliance on the alias table.
- **Aliases are not undoable** by a single command; to reverse an alias, re-alias manually or edit the `subject_aliases` table. This is a design choice: aliasing is a commitment.
- **Entity resolution is lexical** by default. For cases normalization can't catch, aliases are the tool. A fuller embedding-based entity resolution is an obvious next layer.
- **No built-in web UI.** The CLI + JSON output is designed to be wrapped.
- **No streaming ingest for huge files.** A 500-page book works; the agent just makes many `claim-add` calls. Break it into sections for throughput.

## Why "Aleph"

Borges's story: a single point containing all points. A knowledge base is the inverse — many points, one structure that lets you look through any of them back at the whole. Also short and not taken.

## License

MIT.
