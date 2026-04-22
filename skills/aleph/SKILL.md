---
name: aleph
description: Build and query a grounded knowledge base from documents. Use this skill whenever the user wants to ingest articles, papers, notes, or other source files and later ask questions with verifiable citations. Every claim is anchored to an exact source span, so answers never drift from what the source actually says. Triggers include "ingest this paper", "add to my knowledge base", "what does my research say about X", "compile a wiki from these notes", or mentions of the aleph CLI.
---

# Aleph: stateless-view knowledge base

Aleph is a Python CLI (`aleph`) that stores **atomic claims with source spans**, not LLM-authored pages. You — the agent — do the reading, reasoning, and writing. The CLI does the persisting and retrieval. There is no hidden LLM behind aleph; your responses *are* the synthesis.

Install with `pip install aleph-kb`. Database defaults to `./aleph.db` (per-project) or `~/.aleph/aleph.db`.

## Core principle

The user's sources are the source of truth. The database stores atomic `(subject, predicate, object, source_span)` tuples — each one a pointer back to verbatim text in a source. Prose answers ("views") are generated on demand, cited inline, verified against spans, and cached but never treated as authoritative. **If a source is removed, every claim it supported disappears, and every cached answer that cited them is invalidated.** Nothing synthetic compounds.

## Why each workflow matters

The three workflows map to four measurable guarantees (numbers from [`benchmark/run.py`](../../benchmark/run.py)):

- **Ingest** enforces the grounding invariant. Claims written via `claim-add` are refused unless their span is a verbatim substring of the source — this is what makes the verifier meaningful downstream.
- **Ask** surfaces claims, synthesises a cited answer, and runs the verifier per sentence. **83% of generated sentences pass per-span verification** in the benchmark; a freehand answer from the same LLM scores 0% because there is no citation discipline to verify against.
- **Lint** detects cross-source contradictions and invalidates downstream views when they resolve. In the benchmark it catches **4/4 seeded contradictions** at 100% precision; and `source-remove` auto-invalidates every cached view that cited any claim from the removed source.

Keep these numbers in mind: they're the reason the three workflows exist in this specific shape. Don't paraphrase the source in the span; don't skip the verifier; don't ignore lint output.

## When to use

- The user asks you to read files and build a knowledge base from them ("ingest", "add these papers", "build a wiki").
- The user asks a question about material previously ingested ("what does my research say about…", "how did we compare…").
- The user wants to audit, clean, or update an existing knowledge base ("find contradictions", "this source was wrong, remove it").

## Three workflows

Every interaction is one of: **ingest**, **ask**, or **lint**. They are all driven by you through CLI calls.

---

### Workflow 1: INGEST

**Goal:** Convert a source document into atomic, grounded claims.

**Steps:**

1. **Register the source.** Read the file, then call:
   ```
   aleph source-add /path/to/file.md
   ```
   Returns `{"source_id": N, "length": L}`. If the file was already ingested (hash-deduplicated), returns `"status": "already_ingested"` with the existing id — stop here if so.

2. **Read the content.** Either from the file directly, or from:
   ```
   aleph source-get <source_id>
   ```
   which returns `{"content": "..."}`. For long sources, process it in sections.

3. **Extract claims.** For each factual statement in the source, emit ONE claim via:
   ```
   aleph claim-add --source-id N \
     --subject "Tesla Model S packs" \
     --predicate "retain" \
     --object "about 90% of capacity after 200,000 miles" \
     --span "<verbatim substring from the source>" \
     --confidence 0.9
   ```

   **Rules for good claims:**
   - **Atomic.** One fact per claim. If a sentence has five facts, emit five claims.
   - **Span must be verbatim.** Copy the exact substring of the source — same whitespace, same punctuation. If the CLI reports `"span not found"`, you paraphrased; try again with the exact text.
   - **Specific subjects.** `"Tesla Model S packs"` not `"batteries"`. Be as specific as the source is.
   - **Specific objects.** `"8 to 10 years"` not `"a long time"`.
   - **Paraphrase, don't infer.** If the source says "roughly", say "roughly". Don't promote to "exactly."
   - **Confidence** reflects how clearly the source asserts the fact — a direct statement by the author is 0.9+; something attributed to a third party is lower; speculation is not a claim.
   - **Skip** headings, transitions, rhetorical questions, opinions-not-attributed.

4. **Report.** Tell the user how many claims were added and roughly what entities the source touched on.

**Normalization happens automatically.** `"Tesla batteries"` and `"tesla battery"` get stored under the same canonical subject. You don't need to worry about matching case or plural forms.

---

### Workflow 2: ASK

**Goal:** Generate a cited, verified answer from existing claims.

**Steps:**

1. **Check cache.**
   ```
   aleph view-get "<the user's question>"
   ```
   If `"cached": true`, you can return the cached response as-is (mention it's cached). Skip to step 6. If the user wants a fresh answer or the ingested material has changed recently, skip the cache.

2. **Retrieve candidates.**
   ```
   aleph claim-search "<keywords from the question>" -k 30 --with-spans
   ```
   Returns a ranked list of claims with their source spans. If there are very few results, try a broader query. If none, tell the user and stop — don't guess.

3. **Compose the answer.** Write prose that answers the question, using ONLY the returned claims. Each factual sentence must end with `[claim:N]` or `[claim:N,M]` citing the claim(s) that support it. Do not invent claim IDs. If claims disagree with each other, surface the disagreement — don't pick one silently.

4. **Verify each sentence.** Re-read each sentence against the source span of its cited claim. Ask yourself: *does this span actually support this sentence?*
   - **GROUNDED** — every factual element is stated or clearly implied by the span.
   - **PARTIAL** — mostly supported, but the sentence adds or changes something.
   - **UNGROUNDED** — the span doesn't support the sentence.

   Rewrite any PARTIAL or UNGROUNDED sentence until it's GROUNDED, or drop it. Don't emit unverified content.

5. **Cache.**
   ```
   aleph view-cache "<the question>" "<your final answer>" --claim-ids "1,3,7,12"
   ```

6. **Respond.** Give the user the answer. If any sentence cites multiple claims, that's fine — usually it means supporting detail across sources.

---

### Workflow 3: LINT

**Goal:** Find contradictions, stale claims, and synonymous subjects.

**Steps:**

1. **List subjects by frequency.**
   ```
   aleph subjects
   ```
   Returns every distinct canonical subject with a claim count.

2. **Look for synonymous subjects.** Scan the list for entries that refer to the same real-world entity. Common cases:
   - abbreviations vs full names (`"nca"` vs `"nickel-cobalt-aluminum cell"`)
   - partial vs full references (`"model s"` vs `"tesla model s"`)
   - related-but-not-same references that the source treats interchangeably (`"lithium-ion pack"` vs `"battery pack"`)

   For each confirmed synonym pair, call:
   ```
   aleph alias-add "<variant>" "<canonical form>"
   ```
   This rewrites every existing claim and clears stale cached views. Do NOT alias things that are merely related (`"tesla"` ≠ `"tesla battery"`).

3. **Find contradictions.** For each subject with multiple claims, pull them:
   ```
   aleph claim-by-subject "<subject>"
   ```
   For each pair of claims about the same subject with incompatible objects (e.g. "lasts 8 years" vs "lasts 15 years"), decide whether they genuinely contradict. Remember:
   - `"Tesla — makes — cars"` and `"Tesla — makes — batteries"` are NOT contradictions.
   - `"Tesla batteries — last — 8 years"` and `"Tesla batteries — last — 15 years"` IS a contradiction.
   - When the source itself flags one claim as outdated, that's a contradiction the author already resolved; record it and supersede the old one.

   For each real contradiction:
   ```
   aleph contradiction-add <claim_a_id> <claim_b_id>
   ```

4. **Resolve (with user consent for non-obvious cases).**
   For each open contradiction, decide the resolution policy:
   - The user's source explicitly says one is outdated → supersede the old.
   - More recent source wins → supersede the older claim.
   - Both plausible → leave open and surface to the user.

   ```
   aleph contradiction-resolve <contradiction_id> --keep <winner> --drop <loser>
   ```
   The `--drop` marks the loser as `superseded`; it stays in the DB with a pointer to the winner, but won't surface in searches.

5. **Report.** Summarize: aliases added, contradictions found, contradictions resolved, cache invalidations.

---

## Quick reference

| command | what it does |
|---|---|
| `source-add PATH` | register a file as a source |
| `source-get ID` | return source content |
| `source-list` | list all sources |
| `source-remove ID` | cascade-delete source + its claims + stale views |
| `claim-add --source-id N --subject S --predicate P --object O --span TEXT --confidence C` | record a claim |
| `claim-get ID` | full claim + source span |
| `claim-search QUERY [-k N] [--with-spans]` | keyword search |
| `claim-by-subject SUBJECT` | all active claims for a subject (alias-resolved) |
| `claim-supersede OLD NEW` | mark OLD as superseded by NEW |
| `subjects` | all distinct subjects by count |
| `alias-add FROM TO` | declare FROM is the same as TO; rewrites claims |
| `alias-list` | show alias table |
| `contradiction-add A B` | record a contradiction |
| `contradiction-list [--all]` | open (or all) contradictions |
| `contradiction-resolve ID --keep K --drop D` | resolve and supersede |
| `view-get QUERY` | check cache for this question |
| `view-cache QUERY RESPONSE --claim-ids CSV` | cache your composed answer |
| `cache-clear` | drop all cached views |
| `stats-json` | counts |

All commands emit JSON on stdout. Use `--db PATH` to target a non-default database.

## Anti-patterns

- **Don't paraphrase in span fields.** The CLI verifies spans are verbatim substrings. Paraphrased spans are refused — this is the single most important guarantee the system makes.
- **Don't invent claim IDs.** Only cite IDs that `claim-search` returned to you.
- **Don't store opinions as claims.** "The author argues that…" is a claim *about an argument*, not a claim about the world. Phrase it as the former.
- **Don't merge subjects with aliases that aren't actually the same entity.** If the source distinguishes `"LFP cells"` from `"NCA cells"`, keep them distinct.
- **Don't answer from your training data.** If `claim-search` returns nothing, say so and offer to search the web or ingest new sources — don't backfill from memory.
