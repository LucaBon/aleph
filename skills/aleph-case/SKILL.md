---
name: aleph-case
description: Apply an already-ingested Aleph knowledge base to a concrete case scenario (criminal, civil, regulatory) and produce a grounded legal evaluation with [claim:ID] / [concept:ID] citations, then persist the answer via `aleph view-cache`. Specialisation of the base `aleph` skill for "evaluate this fact pattern against my corpus" workflows. Use when the user gives a fact pattern after a KB has been ingested — triggers include "evaluate this case", "what law applies to", "what's the likely outcome under the corpus", "build a legal opinion on…". Runs entirely in agent mode: no `ANTHROPIC_API_KEY` required.
---

# aleph-case — agent-mode case evaluation against a populated KB

Use this skill *after* the base `aleph` skill has ingested the sources. You (the caller) are the LLM; the DB is the store. The skill's output is a grounded written analysis, not new claims — do not invent facts or paraphrase spans.

## When to invoke

A user describes a fact pattern and asks you to evaluate it against the ingested corpus. Typical triggers:
- "Valuta questo caso alla luce di ciò che ho caricato."
- "What does the corpus say about <scenario>?"
- "Build me a legal opinion on <fact pattern>."

If the DB is empty (`stats-json.claims_active == 0`), stop and invoke the base `aleph` skill's INGEST workflow first.

## Preconditions (quick checks)

```bash
which aleph
aleph --db <DB> stats-json            # claims_active must be > 0
aleph --db <DB> concept-list          # optional — concepts are bonus evidence
aleph --db <DB> contradiction-list --all    # dispositional context
```

`ANTHROPIC_API_KEY` is **not** required for this skill. All synthesis happens in your response.

## Workflow

### Step 1 — Decompose the fact pattern into legal questions

Write down each distinct legal question the fact pattern raises, before touching the DB. For a criminal matter in Italian law, always at minimum consider: base offense, aggravanti, scriminanti, attenuanti, imputabilità, constitutional/proporzionalità frame, any domain-specific special regime (femminicidio, codice rosso, etc.).

### Step 2 — Retrieve evidence in parallel

Run these as a single batch of Bash calls (they are independent):

```bash
aleph --db $DB concept-list
aleph --db $DB contradiction-list --all
aleph --db $DB claim-search "<keyword>" -k 15 --compact     # one per legal question
aleph --db $DB claim-by-subject "<canonical subject>"        # one per concept subject
```

**Flag notes (verify before citing):**
- `claim-search` takes `-k N`, not `--limit N`.
- `--compact` gives id + predicate + truncated object + confidence — ideal for screening.
- `--with-spans` is needed when you have to verify the span supports a sentence.

After screening with `--compact`, pull each claim you intend to cite with `claim-get <id>` to read its full span. **Never cite a claim whose span you have not read.**

### Step 3 — Compose the evaluation

Write the opinion with this structure (adapt to domain):

1. **Qualificazione / charge framing** — the base statutory hook.
2. **Aggravating circumstances** — each with the controlling claim(s) and a short applicability judgement.
3. **Defences / scriminanti** — each ruled in or out with reason.
4. **Mitigating circumstances / attenuanti / imputabilità** — same.
5. **Effective penalty range** — combining the above.
6. **Dispositional tensions** — cite relevant entries from `contradiction-list --all` and explain which rule governs this fact pattern.
7. **Gaps / Flagged for expert review** — use **this exact phrase**, matching the `gap` disposition handling in [query.py](../../src/aleph/query.py).

**Citation contract (mirrors SYNTHESIZE_V2):**
- Every factual sentence ends with `[claim:ID]` or `[concept:ID]`, possibly `[claim:1,5,12]`.
- Draft concepts (status `draft`) are citable in agent-mode fallback **but must be annotated** `(concept draft — unvalidated)`. Only `active` concepts are load-bearing under the normal Aleph pipeline.
- If a fact has no backing claim, say so explicitly and mark it `Flagged for expert review` — do not paraphrase around the gap.

### Step 4 — Persist via `view-cache`

The CLI signature is **positional**:

```bash
aleph --db $DB view-cache "<verbatim case question>" "<full answer text>" --claim-ids "id1,id2,id3,…"
```

- `--claim-ids` is a **comma-separated string**, not a JSON array.
- There is no `--concept-ids` flag on the CLI; concept IDs are stored at the DB layer but the current CLI does not expose them. Cite concepts in the answer body anyway; cache only the claim IDs.
- Pass a long answer via bash command substitution from a file:
  ```bash
  ANSWER=$(cat /tmp/answer.md)
  aleph --db $DB view-cache "$QUESTION" "$ANSWER" --claim-ids "$IDS"
  ```

Verify with `aleph --db $DB view-get "<same question verbatim>"` — the `cached` flag must be `true` and `claim_ids` must match.

### Step 5 — Report LLM-dependent follow-ups

When `ANTHROPIC_API_KEY` is absent, the following steps from the full Aleph pipeline are unavailable; list them explicitly:

- `concept-validate` — promotes draft concepts to `active` via the GROUNDED / PARTIAL / UNGROUNDED verdict over the union of supporting spans.
- `aleph ask` — same case question with per-sentence verification against cited spans; would yield the "Verifier flags" footer.
- `contradiction-scan` — if new claims have been added since the last scan.
- `claim-condition-extract` — to tighten retrieval via scope/method/sample conditions.

Tell the user these steps are pending and will sharpen the evaluation once the key is available; in the meantime the agent-mode opinion is the best available output.

## Invariants (inherited from Aleph)

- **Never fabricate a claim or concept ID.** If you cite `[claim:42]`, `claim-get 42` must have returned and its span must support the sentence.
- **Never paraphrase around missing evidence.** Write "Flagged for expert review" instead.
- **Never merge a contradiction silently.** Report both sides, cite the controlling disposition, and explain which side governs the present facts.
- **Draft concepts are honesty hazards.** Mark them; do not promote them in prose.
- **Do not ingest new material inside this skill.** If the corpus is insufficient, recommend the base `aleph` skill's INGEST workflow and stop.
