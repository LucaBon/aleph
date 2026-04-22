# FAQ — the objections Aleph hears most

One paragraph per objection, ordered by how often it comes up. The [proof block](../README.md#proof) in the README backs up several of these answers with numbers.

## "Isn't this just RAG with extra steps?"

RAG re-derives from scratch on every query: chunk, embed, retrieve, stuff into context, generate. There is no persistent, inspectable, deletable artefact between the source documents and the generated answer. When a source turns out to be wrong, you can hope it doesn't get retrieved next time. You cannot remove its influence from the past cached answers, the embeddings index, or any downstream user's mental model.

Aleph keeps the claim graph as a first-class, inspectable artefact. `source-remove` cascades: every claim from the removed source is deleted and every cached view that cited any of those claims is invalidated atomically — [benchmark](../README.md#proof) shows 1/3 cached views auto-invalidating on a single source removal. That property is unavailable to a RAG pipeline whose only durable state is an embeddings index you can't cleanly subtract from.

There's a second difference: the verifier runs per-sentence, against the *specific* span a sentence cites, not against a retrieved context blob. That is what lets Aleph flag a sentence that paraphrases beyond what its cited span supports — not "the retrieval was about this topic", but "this specific sentence has a factual element that is not in the span it points at".

## "Why not just use a database?"

Aleph IS a database. The contribution is the one invariant it enforces at the write boundary: **every claim's span must be a verbatim substring of its source**. [`cmd_claim_add`](../src/aleph/agent_cli.py) refuses any claim whose span cannot be located by `content.find(span)` — so nothing ungrounded ever enters the store.

Strip that invariant and Aleph becomes a regular SQL database with a couple of auxiliary tables. Keep it and you get provenance-by-construction: every row you can retrieve, you can trace back to specific source characters without a second system, without a content-hash scheme, without trust in whoever wrote the row.

## "LLM-extracted claims are themselves hallucinations."

True in general, and Aleph's answer is the span-verbatim rule. When the ingest LLM fabricates a claim (wrong subject, wrong object, wrong everything), it still has to provide a span that is a verbatim substring of the source — otherwise the write fails. The claim-add boundary does exact `content.find(span)` and rejects mismatches. The surface-level claim text (`subject`, `predicate`, `object`) can still be imperfect; but the span — the thing the verifier later reads — is incorruptible.

In practice this means an LLM can hallucinate the *index card* (the tuple) but not the *underlying text*. At synthesis time the verifier reads the span, not the index card, and flags the sentence if the span doesn't support it. The index card is a retrieval hint, not ground truth.

## "Verbatim-span matching is too strict for real text."

That strictness is the feature, not a limitation. Any fuzzy-match fallback at the write boundary — whitespace-normalised match, near-duplicate match, semantic match — breaks the invariant the verifier relies on. If the span is an approximation, the verifier is checking the generated sentence against an approximation of the source, and the whole grounding guarantee becomes probabilistic again.

In practice, agents drive `claim-add` and are perfectly capable of copying verbatim from text they just read. The ingestion code in [`src/aleph/ingest.py`](../src/aleph/ingest.py) does fall back to whitespace-normalised matching *once*, inside the automated claim-extraction path — but the refusal is the default, and an ungrounded claim is dropped rather than silently accepted.

## "Keyword retrieval won't scale."

For the MVP, yes. `Store.search_claims` ([db.py](../src/aleph/db.py)) is SQLite `LIKE`-based keyword ranking. It works well for specific terms and proper nouns; it fails on thesaurus-style queries.

This is the **one** swap point for retrieval. Every other part of the pipeline consumes a ranked list of claim rows and doesn't care how they were ranked. Dropping in BM25 (`rank-bm25`) or embeddings (`sentence-transformers` + FAISS) means rewriting one method. The claim graph, the verifier, the cache, the cascade-delete — all unaffected.

If a library user were deploying Aleph against a million-claim corpus, the first change would be this one method. It's labelled as such in both the code and the README's "Extending it" section.

## "Claim extraction is too expensive to run over a large corpus."

Extraction is one-time per source. Cheap, high-throughput models are fine — the extraction LLM's job is to propose (subject, predicate, object, span) tuples. The span is the ground truth; the tuple is an index card. Small errors in the tuple don't propagate because the verifier reads spans, not tuples.

Once extracted, queries are keyword-indexed SQL: no LLM in the retrieval path, no embeddings to recompute, no rebuild step when a source is added. Query-time LLM cost is one synthesis call plus one verifier call per cited sentence — and the [benchmark](../README.md#proof) shows the entire cost is skipped on a cache hit.

The pricing story ends up inverted from typical RAG: ingest is a one-time LLM-heavy step; steady-state operation is SQL-heavy and LLM-light.

## "Why not embeddings as the primitive?"

Embeddings are a retrieval strategy, not a storage primitive. Aleph decouples the two on purpose: claims are what you store; retrieval is pluggable. You can put embeddings underneath Aleph tomorrow by rewriting `Store.search_claims` — and the claim graph is still there, still auditable, still deletable-with-cascade.

An embeddings-first design conflates "we found a chunk that seems topically related" with "we have a proposition that specific text supports". Aleph separates them. Retrieval surfaces candidate claims; the verifier checks each generated sentence against the specific claim's specific span. Swap the embeddings out for BM25 and nothing else changes. That separation is why Aleph's guarantees — deletion cascade, contradiction detection, per-sentence verifier — are orthogonal to whatever retrieval stack you bolt on.
