# Retrieval eval

Measures claim recall@k: of the claims a question needs, how many the retriever returns in its top k. Phase 3 of the [roadmap](../../docs/roadmap.md) requires hybrid retrieval to beat both FTS-only and embedding-only retrieval on this measure, over a labeled query set.

## Status: draft labels, embeddings not yet run

There are two query files. Every label in both is `"labeler": "draft:claude"`: Claude wrote the queries and chose the relevant claims.

| File | Queries | Runs against | Use |
|---|---|---|---|
| `benchmark_queries.jsonl` | 7 | the 12 hand-written claims in `benchmark/ground_truth.json` (`--from-ground-truth`) | CI smoke test of the harness. It saturates: every retriever scores 1.0 at k=5 |
| `corpus_queries.jsonl` | 17 | a store built from `corpus/` (Italian criminal law) | Draft of the real eval. Most queries paraphrase rather than repeat the claim's words |

On a local store built from `corpus/` (198 claims; the store isn't in the repo), with k=10 on draft labels:

| Retriever | Mean recall@10 | Hit rate |
|---|---|---|
| FTS | 0.79 | 0.94 |
| Keyword | 0.73 | 0.88 |
| Embedding | not run | not run |
| Hybrid (FTS + embedding, RRF) | not run | not run |

The embedding and hybrid retrievers need the optional `embeddings` extra (`sentence-transformers`), which wasn't installed where these numbers were produced. The exit criterion is therefore unmeasured.

To get the Phase 3 exit numbers:

1. Have a person review the relevant sets in `corpus_queries.jsonl`, add queries (aim for 50 or more), and set `labeler` to `human:<id>`.
2. Install the extra with `pip install -e '.[embeddings]'`.
3. Run all four retrievers and publish the report.

## Query format

```json
{"id": "c01", "query": "quando si può invocare la legittima difesa",
 "relevant": [{"source": "corpus/art-52-cp.txt", "span_contains": "Non è punibile chi ha commesso il fatto"}],
 "labeler": "human:<id>"}
```

A `relevant` entry matches every active claim whose source path ends with `source` and whose span contains `span_contains` (whitespace-normalized). Naming claims by their text rather than their ids lets labels survive a store rebuild. An entry that matches nothing is listed under `unresolved`.

## Running

```bash
python -m aleph.retrieval_eval --db aleph.db \
    --queries benchmark/retrieval_eval/corpus_queries.jsonl \
    --retriever fts,keyword,embedding,hybrid -k 10 --out retrieval.json

python -m aleph.retrieval_eval --from-ground-truth benchmark/ground_truth.json \
    --queries benchmark/retrieval_eval/benchmark_queries.jsonl --retriever fts,keyword -k 5
```

The report gives mean recall@k, the hit rate, and per-query relevant versus retrieved ids for each retriever. Queries go through the same keyword extraction that `ask` uses.
