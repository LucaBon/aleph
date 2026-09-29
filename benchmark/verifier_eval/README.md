# Verifier eval

Measures how often a citation verifier accepts a sentence its span doesn't support (false accept), and how often it rejects or can't decide on one the span does support (false reject). Phase 2 of the [roadmap](../../docs/roadmap.md) requires these rates to be published for the baseline verifier and for the layered verifier, from 200–300 human-labeled pairs.

## Status: draft labels, no published rates

`pairs.jsonl` holds 49 pairs drawn from `benchmark/corpus/`. Every label is `"labeler": "draft:claude"`: Claude wrote the sentences and proposed the labels. The same author wrote the fidelity checker, so this set is biased toward errors that checker catches. It is a harness fixture and a starting point for labelers. It is **not** the eval set. Reports on it carry `"labels_status": "draft"`, and no rate computed from it should be quoted.

To get the Phase 2 exit numbers:

1. Grow the set to 200–300 pairs. Include sentences written by people other than the checker's author, and sentences from real `aleph ask` output on `corpus/`.
2. Have a person label each pair and set `labeler` to `human:<id>`. Pairs with two labelers should agree, or be dropped.
3. Run the baseline and the layered verifier (below) and publish both reports.

## Pair format

One JSON object per line:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique pair id |
| `span` | yes | The cited span, verbatim from the source |
| `context` | no | The span's context window (`aleph.fidelity.context_window`) |
| `sentence` | yes | The sentence that cites the span |
| `label` | yes | `SUPPORTED` or `UNSUPPORTED` |
| `error_type` | for `UNSUPPORTED` | What is wrong, e.g. `number`, `date`, `negation`, `entity-swap`, `scope`, `overgeneralization`, `addition`, `contradiction`, `outside-context` |
| `labeler` | yes | `human:<id>` or `draft:<id>` |
| `source` | no | Source file the span came from |

## Labeling rule

Judge the sentence against the `span` and `context` shown in the pair, not against the rest of the document. A sentence is `SUPPORTED` only if every factual element in it (numbers, dates, units, negation, named entities, scope, and causal claims) is stated or clearly implied there.

This rule makes three draft pairs (`v021`, `v035`, `v037`) `UNSUPPORTED` with `error_type: outside-context`. Each takes a date from a section heading. The heading is in the source, but it is outside the context window, which stops at paragraph breaks. Whether the context window should include the enclosing section heading is an open design question. These pairs measure its cost.

## Rates

- **false-accept rate** = pairs predicted `SUPPORTED` / gold `UNSUPPORTED` pairs
- **false-reject rate** = pairs predicted `UNSUPPORTED` or `UNCERTAIN` / gold `SUPPORTED` pairs
- **uncertain rate** = pairs predicted `UNCERTAIN` / all pairs

`UNCERTAIN` goes to a human, so it never counts as an accept. It does count against false-reject, because it costs reviewer time.

## Running

```bash
# No API key: the layered verifier's deterministic layer on its own
python -m aleph.verifier_eval --pairs benchmark/verifier_eval/pairs.jsonl \
    --verifier deterministic --out det.json

# Baseline: ask's default span check (needs ANTHROPIC_API_KEY)
python -m aleph.verifier_eval --pairs benchmark/verifier_eval/pairs.jsonl \
    --verifier baseline --out baseline.json

# Layered: deterministic -> entailment (if ALEPH_ENTAILER=nli) -> LLM
python -m aleph.verifier_eval --pairs benchmark/verifier_eval/pairs.jsonl \
    --verifier layered --out layered.json
```

`--model` picks the LLM. `--mock-llm FIXTURES` swaps in the offline MockLLM, which is only useful for testing the harness. Each report includes the confusion matrix, false accepts by `error_type`, which layer decided each pair, token usage, and every misjudged pair with the verifier's reason.

On the draft set, the deterministic layer decides 14 of 49 pairs (3 `SUPPORTED`, 11 `UNSUPPORTED`) with no errors, and leaves the rest `UNCERTAIN`. Because of the bias described above, this is a smoke result and not an accuracy claim.
