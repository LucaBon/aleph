# Notes: scaling supervised fine-tuning for domain adaptation

## Preliminary results (Patel et al., internal memo, March 2023)

We fine-tuned a 7B-parameter base model on a curated domain corpus of
approximately 10,000 documents drawn from internal engineering writeups.
Evaluation on the DOMAIN-BENCH held-out set showed that supervised
fine-tuning yields a 5% absolute accuracy gain over the base model on
technical question-answering tasks.

Training took 18 hours on 8 A100 GPUs. The gain was largest on
terminology-heavy questions and smaller on procedural questions. No
measurable improvement was observed on questions outside the domain,
consistent with the narrow training distribution.

## Follow-up study (Patel et al., revised manuscript, October 2024)

With an expanded training corpus of 50,000 domain documents and careful
deduplication against the evaluation set, supervised fine-tuning yields
a 12% absolute accuracy gain over the base model on DOMAIN-BENCH. The
larger gain is attributable primarily to the larger and cleaner
training set; the earlier memo underestimated the ceiling because of
evaluation contamination we have since measured and removed.

We now estimate the effective training set size, after deduplication,
at 50,000 documents. We believe earlier reports of a 5% gain should be
considered superseded by these results. Training on the expanded set
took 74 hours on 8 A100 GPUs.
