"""Layered citation verifier (Phase 2).

Judges whether a sentence is supported by the span it cites (plus that
span's context window) in up to three layers, cheapest first. Each layer
either decides or passes; the first decision wins.

1. **Deterministic.** A sentence that restates the whole span verbatim is
   SUPPORTED. A
   sentence stating a number, date, unit or negation the span and context
   don't is UNSUPPORTED (the fidelity checker). Entity flags and dropped
   negations don't decide; they are passed to the LLM layer as hints.
2. **Entailment** (optional). An NLI model's entailment or contradiction
   probability above a threshold decides; anything in between passes.
3. **LLM.** Asked for SUPPORTED / UNSUPPORTED / UNCERTAIN. A failed call or
   an off-vocabulary answer is UNCERTAIN, never a silent UNSUPPORTED.

Verdicts: SUPPORTED, UNSUPPORTED, or UNCERTAIN (no layer could decide; a
human should look). ``baseline_verdict`` runs the verifier ``ask`` uses by
default on the same three-way scale, so the two can be compared on the
labeled eval set (``aleph.verifier_eval``).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .fidelity import check_fidelity
from .log import log

SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
UNCERTAIN = "UNCERTAIN"
VERDICTS = (SUPPORTED, UNSUPPORTED, UNCERTAIN)

# Fidelity issues that reject on their own. Entity flags and a dropped
# negation are only hints: a sentence may restate the positive half of a
# negated span ("fixed at 50 items" from "fixed at 50 items and cannot be
# configured"), and a capitalized opener isn't a fabricated name.
def _rejects(issue: dict) -> bool:
    if issue["kind"] == "negation":
        return issue.get("direction") == "added"
    return issue["kind"] in {"number", "date", "unit"}

_CITATION = re.compile(r"\s*\[(?:claim|concept):[0-9,\s]+\]")


@dataclass
class VerifierResult:
    verdict: str        # SUPPORTED | UNSUPPORTED | UNCERTAIN
    layer: str          # deterministic | entailment | llm | none
    reason: str = ""
    issues: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "layer": self.layer,
                "reason": self.reason, "issues": self.issues}


class Entailer(Protocol):
    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Probabilities keyed ``entailment``, ``contradiction``, ``neutral``."""


class CrossEncoderEntailer:
    """NLI entailment via a sentence-transformers cross-encoder (optional
    ``embeddings`` extra). Label order comes from the model's ``id2label``;
    the fallback is the default model's (contradiction, entailment, neutral)."""

    LABELS = ("contradiction", "entailment", "neutral")

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model = CrossEncoder(model_name)

    def _labels(self) -> tuple[str, ...]:
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if not id2label:
            return self.LABELS
        labels = tuple(str(id2label[i]).lower() for i in sorted(id2label))
        if not {"entailment", "contradiction"} <= set(labels):
            raise ValueError(f"not an NLI model: labels {labels}")
        return labels

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        scores = [float(x) for x in self.model.predict([(premise, hypothesis)])[0]]
        already_probs = min(scores) >= 0 and abs(sum(scores) - 1) < 1e-3
        if not already_probs:  # logits: softmax them
            top = max(scores)
            exps = [math.exp(x - top) for x in scores]
            scores = [e / sum(exps) for e in exps]
        return dict(zip(self._labels(), scores))


def default_entailer() -> Optional[Entailer]:
    """``ALEPH_ENTAILER=nli`` (optionally ``ALEPH_ENTAILER_MODEL``) turns the
    entailment layer on; it is off by default."""
    if os.environ.get("ALEPH_ENTAILER", "").lower() != "nli":
        return None
    model = os.environ.get("ALEPH_ENTAILER_MODEL")
    return CrossEncoderEntailer(model) if model else CrossEncoderEntailer()


VERIFY_LAYERED_SYSTEM = """You are a verifier. Given a source span, the text
around it, and a sentence that cites the span, decide whether the span
(read in its context) supports the sentence.

Reply in JSON: {"verdict": "SUPPORTED" | "UNSUPPORTED" | "UNCERTAIN", "reason": "one short sentence"}

- SUPPORTED: every factual element of the sentence is stated or clearly implied by the span in its context.
- UNSUPPORTED: the sentence adds, changes, overstates or contradicts something.
- UNCERTAIN: you cannot tell from this text alone.

Judge only what the text says, not whether the sentence is true."""

VERIFY_LAYERED_USER_TEMPLATE = """Source span:
---
{span}
---

Context around the span:
---
{context}
---
{hints}
Sentence: {sentence}

Verdict (JSON)."""


def _strip_citations(sentence: str) -> str:
    return _CITATION.sub("", sentence).strip()


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(".!?;:").strip().lower()


class LayeredVerifier:
    def __init__(
        self,
        llm=None,
        entailer: Optional[Entailer] = None,
        *,
        entail_threshold: float = 0.9,
        contradict_threshold: float = 0.9,
    ):
        self.llm = llm
        self.entailer = entailer
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold

    def verify(self, sentence: str, span: str, context: str = "") -> VerifierResult:
        sentence = _strip_citations(sentence)

        # 1. deterministic. Only a restatement of the whole span counts: any
        # fragment can drop what scopes it ("It is false that ...",
        # "Suppose ...").
        if _normalize(sentence) and _normalize(sentence) == _normalize(span):
            return VerifierResult(SUPPORTED, "deterministic", "sentence restates the span")
        issues = [i.to_dict() for i in check_fidelity(sentence, span, context)]
        rejecting = [i for i in issues if _rejects(i)]
        if rejecting:
            return VerifierResult(
                UNSUPPORTED, "deterministic",
                "; ".join(i["message"] for i in rejecting), issues)

        # 2. entailment
        if self.entailer is not None:
            scores = self.entailer.predict(context or span, sentence)
            if scores.get("entailment", 0.0) >= self.entail_threshold:
                return VerifierResult(
                    SUPPORTED, "entailment",
                    f"entailment p={scores['entailment']:.2f}", issues)
            if scores.get("contradiction", 0.0) >= self.contradict_threshold:
                return VerifierResult(
                    UNSUPPORTED, "entailment",
                    f"contradiction p={scores['contradiction']:.2f}", issues)

        # 3. LLM
        if self.llm is None:
            return VerifierResult(UNCERTAIN, "none", "no layer could decide", issues)
        hints = ""
        if issues:
            hints = ("\nA lexical check noted (possibly harmless, e.g. a sentence "
                     "opener): " + "; ".join(i["message"] for i in issues) + "\n")
        try:
            result = self.llm.complete_json(
                VERIFY_LAYERED_SYSTEM,
                VERIFY_LAYERED_USER_TEMPLATE.format(
                    span=span, context=context or span, hints=hints, sentence=sentence),
                max_tokens=256,
            )
        except Exception as e:
            log("layered_verifier_error", level="warning",
                error=type(e).__name__, message=str(e))
            return VerifierResult(UNCERTAIN, "llm", f"verifier error: {e}", issues)
        if not isinstance(result, dict):
            return VerifierResult(UNCERTAIN, "llm", "verifier returned no verdict", issues)
        verdict = str(result.get("verdict", "")).upper()
        reason = str(result.get("reason", ""))
        if verdict not in VERDICTS:
            return VerifierResult(
                UNCERTAIN, "llm", f"off-vocabulary verdict {verdict!r}", issues)
        return VerifierResult(verdict, "llm", reason, issues)


def baseline_verdict(llm, sentence: str, span: str) -> VerifierResult:
    """The verifier ``ask`` uses by default, on the three-way scale:
    GROUNDED -> SUPPORTED; PARTIAL / UNGROUNDED -> UNSUPPORTED; a failed
    call (ERROR) -> UNCERTAIN."""
    from .query import _verify_span

    verdict, reason = _verify_span(llm, span, sentence)
    mapped = {"GROUNDED": SUPPORTED, "ERROR": UNCERTAIN}.get(verdict, UNSUPPORTED)
    return VerifierResult(mapped, "llm", reason)
