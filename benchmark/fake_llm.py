"""Deterministic fake LLM that drives the offline benchmark.

Three modes the benchmark needs:
- claim extraction: return the ground-truth claims whose spans appear in the user prompt
- synthesis (with Aleph): return the pre-canned grounded answer, substituting real DB claim ids
- synthesis (naive, no Aleph): return the pre-canned ungrounded answer (no citations)
- verifier: GROUNDED if the sentence's numeric facts appear in the span, else UNGROUNDED
- contradiction judge: mark seeded pairs as contradicting, others as not

Keeping this in its own module so both run.py (offline) and any future instrumentation
can import it without pulling in the benchmark driver.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class BenchmarkFakeLLM:
    """Deterministic LLM whose behaviour is fully driven by ground_truth.json."""

    def __init__(self, ground_truth: dict):
        self.ground_truth = ground_truth
        self.local_to_real_id: dict[str, int] = {}
        self.calls = 0

    def set_id_map(self, mapping: dict[str, int]) -> None:
        """Install the local_id -> real DB claim id mapping (after ingest)."""
        self.local_to_real_id = dict(mapping)

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        self.calls += 1
        if "You answer a question" in system:
            return self._synthesize(user)
        return ""

    def complete_json(self, system: str, user: str, max_tokens: int = 4096):
        self.calls += 1
        if "You extract atomic claims" in system:
            return self._extract(user)
        if "You are a verifier" in system:
            return self._verify(user)
        if "You judge whether two claims" in system:
            return self._judge(user)
        return {}

    # ------- extraction -------

    def _extract(self, user: str) -> list:
        out = []
        for _fname, block in self.ground_truth["sources"].items():
            for c in block["claims"]:
                if c["span"] in user:
                    out.append({
                        "subject": c["subject"],
                        "predicate": c["predicate"],
                        "object": c["object"],
                        "span": c["span"],
                        "confidence": c["confidence"],
                    })
        return out

    # ------- synthesis -------

    def _synthesize(self, user: str) -> str:
        for q in self.ground_truth["questions"]:
            if q["query"].lower() in user.lower():
                ans = q["grounded_answer_with_aleph"]
                for local, real in self.local_to_real_id.items():
                    ans = ans.replace("{" + local + "}", str(real))
                return ans
        return "No claims in the supplied block cover this question."

    def naive_answer(self, query: str) -> str:
        """Answer freehand with no citations — what a vanilla LLM would produce.
        Called directly by the benchmark, not via complete().
        """
        for q in self.ground_truth["questions"]:
            if q["query"].lower() == query.lower():
                return q["naive_answer_without_aleph"]
        return "I don't know."

    # ------- verifier -------

    def _verify(self, user: str) -> dict:
        """GROUNDED iff every numeric token in the sentence appears in the span.
        Claim-id tokens (`[claim:N]`) are stripped before extraction — they are
        Aleph's bookkeeping, not a factual assertion.
        """
        m = re.search(r"Source span:\s*---\s*(.*?)\s*---\s*Sentence:\s*(.*?)\s*Verdict", user, re.S)
        if not m:
            return {"verdict": "UNGROUNDED", "reason": "could not parse prompt"}
        span, sentence = m.group(1), m.group(2)
        nums_sent = self._numbers(sentence)
        nums_span = self._numbers(span)
        if not nums_sent:
            return {"verdict": "GROUNDED", "reason": "no numeric claim to verify"}
        missing = [n for n in nums_sent if n not in nums_span]
        if missing:
            return {"verdict": "UNGROUNDED", "reason": f"numbers not in span: {missing}"}
        return {"verdict": "GROUNDED", "reason": "numbers in sentence appear in span"}

    def _judge(self, user: str) -> dict:
        """Contradiction judge: mark seeded pairs as contradicting, others not.

        Looks up each seeded pair's (object_a, object_b) text and returns True if
        both appear in the prompt the linter supplied. Non-seeded pairs default
        to False — matches the lint-semantics: different objects for same subject
        aren't automatically contradictions (see Tesla example in lint.py).
        """
        seeded_objects = []
        claim_by_id = {c["local_id"]: c for b in self.ground_truth["sources"].values() for c in b["claims"]}
        for pair in self.ground_truth["seeded_contradictions"]:
            a = claim_by_id.get(pair["a"])
            b = claim_by_id.get(pair["b"])
            if a and b:
                seeded_objects.append((a["object"], b["object"]))
        for a_obj, b_obj in seeded_objects:
            if a_obj in user and b_obj in user:
                return {"contradicts": True, "reason": "seeded contradiction"}
            if b_obj in user and a_obj in user:
                return {"contradicts": True, "reason": "seeded contradiction"}
        return {"contradicts": False, "reason": "different facets or non-seeded pair"}

    _CLAIM_TOKEN = re.compile(r"\[claim:[\d,\s]+\]")

    @classmethod
    def _numbers(cls, text: str) -> set[str]:
        """Normalised numeric tokens: strip commas and claim-id tokens."""
        scrubbed = cls._CLAIM_TOKEN.sub("", text)
        raw = re.findall(r"\d[\d,]*", scrubbed)
        return {n.replace(",", "") for n in raw}


def load_ground_truth(root: Path) -> dict:
    return json.loads((root / "benchmark" / "ground_truth.json").read_text())
