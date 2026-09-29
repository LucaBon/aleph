"""Deterministic claim-fidelity checks (Phase 2).

The verbatim-span contract proves a claim points at real source text. It does
not prove the claim says what that text says: an extractor can copy the right
span and still write 80% for 90%, drop a "not", or swap the entity. This
module catches the mechanical versions of those errors without an LLM.

Every number, date, unit, negation and entity in the claim must appear in the
span or in the span's context window. Negation is checked in both directions:
a claim may not add a negation the text lacks, nor drop one the *span* states.

The checker only flags. Flags go to the review queue; they never block a
write, because a faithful paraphrase can still trip a lexical check
("ninety" vs "90"). The verbatim-span check stays the only hard gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

# ---------- context window ----------

_SENTENCE_BREAK = re.compile(r"[.!?][\"')\]]*\s+")


def context_window(
    text: str, start: int, end: int, *, radius: int = 1, max_chars: int = 600,
) -> tuple[int, int]:
    """Offsets of the span's enclosing sentence plus ``radius`` sentences on
    each side, never crossing a paragraph break, capped at ``max_chars``
    (trimmed symmetrically around the span). Always contains ``[start, end)``.

    To change how windows are computed, write a new function and point this
    one at it. Never edit :func:`_context_window_v2`: schema migration 2
    backfilled existing stores with it.
    """
    return _context_window_v2(text, start, end, radius=radius, max_chars=max_chars)


def _context_window_v2(
    text: str, start: int, end: int, *, radius: int = 1, max_chars: int = 600,
) -> tuple[int, int]:
    """Frozen: the windowing schema migration 2 wrote. See context_window."""
    if not 0 <= start <= end <= len(text):
        return start, end
    para_start = text.rfind("\n\n", 0, start)
    para_start = 0 if para_start < 0 else para_start + 2
    para_end = text.find("\n\n", end)
    para_end = len(text) if para_end < 0 else para_end
    para_start = min(para_start, start)
    para_end = max(para_end, end)

    # Sentence start offsets within the paragraph.
    starts = [para_start] + [
        m.end() for m in _SENTENCE_BREAK.finditer(text, para_start, para_end)
    ]
    first = max(i for i, s in enumerate(starts) if s <= start)
    last = max(i for i, s in enumerate(starts) if s < end) if end > start else first
    lo = starts[max(0, first - radius)]
    hi_idx = last + radius + 1
    hi = starts[hi_idx] if hi_idx < len(starts) else para_end

    budget = max(0, max_chars - (end - start))
    left = budget // 2
    lo = max(lo, start - left)
    hi = min(hi, end + (budget - left))

    while lo < start and text[lo].isspace():
        lo += 1
    while hi > end and text[hi - 1].isspace():
        hi -= 1
    return lo, hi


# ---------- fidelity ----------

@dataclass(frozen=True)
class FidelityIssue:
    kind: str      # number | date | unit | negation | entity
    value: str
    message: str
    direction: str | None = None  # negation only: added | dropped

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "value": self.value, "message": self.message}
        if self.direction:
            d["direction"] = self.direction
        return d


_MONTHS = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    # Italian
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5,
    "giugno": 6, "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10,
    "novembre": 11, "dicembre": 12,
}
# "may" is too common a word to treat as a month on its own.
_MONTH_RE = re.compile(
    r"\b(" + "|".join(m for m in _MONTHS if m != "may") + r")\b", re.IGNORECASE,
)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_DMY_DATE = re.compile(r"(?<!\d)(\d{1,2})[/.](\d{1,2})[/.](\d{4})(?!\d)")

# Longest alternatives first so "per cent" wins over "per".
_UNITS = {
    "%": "pct", "percent": "pct", "per cent": "pct", "per cento": "pct",
    "percento": "pct",
    "km": "km", "kilometers": "km", "kilometres": "km", "chilometri": "km",
    "mi": "mi", "mile": "mi", "miles": "mi",
    "kg": "kg", "kilograms": "kg", "chili": "kg",
    "lb": "lb", "lbs": "lb", "pounds": "lb",
    "g": "g", "grams": "g", "mg": "mg",
    "ms": "ms", "milliseconds": "ms",
    "s": "s", "sec": "s", "seconds": "s", "secondi": "s",
    "min": "min", "minutes": "min", "minuti": "min",
    "h": "h", "hours": "h", "ore": "h",
    "kwh": "kwh", "kw": "kw", "w": "w",
    "mb": "mb", "gb": "gb", "tb": "tb",
    "day": "day", "days": "day", "giorni": "day", "giorno": "day",
    "week": "week", "weeks": "week", "settimane": "week",
    "month": "month", "months": "month", "mesi": "month", "mese": "month",
    "year": "year", "years": "year", "anni": "year", "anno": "year",
    "usd": "usd", "eur": "eur", "euro": "eur",
}
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
# Scale words multiply the number: "5 million" is 5000000, not 5.
_SCALES = {
    "thousand": 10**3, "k": 10**3, "mila": 10**3,
    "million": 10**6, "millions": 10**6, "milione": 10**6, "milioni": 10**6,
    "billion": 10**9, "billions": 10**9, "bn": 10**9, "miliardo": 10**9,
    "miliardi": 10**9, "trillion": 10**12,
}
_SCALE_ALT = "|".join(sorted(_SCALES, key=len, reverse=True))
_NUMBER = re.compile(
    r"(?<![\w.,])(\d+(?:[.,]\d+)*)"
    r"(?:\s?(" + _SCALE_ALT + r")(?![\w]))?"
    r"(?:\s?(" + _UNIT_ALT + r")(?![\w]))?",
    re.IGNORECASE,
)
# Tokens mixing letters and digits (v2, GPT-4, B2B) are identifiers.
_IDENTIFIER = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)[\w]+(?:-\w+)*\b")

_NEGATION_WORDS = (
    r"not|never|none|neither|nor|without|cannot|"
    r"non|mai|nessun[oa]?|né|senza|neppure|nemmeno"
)
# "no" but not the abbreviation "No. 5"; contractions with either apostrophe.
_NEGATION = re.compile(
    rf"\b(?:{_NEGATION_WORDS})\b|\bno\b(?!\.?\s*\d)|n['’]t\b", re.IGNORECASE,
)
_TOKEN = re.compile(r"[\w'’]+")
_AUXILIARIES = {
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "has", "have", "had", "will", "would", "can", "could", "may", "might",
    "shall", "should", "must", "longer", "è", "sono", "era", "erano", "ha",
    "hanno", "viene", "vengono", "si",
}

_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "in", "on",
    "at", "of", "for", "by", "to", "and", "or", "but", "if", "when", "as",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "da", "del",
    "della", "dei", "nel", "nella", "per", "con", "e", "o", "se", "che",
}
_CAPITALIZED = re.compile(r"\b[A-ZÀ-Ý][\w'’-]*")


def _canon(digits: str, scale: int) -> str:
    return format((Decimal(digits) * scale).normalize(), "f")


def _number_values(raw: str, scale: int = 1) -> set[str]:
    """The values a written number can mean, canonicalized. A separator is
    dropped only when it groups thousands; otherwise it is a decimal point
    (so 9.0 stays 9, never 90). "2,500" is ambiguous (en 2500, it 2.5) and
    yields both."""
    seps = [c for c in raw if c in ".,"]
    if not seps:
        return {_canon(raw, scale)}
    if len(set(seps)) == 2:  # 1,234.5 or 1.234,5: the last separator is decimal
        dec = raw[max(raw.rfind("."), raw.rfind(","))]
        thou = "," if dec == "." else "."
        return {_canon(raw.replace(thou, "").replace(dec, "."), scale)}
    sep = seps[0]
    head, *groups = raw.split(sep)
    grouped = len(head) <= 3 and all(len(g) == 3 for g in groups)
    if len(groups) > 1:  # 1,000,000: only a thousands grouping reads as a number
        return {_canon(raw.replace(sep, ""), scale)} if grouped else {raw}
    values = {_canon(raw.replace(sep, "."), scale)}
    if grouped:
        values.add(_canon(raw.replace(sep, ""), scale))
    return values


def _strip_dates(text: str) -> tuple[str, list[tuple[str, set[tuple]]]]:
    """Remove numeric dates; return them as (as written, possible (y, m, d)).
    ISO order is unambiguous; d/m/y may also be m/d/y."""
    dates: list[tuple[str, set[tuple]]] = []

    def iso(m):
        dates.append((m[0], {(int(m[1]), int(m[2]), int(m[3]))}))
        return " "

    def dmy(m):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        dates.append((m[0], {(y, b, a), (y, a, b)}))
        return " "

    text = _ISO_DATE.sub(iso, text)
    text = _DMY_DATE.sub(dmy, text)
    return text, dates


def _numbers(text: str) -> list[tuple[str, set[str], str | None]]:
    """(as written, possible values, canonical unit or None), dates removed."""
    out = []
    for m in _NUMBER.finditer(text):
        scale = _SCALES[m[2].lower()] if m[2] else 1
        unit = _UNITS.get(m[3].lower()) if m[3] else None
        written = m[1] + (f" {m[2]}" if m[2] else "")
        out.append((written, _number_values(m[1], scale), unit))
    return out


def _is_negation(token: str) -> bool:
    return bool(_NEGATION.fullmatch(token)) or bool(re.search(r"n['’]t$", token))


def _negated_heads(text: str) -> list[str]:
    """The word each negation in ``text`` applies to: the next token that
    isn't an auxiliary ("is not *safe*", "did not *improve*")."""
    toks = [t.lower() for t in _TOKEN.findall(text)]
    heads = []
    for i, t in enumerate(toks):
        if not _is_negation(t):
            continue
        for nxt in toks[i + 1:i + 4]:
            if nxt not in _AUXILIARIES and not _is_negation(nxt):
                heads.append(nxt)
                break
    return heads


def _stem(word: str) -> str:
    return word[:max(4, len(word) - 2)]


def _mentions(text: str, head: str) -> bool:
    stem = _stem(head)
    return any(t.lower().startswith(stem) for t in _TOKEN.findall(text))


_CLAUSE_TOKEN = re.compile(r"[\w'’]+|[;:.!?,]")


def _negation_near(text: str, head: str, window: int = 3) -> bool:
    """Does a negation precede a form of ``head`` within ``window`` tokens of
    the same clause?"""
    stem = _stem(head)
    toks = [t.lower() for t in _CLAUSE_TOKEN.findall(text)]
    for i, tok in enumerate(toks):
        if not tok.startswith(stem):
            continue
        for prev in reversed(toks[max(0, i - window):i]):
            if prev in ";:.!?,":
                break
            if _is_negation(prev):
                return True
    return False


def _is_year(raw: str) -> bool:
    return raw.isdigit() and len(raw) == 4 and 1000 <= int(raw) <= 2099


def check_fidelity(claim: str, span: str, context: str = "") -> list[FidelityIssue]:
    """Issues where ``claim`` states something ``span`` + ``context`` do not.

    ``claim`` is the claim's proposition, or its triple rendered as text.
    """
    issues: list[FidelityIssue] = []
    evidence = f"{span}\n{context}" if context else span
    evidence_lower = evidence.lower()

    claim_rest, claim_dates = _strip_dates(claim)
    evidence_rest, evidence_dates = _strip_dates(evidence)

    ev_date_values = set().union(*(v for _, v in evidence_dates))
    for written, values in claim_dates:
        if not values & ev_date_values:
            issues.append(FidelityIssue(
                "date", written, f"date {written} not found in span or context"))

    for m in _MONTH_RE.finditer(claim_rest):
        if not re.search(rf"\b{m[1]}\b", evidence_lower, re.IGNORECASE):
            issues.append(FidelityIssue(
                "date", m[1], f"month {m[1]!r} not found in span or context"))

    ev_units: dict[str, set] = {}
    for _written, values, unit in _numbers(evidence_rest):
        for v in values:
            ev_units.setdefault(v, set()).add(unit)
    for raw, values, unit in _numbers(claim_rest):
        found = set().union(*(ev_units.get(v, set()) for v in values))
        if not any(v in ev_units for v in values):
            kind = "date" if _is_year(raw) else "number"
            issues.append(FidelityIssue(kind, raw, f"{raw} not found in span or context"))
        elif unit is not None and None not in found and unit not in found:
            issues.append(FidelityIssue(
                "unit", f"{raw} {unit}",
                f"{raw} appears with unit {sorted(u for u in found if u)} in the "
                f"source, not {unit}"))

    # Added negation is checked locally: some negation elsewhere in the
    # evidence ("Y is not tested; X is safe") doesn't license "X is not safe".
    # A negated word the evidence doesn't contain can't be compared locally
    # (the predicate may say "non si applica" where the span says "esclusa");
    # then any negation in the evidence counts.
    claim_neg = bool(_NEGATION.search(claim))
    heads = [h for h in _negated_heads(claim) if _mentions(evidence, h)]
    unmatched = [h for h in heads if not _negation_near(evidence, h)]
    if claim_neg and (unmatched if heads else not _NEGATION.search(evidence)):
        issues.append(FidelityIssue(
            "negation", unmatched[0] if unmatched else _NEGATION.search(claim)[0],
            "claim negates something the span and context don't", "added"))
    elif not claim_neg and _NEGATION.search(span):
        issues.append(FidelityIssue(
            "negation", _NEGATION.search(span)[0],
            "span is negated but the claim is not", "dropped"))

    seen: set[str] = set()
    for m in _IDENTIFIER.finditer(claim):
        tok = m[0]
        found = re.search(rf"(?<!\w){re.escape(tok.lower())}(?!\w)", evidence_lower)
        if tok.lower() not in seen and not found:
            seen.add(tok.lower())
            issues.append(FidelityIssue(
                "entity", tok, f"identifier {tok!r} not found in span or context"))
    for m in _CAPITALIZED.finditer(claim):
        tok = re.sub(r"['’]s$", "", m[0])
        low = tok.lower()
        if low in _STOPWORDS or low in _MONTHS or low in seen:
            continue
        if not re.search(rf"(?<!\w){re.escape(low)}(?!\w)", evidence_lower):
            seen.add(low)
            issues.append(FidelityIssue(
                "entity", tok, f"entity {tok!r} not found in span or context"))
    return issues
