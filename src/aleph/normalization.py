"""Subject normalization: swappable per-locale singularization rules.

`normalize_subject` is called on every `add_claim` and on both sides of
`resolve_subject`, so the ruleset directly shapes which surface variants
collapse to the same canonical subject. The honesty contract: each
normalizer is deterministic (lowercase, whitespace, punctuation stripping,
then locale-specific singularization on the last word) and ships with a
conservative ruleset — prefer leaving a word alone over producing a wrong
singularization; an alias in the store is always available for edge cases.

English is the default. Italian covers the common legal/scientific
plural endings (`-zioni`, `-nti`, `-nze`, …) observed in the corpus
described in docs/improvement-proposals.md (P1.3).

Switching the locale on an existing store is a migration event: the alias
table stores its keys in normalized form under the old rules, so aliases
may stop matching. :class:`Store` refuses to change the configured locale
implicitly; callers must use the `config-set` command.
"""
from __future__ import annotations

import re
from typing import Protocol


class Normalizer(Protocol):
    """The one method every normalizer must implement."""

    locale: str

    def normalize(self, s: str) -> str:
        """Return the canonical form of ``s`` for this locale."""
        ...


def _preclean(s: str) -> list[str]:
    """Shared preprocessing: lowercase, collapse whitespace, strip
    punctuation that is never part of a subject (keep letters including
    accented, digits, hyphens, whitespace). Returns the word list; the
    caller decides how to singularize."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Drop apostrophes, periods, parentheses, quotes, etc. Preserve letters
    # (\w in unicode mode includes accented chars), digits, hyphen, space.
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    return s.split()


class EnglishNormalizer:
    """Simple English plural -> singular rules on the last word.

    Rules (applied in order on the last word):
      - ``-ies`` -> ``-y`` (``batteries`` -> ``battery``)
      - ``-sses`` -> ``-ss`` (``glasses`` -> ``glass``)
      - trailing ``-s`` unless ``-ss`` / ``-us`` (``cars`` -> ``car``;
        ``bus`` stays ``bus``).

    This is deliberately naive — it over-matches on words like ``gas``
    (``ga``) and under-matches on irregular plurals (``men``, ``feet``).
    The alias table is the escape hatch.
    """

    locale = "en"

    def normalize(self, s: str) -> str:
        words = _preclean(s)
        if not words:
            return ""
        last = words[-1]
        if len(last) > 4 and last.endswith("ies"):
            words[-1] = last[:-3] + "y"
        elif len(last) > 4 and last.endswith("sses"):
            words[-1] = last[:-2]  # glasses -> glass
        elif (
            len(last) > 3 and last.endswith("s")
            and not last.endswith("ss")
            and not last.endswith("us")
        ):
            words[-1] = last[:-1]
        return " ".join(words)


class ItalianNormalizer:
    """Conservative Italian plural -> singular rules on the last word.

    Italian pluralization depends on gender and ending of the singular
    (``-o`` -> ``-i``, ``-a`` -> ``-e``, ``-e`` -> ``-i``), so pure
    morphological rules will misfire on some words. This ruleset
    intentionally prefers correctness on the common patterns to aggressive
    coverage.

    Applied in order on the last word (longest suffix first):
      - ``-zioni`` -> ``-zione``   (``azioni`` -> ``azione``)
      - ``-sioni`` -> ``-sione``   (``espressioni`` -> ``espressione``)
      - ``-nti``   -> ``-nte``     (``aggravanti`` -> ``aggravante``)
      - ``-nze``   -> ``-nza``     (``sentenze`` -> ``sentenza``)
      - ``-zze``   -> ``-zza``     (``ragazze`` -> ``ragazza``)
      - ``-che``   -> ``-ca``      (``amiche`` -> ``amica``)
      - ``-ghe``   -> ``-ga``      (``colleghe`` -> ``collega``)
      - fallback ``-i`` -> ``-o``  (``articoli`` -> ``articolo``)

    Invariants (never modified): words ending in a stressed vowel
    (``-ità``, ``-tù``, ``-à``, ``-ù``, ``-ò``, ``-ì``, ``-è``, ``-é``),
    which in Italian are identical in singular and plural.

    ``-e`` endings are left alone: too many Italian singulars end in
    ``-e`` (``legge``, ``padre``, ``pace``) for a ``-e`` -> ``-a`` rule
    to be safe. Aliases cover the edge cases.
    """

    locale = "it"

    SUFFIX_RULES: list[tuple[str, str]] = [
        ("zioni", "zione"),
        ("sioni", "sione"),
        ("nti", "nte"),
        ("nze", "nza"),
        ("zze", "zza"),
        ("che", "ca"),
        ("ghe", "ga"),
    ]
    # Endings that mark an invariant (same in singular and plural). Stressed
    # final vowels + a few regular ``-i``-ending invariants (``crisi``,
    # ``analisi``, ``ipotesi``, ``tesi``, ``sintesi``).
    INVARIANT_SUFFIXES: tuple[str, ...] = (
        "ità", "tù", "à", "ù", "ò", "ì", "è", "é",
        "crisi", "analisi", "ipotesi", "sintesi", "tesi",
    )

    def normalize(self, s: str) -> str:
        words = _preclean(s)
        if not words:
            return ""
        last = words[-1]
        if any(last.endswith(inv) for inv in self.INVARIANT_SUFFIXES):
            words[-1] = last
            return " ".join(words)
        for suf, repl in self.SUFFIX_RULES:
            # require at least one char before the suffix, so the whole word
            # isn't the suffix ("zioni" alone would leave the stem empty).
            if last.endswith(suf) and len(last) > len(suf):
                words[-1] = last[: -len(suf)] + repl
                return " ".join(words)
        # Fallback: -i -> -o (covers masculine -o plurals, the most common
        # pattern in the corpus). -e endings are left alone: many singular
        # -e nouns exist and -e -> -a would wreck them.
        if last.endswith("i") and len(last) > 3:
            last = last[:-1] + "o"
        words[-1] = last
        return " ".join(words)


NORMALIZERS: dict[str, type] = {
    "en": EnglishNormalizer,
    "it": ItalianNormalizer,
}


def get_normalizer(locale: str) -> Normalizer:
    """Return the normalizer instance for ``locale``. Raises
    :class:`ValueError` with the list of known locales if unknown."""
    cls = NORMALIZERS.get(locale)
    if cls is None:
        raise ValueError(
            f"unknown locale {locale!r}; known: {sorted(NORMALIZERS)}"
        )
    return cls()
