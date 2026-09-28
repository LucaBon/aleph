"""Predicate alias resolution (WS-E.1).

Domain-scoped equivalence between surface predicates. The verbatim
``claim.predicate`` is never modified by reads; resolution happens at
retrieval and contradiction-pre-filter time. Mutation goes through
``Store.add_predicate_alias``, which rewrites existing claims in the same
transaction as the alias-row insert (atomic, cache-invalidating,
concept-staleness-cascading).

This module is the import surface for the rest of the codebase. The state
lives in the SQLite ``predicate_aliases`` table.
"""
from __future__ import annotations

from typing import Optional

from .db import Store


def resolve(
    store: Store, predicate: str, *, domain: Optional[str] = None,
) -> str:
    """Thin wrapper around :meth:`Store.resolve_predicate`."""
    return store.resolve_predicate(predicate, domain=domain)


def expand_keywords(
    store: Store, keywords: list[str], *, domain: Optional[str] = None,
) -> list[str]:
    """Augment a keyword list with all surface predicates that canonicalize
    to any of those keywords in the given domain.

    Used by retrievers when ``expand_aliases=True`` so a search for
    "enhances" surfaces claims whose verbatim predicate is "improves" (after
    the alias ``improves -> enhances`` is in place). Deduplicates; preserves
    input order; appends discovered surface forms to the end.
    """
    if not keywords:
        return list(keywords)
    seen: set[str] = set()
    ordered: list[str] = []
    for kw in keywords:
        if kw in seen:
            continue
        seen.add(kw)
        ordered.append(kw)
    # For each existing keyword, find every alias_from that resolves to it in
    # the requested domain (with `*` fallback). Use the same chain walk as
    # ``Store.resolve_predicate`` so transitive aliases are caught.
    canonical_targets: dict[str, list[str]] = {}
    for kw in list(ordered):
        norm = store.normalize_subject(kw)
        canonical_targets.setdefault(norm, []).append(kw)
    if domain:
        rows = store.conn.execute(
            "SELECT alias_from, canonical_to, domain FROM predicate_aliases "
            "WHERE domain = ? OR domain = '*'",
            (domain,),
        ).fetchall()
    else:
        rows = store.conn.execute(
            "SELECT alias_from, canonical_to, domain FROM predicate_aliases "
            "WHERE domain = '*'",
        ).fetchall()
    # walk: alias_from resolves transitively; if its terminal canonical hits
    # one of our keywords, surface alias_from too.
    for row in rows:
        af = row["alias_from"]
        if af in seen:
            continue
        terminal = store.resolve_predicate(af, domain=domain)
        if terminal in canonical_targets:
            seen.add(af)
            ordered.append(af)
    return ordered
