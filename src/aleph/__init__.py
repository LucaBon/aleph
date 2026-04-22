"""Aleph: stateless-view knowledge base for LLMs.

Core idea: atomic claims with source spans are the compounding primitive;
prose views are ephemeral, regenerable, and verified against source at read time.
"""
from .db import Store
from .llm import LLM
from .ingest import ingest_file, ingest_paths
from .query import query, QueryResult, Citation
from .lint import lint, resolve_by_recency

__version__ = "0.1.0"
__all__ = [
    "Store", "LLM",
    "ingest_file", "ingest_paths",
    "query", "QueryResult", "Citation",
    "lint", "resolve_by_recency",
]
