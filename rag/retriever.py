from __future__ import annotations

from pathlib import Path

import numpy as np

from .embeddings import embed_texts
from .vector_store import build_index, search

RAG_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = RAG_DIR / "knowledge.txt"

_docs: list[str] | None = None
_index = None


def _load_docs() -> list[str]:
    raw = KNOWLEDGE_PATH.read_text(encoding="ascii", errors="ignore")
    docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
    return docs


def _ensure_index() -> None:
    global _docs, _index
    if _docs is not None and _index is not None:
        return

    _docs = _load_docs()
    _index, _ = build_index(_docs)


def retrieve(query: str, k: int = 2) -> str:
    """Returns concatenated top-k coping strategies."""
    _ensure_index()

    if not query or not query.strip():
        return ""

    query_vec = embed_texts([query])
    if query_vec.size == 0:
        return ""

    idxs = search(_index, query_vec, k=k)
    results = [_docs[i] for i in idxs if 0 <= i < len(_docs)]
    return "\n\n".join(results)
