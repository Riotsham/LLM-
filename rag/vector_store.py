from __future__ import annotations

from typing import Tuple

import faiss
import numpy as np

from .embeddings import embed_texts


def build_index(docs: list[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Returns (index, embeddings)."""
    embeddings = embed_texts(docs)
    if embeddings.size == 0:
        raise ValueError("No documents provided for indexing.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def search(index: faiss.IndexFlatIP, query_vector: np.ndarray, k: int = 3) -> list[int]:
    """Returns indices of top-k matches."""
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    _, indices = index.search(query_vector, k)
    return [int(i) for i in indices[0] if i >= 0]
