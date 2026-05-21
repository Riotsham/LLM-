from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns normalized float32 embeddings."""
    cleaned: List[str] = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return np.zeros((0, 384), dtype=np.float32)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        cleaned,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)
