"""
FAISS Vector Index
In-memory flat L2 index rebuilt from DB on startup.
Supports add, search, and rebuild operations.
"""
import numpy as np
import json
from typing import Optional

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

from app.core.config import FINGERPRINT_DIM


class VectorIndex:
    def __init__(self, dim: int = FINGERPRINT_DIM):
        self.dim = dim
        self._segment_ids: list[str] = []
        self._index = None
        self._vectors: list[np.ndarray] = []

    def _ensure_index(self, dim: int):
        """Lazily create FAISS index on first add, using actual embedding dim."""
        if self._index is None:
            self.dim = dim
            if FAISS_OK:
                self._index = faiss.IndexFlatIP(dim)

    def add(self, segment_id: str, embedding: np.ndarray):
        vec = embedding.astype(np.float32).reshape(1, -1)
        self._ensure_index(vec.shape[1])
        self._segment_ids.append(segment_id)
        if FAISS_OK:
            self._index.add(vec)
        else:
            self._vectors.append(vec[0])

    def search(self, query: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Returns list of (segment_id, similarity_score) sorted descending."""
        if not self._segment_ids:
            return []

        q = query.astype(np.float32).reshape(1, -1)

        if FAISS_OK and self._index is not None:
            k = min(top_k, len(self._segment_ids))
            scores, indices = self._index.search(q, k)
            return [
                (self._segment_ids[int(i)], float(s))
                for s, i in zip(scores[0], indices[0])
                if i >= 0
            ]
        else:
            mat = np.stack(self._vectors)
            sims = mat @ q[0]
            top = np.argsort(sims)[::-1][:top_k]
            return [(self._segment_ids[int(i)], float(sims[i])) for i in top]

    def rebuild(self, entries: list[tuple[str, np.ndarray]]):
        """Rebuild index from (segment_id, embedding) pairs."""
        self._segment_ids = []
        self._index = None
        self._vectors = []
        for seg_id, emb in entries:
            self.add(seg_id, emb)

    @property
    def size(self) -> int:
        return len(self._segment_ids)


# Global singleton
_index = VectorIndex()


def get_index() -> VectorIndex:
    return _index
