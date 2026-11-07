
from pathlib import Path

import faiss
import numpy as np


DEFAULT_INDEX_PATH = Path("database/faiss.index")


class FAISSVectorStore:
    """Manage vector indexing, similarity search and persistence."""

    def __init__(self, dimension=384):
        self.dimension = dimension

        # Exact inner-product similarity index.
        base_index = faiss.IndexFlatIP(dimension)

        # Allows custom IDs instead of relying on positional IDs.
        self.index = faiss.IndexIDMap2(base_index)

    def add_embeddings(self, embedded_chunks):
        """Add normalized chunk embeddings to the FAISS index."""

        if not embedded_chunks:
            return 0

        vectors = np.asarray(
            [chunk["embedding"] for chunk in embedded_chunks],
            dtype=np.float32
        )

        ids = np.asarray(
            [chunk["faiss_id"] for chunk in embedded_chunks],
            dtype=np.int64
        )

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors with dimension {self.dimension}, "
                f"received shape {vectors.shape}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        if len(ids) != len(vectors):
            raise ValueError("Number of IDs and embeddings must match.")

        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate FAISS IDs in input.")

        if np.any(ids < 0):
            raise ValueError("FAISS IDs must be non-negative.")

        if len(set(ids.tolist()).intersection(
            self._existing_ids()
        )):
            raise ValueError("Some FAISS IDs already exist in the index.")

        self.index.add_with_ids(vectors, ids)

        return len(vectors)

    def search(self, query_embedding, top_k=5):
        """Return the most similar vector IDs and similarity scores."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.shape != (1, self.dimension):
            raise ValueError(
                f"Expected query shape (1, {self.dimension}), "
                f"received {query_vector.shape}"
            )

        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains invalid values.")

        scores, ids = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for faiss_id, score in zip(ids[0], scores[0]):
            if faiss_id == -1:
                continue

            results.append({
                "faiss_id": int(faiss_id),
                "score": float(score)
            })

        return results

    def save(self, index_path=DEFAULT_INDEX_PATH):
        """Persist the FAISS index to disk."""

        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

    @classmethod
    def load(cls, index_path=DEFAULT_INDEX_PATH):
        """Load an existing FAISS index from disk."""

        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        loaded_index = faiss.read_index(str(index_path))

        store = cls(dimension=loaded_index.d)
        store.index = loaded_index

        return store

    def _existing_ids(self):
        """Get IDs already present in the index."""

        if self.index.ntotal == 0:
            return set()

        return set(
            faiss.vector_to_array(
                self.index.id_map
            ).tolist()
        )
