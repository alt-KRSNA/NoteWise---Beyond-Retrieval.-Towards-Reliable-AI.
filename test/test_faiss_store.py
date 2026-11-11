
import tempfile
import unittest

from pathlib import Path

import numpy as np

from app.vector_store.faiss_store import FAISSVectorStore


class TestFAISSVectorStore(unittest.TestCase):

    def setUp(self):
        self.dimension = 384
        self.store = FAISSVectorStore(self.dimension)

        self.sample_chunks = [
            {
                "faiss_id": 1,
                "embedding": np.eye(1, self.dimension, 0)[0]
            },
            {
                "faiss_id": 2,
                "embedding": np.eye(1, self.dimension, 1)[0]
            },
            {
                "faiss_id": 3,
                "embedding": np.eye(1, self.dimension, 2)[0]
            }
        ]

        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_path = Path(self.temp_dir.name) / "test.index"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_embeddings(self):
        count = self.store.add_embeddings(self.sample_chunks)

        self.assertEqual(count, 3)
        self.assertEqual(self.store.index.ntotal, 3)

    def test_search(self):
        self.store.add_embeddings(self.sample_chunks)

        query = np.zeros(self.dimension, dtype=np.float32)
        query[0] = 1.0

        results = self.store.search(query, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["faiss_id"], 1)

    def test_empty_search(self):
        results = self.store.search(
            np.zeros(self.dimension, dtype=np.float32)
        )

        self.assertEqual(results, [])

    def test_empty_embeddings(self):
        count = self.store.add_embeddings([])

        self.assertEqual(count, 0)

    def test_save_and_load(self):
        self.store.add_embeddings(self.sample_chunks)
        self.store.save(self.index_path)

        loaded_store = FAISSVectorStore.load(self.index_path)

        self.assertEqual(loaded_store.index.ntotal, 3)

    def test_duplicate_ids(self):
        duplicate_chunks = [
            self.sample_chunks[0],
            self.sample_chunks[0]
        ]

        with self.assertRaises(ValueError):
            self.store.add_embeddings(duplicate_chunks)

    def test_invalid_dimension(self):
        invalid_chunk = [{
            "faiss_id": 4,
            "embedding": np.zeros(128)
        }]

        with self.assertRaises(ValueError):
            self.store.add_embeddings(invalid_chunk)


if __name__ == "__main__":
    unittest.main()
