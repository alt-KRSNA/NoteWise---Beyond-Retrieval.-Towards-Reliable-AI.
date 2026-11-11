
import unittest
import numpy as np

from app.ingestion.data_loader import load_pdf
from app.ingestion.chunker import chunk_pages

from app.embeddings.embeddings import (
    generate_embeddings,
    generate_query_embedding
)


class TestEmbeddings(unittest.TestCase):

    def test_embedding_generation(self):

        # Step 1: Load PDF
        document = load_pdf("AI_Sample.pdf")

        # Step 2: Generate chunks
        chunks = chunk_pages(
            document["pages"],
            chunk_size=200,
            overlap=30
        )

        print("\nTotal chunks:", len(chunks))

        self.assertGreater(len(chunks), 0)

        # Step 3: Generate embeddings
        embedded_chunks = generate_embeddings(chunks)

        # Step 4: Verify embedding results
        print("\nTotal embedded chunks:", len(embedded_chunks))

        self.assertEqual(len(embedded_chunks), len(chunks))

        # Inspect first embedding
        first_chunk = embedded_chunks[0]

        print("\nFirst chunk ID:", first_chunk["chunk_id"])
        print("Page number:", first_chunk["page_number"])
        print("Embedding dimensions:", first_chunk["embedding"].shape)
        print("Token count:", first_chunk["token_count"])

        # Step 5: Assertions
        for chunk in embedded_chunks:

            self.assertIsNotNone(chunk["embedding"])

            self.assertEqual(
                chunk["embedding"].shape,
                (384,)
            )

        print("\nAll embedding checks passed!")

    def test_query_embedding_generation(self):

        query = "What is machine learning?"

        embedding = generate_query_embedding(query)

        self.assertEqual(embedding.shape, (384,))

        self.assertTrue(np.isfinite(embedding).all())

        self.assertAlmostEqual(
            float(np.linalg.norm(embedding)),
            1.0,
            places=5
        )

        print("\nQuery embedding checks passed!")


if __name__ == "__main__":
    unittest.main()
