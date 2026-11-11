
import sqlite3
import tempfile
import unittest

from pathlib import Path

from app.storage.sqlite_store import (
    initialize_database,
    save_document,
    save_chunks,
    get_chunk,
    get_document_chunks,
    get_or_create_faiss_ids,
    get_chunks_by_faiss_ids,
)


class TestSQLiteStorage(unittest.TestCase):

    def setUp(self):
        """Prepare a fresh temporary database for every test."""

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

        initialize_database(self.db_path)

        self.document_id = save_document(
            filename="AI_Sample.pdf",
            file_path="data/raw/AI_Sample.pdf",
            db_path=self.db_path
        )

        self.sample_chunks = [
            {
                "chunk_id": 0,
                "page_number": 1,
                "text": "Introduction to Artificial Intelligence.",
                "token_count": 8
            },
            {
                "chunk_id": 1,
                "page_number": 1,
                "text": "Machine learning is a subset of AI.",
                "token_count": 9
            }
        ]

    def tearDown(self):
        """Remove temporary database files."""

        self.temp_dir.cleanup()

    def test_save_chunks(self):

        count = save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        self.assertEqual(count, 2)

    def test_save_empty_chunks(self):

        count = save_chunks(
            self.document_id,
            [],
            self.db_path
        )

        self.assertEqual(count, 0)

    def test_retrieve_chunk(self):

        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        chunk = get_chunk(
            self.document_id,
            0,
            self.db_path
        )

        self.assertIsNotNone(chunk)

        self.assertEqual(
            chunk["text"],
            "Introduction to Artificial Intelligence."
        )

        self.assertEqual(chunk["chunk_id"], "0")

    def test_retrieve_all_chunks(self):

        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        chunks = get_document_chunks(
            self.document_id,
            self.db_path
        )

        self.assertEqual(len(chunks), 2)

        self.assertEqual(chunks[0]["chunk_id"], "0")
        self.assertEqual(chunks[1]["chunk_id"], "1")

    def test_document_isolation(self):

        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        chunks = get_document_chunks(
            "non_existing_document",
            self.db_path
        )

        self.assertEqual(chunks, [])

    def test_update_existing_chunk(self):

        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        updated_chunks = [{
            **self.sample_chunks[0],
            "text": "Updated AI introduction."
        }]

        save_chunks(
            self.document_id,
            updated_chunks,
            self.db_path
        )

        chunk = get_chunk(
            self.document_id,
            0,
            self.db_path
        )

        self.assertEqual(
            chunk["text"],
            "Updated AI introduction."
        )

    def test_invalid_document_reference(self):

        with self.assertRaises(sqlite3.IntegrityError):

            save_chunks(
                "invalid_document_id",
                self.sample_chunks,
                self.db_path
            )

    def test_retrieval_output_consistency(self):

        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        single_chunk = get_chunk(
            self.document_id,
            0,
            self.db_path
        )

        all_chunks = get_document_chunks(
            self.document_id,
            self.db_path
        )

        self.assertEqual(
            set(single_chunk.keys()),
            set(all_chunks[0].keys())
        )

    def test_faiss_id_mapping(self):

        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        mappings = get_or_create_faiss_ids(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        self.assertEqual(len(mappings), 2)

        self.assertEqual(mappings[0]["chunk_id"], "0")
        self.assertEqual(mappings[1]["chunk_id"], "1")

        self.assertIsInstance(mappings[0]["faiss_id"], int)

    def test_retrieve_chunks_using_faiss_ids(self):

        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        mappings = get_or_create_faiss_ids(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        faiss_ids = [
            mapping["faiss_id"]
            for mapping in mappings
        ]

        chunks = get_chunks_by_faiss_ids(
            faiss_ids,
            self.db_path
        )

        self.assertEqual(len(chunks), 2)

        self.assertEqual(
            chunks[0]["text"],
            "Introduction to Artificial Intelligence."
        )

        self.assertEqual(
            chunks[1]["text"],
            "Machine learning is a subset of AI."
        )

    def test_faiss_ids_are_stable(self):

        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        first_mapping = get_or_create_faiss_ids(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        second_mapping = get_or_create_faiss_ids(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        self.assertEqual(first_mapping, second_mapping)


if __name__ == "__main__":
    unittest.main()
