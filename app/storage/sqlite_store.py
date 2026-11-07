
from contextlib import contextmanager
import json
import sqlite3

from pathlib import Path
from uuid import uuid4


DEFAULT_DB_PATH = Path("database/notewise.db")


@contextmanager
def get_connection(db_path=DEFAULT_DB_PATH):
    """Create and safely close a SQLite connection."""

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = None

    try:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        yield connection

        connection.commit()

    except Exception:
        if connection is not None:
            connection.rollback()
        raise

    finally:
        if connection is not None:
            connection.close()


def initialize_database(db_path=DEFAULT_DB_PATH):
    """Create document, chunk and vector mapping tables."""

    with get_connection(db_path) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                page_number INTEGER,
                content TEXT NOT NULL,
                token_count INTEGER,
                metadata TEXT NOT NULL DEFAULT '{}',

                PRIMARY KEY (document_id, chunk_id),

                FOREIGN KEY (document_id)
                REFERENCES documents(id)
                ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS vector_mappings (
                faiss_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,

                UNIQUE(document_id, chunk_id),

                FOREIGN KEY (document_id, chunk_id)
                REFERENCES chunks(document_id, chunk_id)
                ON DELETE CASCADE
            )
        """)


def save_document(filename, file_path=None, db_path=DEFAULT_DB_PATH):
    """Register a document and return its unique ID."""

    document_id = str(uuid4())

    with get_connection(db_path) as connection:

        connection.execute(
            """
            INSERT INTO documents (id, filename, file_path)
            VALUES (?, ?, ?)
            """,
            (document_id, filename, file_path)
        )

    return document_id


def save_chunks(document_id, chunks, db_path=DEFAULT_DB_PATH):
    """Save or update extracted document chunks."""

    records = []

    for chunk in chunks:

        records.append((
            document_id,
            str(chunk["chunk_id"]),
            chunk.get("page_number"),
            chunk["text"],
            chunk.get("token_count"),
            json.dumps(chunk.get("metadata", {}))
        ))

    if not records:
        return 0

    with get_connection(db_path) as connection:

        connection.executemany(
            """
            INSERT INTO chunks (
                document_id,
                chunk_id,
                page_number,
                content,
                token_count,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT(document_id, chunk_id)
            DO UPDATE SET
                page_number = excluded.page_number,
                content = excluded.content,
                token_count = excluded.token_count,
                metadata = excluded.metadata
            """,
            records
        )

    return len(records)


def get_chunk(document_id, chunk_id, db_path=DEFAULT_DB_PATH):
    """Retrieve one chunk using document and chunk IDs."""

    with get_connection(db_path) as connection:

        result = connection.execute(
            """
            SELECT
                document_id,
                chunk_id,
                page_number,
                content AS text,
                token_count,
                metadata
            FROM chunks
            WHERE document_id = ? AND chunk_id = ?
            """,
            (document_id, str(chunk_id))
        ).fetchone()

    if result is None:
        return None

    chunk = dict(result)
    chunk["metadata"] = json.loads(chunk["metadata"])

    return chunk


def get_document_chunks(document_id, db_path=DEFAULT_DB_PATH):
    """Retrieve all chunks belonging to a document."""

    with get_connection(db_path) as connection:

        results = connection.execute(
            """
            SELECT
                document_id,
                chunk_id,
                page_number,
                content AS text,
                token_count,
                metadata
            FROM chunks
            WHERE document_id = ?
            ORDER BY page_number, chunk_id
            """,
            (document_id,)
        ).fetchall()

    chunks = []

    for result in results:

        chunk = dict(result)
        chunk["metadata"] = json.loads(chunk["metadata"])

        chunks.append(chunk)

    return chunks


def get_or_create_faiss_ids(document_id, chunks, db_path=DEFAULT_DB_PATH):
    """Create or retrieve stable FAISS IDs for document chunks."""

    mappings = []

    with get_connection(db_path) as connection:

        for chunk in chunks:

            chunk_id = str(chunk["chunk_id"])

            connection.execute(
                """
                INSERT OR IGNORE INTO vector_mappings (
                    document_id,
                    chunk_id
                )
                VALUES (?, ?)
                """,
                (document_id, chunk_id)
            )

            result = connection.execute(
                """
                SELECT faiss_id, chunk_id
                FROM vector_mappings
                WHERE document_id = ? AND chunk_id = ?
                """,
                (document_id, chunk_id)
            ).fetchone()

            mappings.append({
                "faiss_id": result["faiss_id"],
                "chunk_id": result["chunk_id"]
            })

    return mappings


def get_chunks_by_faiss_ids(faiss_ids, db_path=DEFAULT_DB_PATH):
    """Retrieve original chunks using their FAISS IDs."""

    if not faiss_ids:
        return []

    placeholders = ",".join("?" for _ in faiss_ids)

    with get_connection(db_path) as connection:

        results = connection.execute(
            f"""
            SELECT
                vm.faiss_id,
                c.document_id,
                c.chunk_id,
                c.page_number,
                c.content AS text,
                c.token_count,
                c.metadata
            FROM vector_mappings vm
            JOIN chunks c
                ON vm.document_id = c.document_id
                AND vm.chunk_id = c.chunk_id
            WHERE vm.faiss_id IN ({placeholders})
            """,
            [int(faiss_id) for faiss_id in faiss_ids]
        ).fetchall()

    chunks_by_id = {}

    for result in results:
        chunk = dict(result)
        chunk["metadata"] = json.loads(chunk["metadata"])
        chunks_by_id[chunk["faiss_id"]] = chunk

    return [
        chunks_by_id[faiss_id]
        for faiss_id in faiss_ids
        if faiss_id in chunks_by_id
    ]
