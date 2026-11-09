
from pathlib import Path

from app.ingestion.data_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.embeddings.embeddings import generate_embeddings

from app.storage.sqlite_store import (
    DEFAULT_DB_PATH,
    initialize_database,
    save_document,
    save_chunks,
    get_or_create_faiss_ids,
)

from app.vector_store.faiss_store import (
    FAISSVectorStore,
    DEFAULT_INDEX_PATH,
)


EMBEDDING_DIMENSION = 384


def index_pdf(
    pdf_path,
    db_path=DEFAULT_DB_PATH,
    index_path=DEFAULT_INDEX_PATH,
    chunk_size=200,
    overlap=30,
):
    """Extract, embed and index a PDF document."""

    pdf_path = Path(pdf_path)

    # Validate PDF path
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    # Step 1: Initialize SQLite
    initialize_database(db_path)

    # Step 2: Extract PDF text
    document = load_pdf(str(pdf_path))

    # Step 3: Generate chunks
    chunks = chunk_pages(
        document["pages"],
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError("No text chunks were generated.")

    print(f"\nTotal chunks: {len(chunks)}")

    # Step 4: Generate embeddings
    embedded_chunks = generate_embeddings(chunks)

    print(f"Total embeddings: {len(embedded_chunks)}")

    # Step 5: Save document in SQLite
    document_id = save_document(
        filename=pdf_path.name,
        file_path=str(pdf_path.resolve()),
        db_path=db_path,
    )

    # Step 6: Save original chunks in SQLite
    save_chunks(
        document_id,
        chunks,
        db_path,
    )

    # Step 7: Generate stable FAISS IDs
    mappings = get_or_create_faiss_ids(
        document_id,
        chunks,
        db_path,
    )

    id_by_chunk = {
        mapping["chunk_id"]: mapping["faiss_id"]
        for mapping in mappings
    }

    # Step 8: Attach FAISS IDs to embeddings
    faiss_chunks = []

    for chunk in embedded_chunks:
        faiss_chunks.append({
            **chunk,
            "faiss_id": id_by_chunk[str(chunk["chunk_id"])],
        })

    # Step 9: Load existing index or create a new one
    index_path = Path(index_path)

    if index_path.exists():
        vector_store = FAISSVectorStore.load(index_path)
    else:
        vector_store = FAISSVectorStore(
            dimension=EMBEDDING_DIMENSION
        )

    # Step 10: Avoid duplicate vector insertion
    existing_ids = vector_store.get_ids()

    new_chunks = [
        chunk
        for chunk in faiss_chunks
        if chunk["faiss_id"] not in existing_ids
    ]

    added_count = vector_store.add_embeddings(new_chunks)

    # Step 11: Save FAISS index
    vector_store.save(index_path)

    print("\nIndexing completed successfully!")

    return {
        "document_id": document_id,
        "filename": pdf_path.name,
        "total_chunks": len(chunks),
        "total_embeddings": len(embedded_chunks),
        "new_vectors_indexed": added_count,
        "total_vectors_in_index": vector_store.index.ntotal,
        "index_path": str(index_path),
    }
