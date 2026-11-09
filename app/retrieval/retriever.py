
from app.embeddings.embeddings import generate_query_embedding

from app.vector_store.faiss_store import (
    FAISSVectorStore,
    DEFAULT_INDEX_PATH,
)

from app.storage.sqlite_store import (
    DEFAULT_DB_PATH,
    get_chunks_by_faiss_ids,
)


def retrieve(
    query,
    top_k=5,
    db_path=DEFAULT_DB_PATH,
    index_path=DEFAULT_INDEX_PATH,
):
    """Retrieve ranked evidence chunks for a user query."""

    # Validate top_k
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # 1. Generate query embedding
    query_embedding = generate_query_embedding(query)

    # 2. Load persisted FAISS index
    vector_store = FAISSVectorStore.load(index_path)

    # 3. Search for similar vectors
    search_results = vector_store.search(
        query_embedding,
        top_k=top_k,
    )

    if not search_results:
        return []

    # 4. Extract FAISS IDs and scores
    faiss_ids = [
        result["faiss_id"]
        for result in search_results
    ]

    scores = {
        result["faiss_id"]: result["score"]
        for result in search_results
    }

    # 5. Retrieve original chunks from SQLite
    chunks = get_chunks_by_faiss_ids(
        faiss_ids,
        db_path=db_path,
    )

    # 6. Check SQLite and FAISS consistency
    if len(chunks) != len(faiss_ids):
        raise RuntimeError(
            "Retrieval integrity error: some FAISS IDs "
            "have no corresponding SQLite chunks."
        )

    # 7. Attach similarity scores
    for chunk in chunks:
        chunk["score"] = scores[chunk["faiss_id"]]

    return chunks
