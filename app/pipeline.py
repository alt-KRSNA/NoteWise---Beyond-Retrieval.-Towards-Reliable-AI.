
from app.retrieval.retriever import retrieve
from app.generation.generator import generate_response
from app.generation.citation_validator import validate_citations

from app.storage.sqlite_store import DEFAULT_DB_PATH
from app.vector_store.faiss_store import DEFAULT_INDEX_PATH


def answer_question(
    query,
    top_k=5,
    db_path=DEFAULT_DB_PATH,
    index_path=DEFAULT_INDEX_PATH,
):
    """
    Retrieve relevant evidence, generate a grounded answer,
    and validate its source and page citations.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # 1. Retrieve relevant chunks
    chunks = retrieve(
        query,
        top_k=top_k,
        db_path=db_path,
        index_path=index_path,
    )

    # 2. Handle the case where no evidence is retrieved
    if not chunks:
        answer = (
            "I could not find relevant evidence "
            "in the available documents."
        )

        sources = []

    else:
        # 3. Prepare evidence context
        context_parts = []

        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"""
[Source {i}]
Document ID: {chunk['document_id']}
Page: {chunk['page_number']}
Evidence: {chunk['text']}
"""
            )

        context = "\n".join(context_parts)

        # 4. Construct grounded prompt
        prompt = f"""
You are NoteWise, an evidence-based AI research assistant.

Answer the user's question using ONLY the evidence provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or citations.
3. If the evidence is insufficient, clearly say so.
4. Cite supporting sources using [Source X, Page Y].
5. Keep the answer clear and relevant.

EVIDENCE:
{context}

USER QUESTION:
{query}

ANSWER:
"""

        # 5. Generate answer
        answer = generate_response(prompt, max_tokens=800)

        # 6. Prepare source metadata
        sources = [
            {
                "document_id": chunk["document_id"],
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "faiss_id": chunk["faiss_id"],
                "score": chunk["score"],
            }
            for chunk in chunks
        ]

    # 7. Validate citations against retrieved chunks
    citation_validation = validate_citations(
        answer,
        chunks,
    )

    # 8. Return answer, sources, and citation validation
    return {
        "answer": answer,
        "sources": sources,
        "citation_validation": citation_validation,
    }
