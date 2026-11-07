
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_embedding_model():
    """Load the embedding model once and reuse it."""

    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def generate_embeddings(chunks):
    """Generate embeddings for document chunks."""

    if not chunks:
        raise ValueError("Chunks list is empty")

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):

        embedded_chunks.append({
            **chunk,
            "embedding": embedding
        })

    return embedded_chunks


def generate_query_embedding(query):
    """Convert a user query into a normalized embedding."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    model = get_embedding_model()

    embedding = model.encode(
        query.strip(),
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(embedding, dtype=np.float32)
