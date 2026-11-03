from sentence_transformers import SentenceTransformers

Model_NAME = "BAAI/bge-small-en-v1.5"

def generate_embeddings(chunks):
        """
    Generate numerical embeddings for document chunks.

    Args:
        chunks: List of dictionaries containing chunk information.

    Returns:
        List of chunks with their corresponding embeddings.
    """
        if not chunks:
            raise ValueError("Chunks list is empty")

        model = SentenceTransformers(Model_NAME)

        texts = [chunk['text'] for chunk in chunks]

        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):

              embedded_chunk = {
                    **chunk,
                    "embedding": embedding
              }

        embedded_chunks.append(embedded_chunk)

        return embedded_chunks