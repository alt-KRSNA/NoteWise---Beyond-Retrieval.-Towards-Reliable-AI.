
from transformers import AutoTokenizer


# Initial tokenizer for NoteWise
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def chunk_pages(pages, chunk_size=200, overlap=30):
    """
    Split extracted PDF pages into fixed-size token chunks.

    Args:
        pages: List of dictionaries containing page_number and text.
        chunk_size: Maximum tokens in each chunk.
        overlap: Number of tokens repeated between chunks.

    Returns:
        List of dictionaries containing chunk information.
    """

    # Validate chunking parameters
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    chunks = []
    chunk_id = 1

    # Process each page
    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        if not text or not text.strip():
            continue

        # Convert text into token IDs
        token_ids = tokenizer(text,add_special_tokens=False,truncation=False)["input_ids"]

        # Calculate how far the next chunk should move
        step_size = chunk_size - overlap

        # Create chunks using sliding window
        for start in range(0, len(token_ids), step_size):

            chunk_tokens = token_ids[start:start + chunk_size]

            # Convert token IDs back into readable text
            chunk_text = tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True
            )

            if not chunk_text.strip():
                continue

            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_number,
                "text": chunk_text,
                "token_count": len(chunk_tokens)
            })

            chunk_id += 1

    return chunks
