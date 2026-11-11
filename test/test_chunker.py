
from app.ingestion.data_loader import load_pdf
from app.ingestion.chunker import chunk_pages


def test_chunker():

    document = load_pdf("AI_Sample.pdf")

    chunks = chunk_pages(
        document["pages"],
        chunk_size=200,
        overlap=30
    )

        # 1. Total number of chunks
    print("Total chunks:", len(chunks))


    # 2. Check whether chunk IDs are unique
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    print("Unique IDs:", len(chunk_ids) == len(set(chunk_ids)))


    # 3. Check whether page numbers exist
    print(
        "Page numbers present:",
        all(chunk["page_number"] > 0 for chunk in chunks)
    )


    # 4. Check whether token count is within the limit
    print(
        "Token limit respected:",
        all(chunk["token_count"] <= 200 for chunk in chunks)
    )


    # 5. Display the first three chunks
    for chunk in chunks[:3]:
        print("\n", chunk)

    print("Total chunks:", len(chunks))

    for chunk in chunks[:3]:
        print("\n", chunk)


if __name__ == "__main__":
    test_chunker()
