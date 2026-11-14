
import re


CITATION_PATTERN = re.compile(
    r"\[Source\s+(\d+)\s*,\s*Page\s+(\d+)\]",
    re.IGNORECASE,
)


def validate_citations(answer, retrieved_chunks):
    """
    Validate source numbers and page numbers in an answer
    against the chunks retrieved for that question.

    Note:
    This validates citation references, not whether the
    cited evidence semantically supports the generated claim.
    """

    # Validate inputs
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be a non-empty string.")

    if not isinstance(retrieved_chunks, list):
        raise ValueError("Retrieved chunks must be a list.")

    # Extract citations from the generated answer
    citations = CITATION_PATTERN.findall(answer)

    results = []

    for source_number, cited_page in citations:

        source_number = int(source_number)
        cited_page = int(cited_page)

        # Check whether the source number exists
        if source_number < 1 or source_number > len(retrieved_chunks):
            results.append({
                "source_number": source_number,
                "cited_page": cited_page,
                "valid": False,
                "reason": "Source does not exist in retrieved evidence."
            })
            continue

        # Map Source X to its corresponding retrieved chunk
        chunk = retrieved_chunks[source_number - 1]

        actual_page = chunk["page_number"]

        # Check whether the cited page matches the actual page
        if cited_page != actual_page:
            results.append({
                "source_number": source_number,
                "cited_page": cited_page,
                "actual_page": actual_page,
                "valid": False,
                "reason": "Cited page does not match the retrieved chunk."
            })
            continue

        # Citation is valid
        results.append({
            "source_number": source_number,
            "cited_page": cited_page,
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "valid": True,
            "reason": "Citation matches retrieved evidence."
        })

    return {
        "has_citations": len(citations) > 0,
        "total_citations": len(results),
        "valid_count": sum(item["valid"] for item in results),
        "invalid_count": sum(not item["valid"] for item in results),
        "citations": results,
    }
