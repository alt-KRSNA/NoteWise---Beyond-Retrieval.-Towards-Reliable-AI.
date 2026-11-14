
import pytest

from app.generation.citation_validator import validate_citations


@pytest.fixture
def retrieved_chunks():
    """Sample retrieved chunks for citation validation tests."""

    return [
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "page_number": 5,
        },
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-2",
            "page_number": 11,
        },
    ]


def test_valid_citation(retrieved_chunks):
    answer = "Machine learning learns from data. [Source 1, Page 5]"

    result = validate_citations(answer, retrieved_chunks)

    assert result["has_citations"] is True
    assert result["total_citations"] == 1
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 0


def test_wrong_page_number(retrieved_chunks):
    answer = "Machine learning learns from data. [Source 1, Page 20]"

    result = validate_citations(answer, retrieved_chunks)

    assert result["invalid_count"] == 1
    assert result["citations"][0]["valid"] is False
    assert "page does not match" in result["citations"][0]["reason"].lower()


def test_nonexistent_source(retrieved_chunks):
    answer = "Some information. [Source 5, Page 10]"

    result = validate_citations(answer, retrieved_chunks)

    assert result["invalid_count"] == 1
    assert result["citations"][0]["valid"] is False
    assert "does not exist" in result["citations"][0]["reason"]


def test_multiple_valid_citations(retrieved_chunks):
    answer = (
        "First explanation. [Source 1, Page 5] "
        "Second explanation. [Source 2, Page 11]"
    )

    result = validate_citations(answer, retrieved_chunks)

    assert result["total_citations"] == 2
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 0


def test_mixed_valid_and_invalid_citations(retrieved_chunks):
    answer = (
        "First explanation. [Source 1, Page 5] "
        "Second explanation. [Source 2, Page 20]"
    )

    result = validate_citations(answer, retrieved_chunks)

    assert result["total_citations"] == 2
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1


def test_missing_citations(retrieved_chunks):
    answer = "Machine learning is a method of learning from data."

    result = validate_citations(answer, retrieved_chunks)

    assert result["has_citations"] is False
    assert result["total_citations"] == 0
    assert result["valid_count"] == 0
    assert result["invalid_count"] == 0


def test_empty_answer(retrieved_chunks):
    with pytest.raises(ValueError):
        validate_citations("", retrieved_chunks)


def test_invalid_answer_type(retrieved_chunks):
    with pytest.raises(ValueError):
        validate_citations(None, retrieved_chunks)


def test_invalid_chunks_type():
    with pytest.raises(ValueError):
        validate_citations(
            "Answer [Source 1, Page 5]",
            None
        )


def test_empty_retrieved_chunks():
    answer = "Answer [Source 1, Page 5]"

    result = validate_citations(answer, [])

    assert result["invalid_count"] == 1
    assert result["citations"][0]["valid"] is False
