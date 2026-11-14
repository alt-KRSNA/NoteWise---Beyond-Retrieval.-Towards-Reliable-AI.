
import pytest

import app.pipeline as pipeline


# --------------------------------------------------
# SAMPLE RETRIEVED EVIDENCE
# --------------------------------------------------

@pytest.fixture
def sample_chunks():
    return [
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "page_number": 5,
            "faiss_id": 1,
            "score": 0.85,
            "text": "Machine learning is a branch of AI that learns from data.",
        },
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-2",
            "page_number": 11,
            "faiss_id": 2,
            "score": 0.78,
            "text": "Machine learning algorithms identify patterns in data.",
        },
    ]


# --------------------------------------------------
# INTEGRATION TESTS
# --------------------------------------------------

def test_pipeline_with_valid_citations(monkeypatch, sample_chunks):
    """Test answer generation and successful citation validation."""

    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda *args, **kwargs: sample_chunks,
    )

    monkeypatch.setattr(
        pipeline,
        "generate_response",
        lambda *args, **kwargs: (
            "Machine learning learns from data. [Source 1, Page 5]"
        ),
    )

    result = pipeline.answer_question("What is machine learning?")

    assert "answer" in result
    assert "sources" in result
    assert "citation_validation" in result

    assert result["citation_validation"]["has_citations"] is True
    assert result["citation_validation"]["valid_count"] == 1
    assert result["citation_validation"]["invalid_count"] == 0

    assert len(result["sources"]) == 2


def test_pipeline_with_invalid_citation(monkeypatch, sample_chunks):
    """Test whether the pipeline detects an incorrect page citation."""

    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda *args, **kwargs: sample_chunks,
    )

    monkeypatch.setattr(
        pipeline,
        "generate_response",
        lambda *args, **kwargs: (
            "Machine learning learns from data. [Source 1, Page 99]"
        ),
    )

    result = pipeline.answer_question("What is machine learning?")

    validation = result["citation_validation"]

    assert validation["has_citations"] is True
    assert validation["total_citations"] == 1
    assert validation["valid_count"] == 0
    assert validation["invalid_count"] == 1


def test_pipeline_without_retrieved_evidence(monkeypatch):
    """Test pipeline behavior when retrieval returns no chunks."""

    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda *args, **kwargs: [],
    )

    def unexpected_generation(*args, **kwargs):
        pytest.fail("LLM generation should not run without evidence.")

    monkeypatch.setattr(
        pipeline,
        "generate_response",
        unexpected_generation,
    )

    result = pipeline.answer_question("What is machine learning?")

    assert result["sources"] == []
    assert "could not find relevant evidence" in result["answer"].lower()

    assert result["citation_validation"]["has_citations"] is False
    assert result["citation_validation"]["total_citations"] == 0
    assert result["citation_validation"]["valid_count"] == 0
    assert result["citation_validation"]["invalid_count"] == 0


def test_pipeline_rejects_empty_query():
    """Test that an empty query raises a ValueError."""

    with pytest.raises(ValueError):
        pipeline.answer_question("")


# --------------------------------------------------
# MANUAL PIPELINE RUNNER
# --------------------------------------------------

def run_manual_pipeline():
    """
    Run the actual RAG pipeline using the configured
    retrieval system and LLM.
    """

    query = "What is machine learning?"

    result = pipeline.answer_question(query, top_k=3)

    print("\n" + "=" * 60)
    print("NOTEWISE — RAG PIPELINE TEST")
    print("=" * 60)

    print("\nQuestion:")
    print(query)

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nRetrieved Sources:")

    for source in result["sources"]:
        print(
            f"Document: {source['document_id']} | "
            f"Page: {source['page_number']} | "
            f"Chunk: {source['chunk_id']} | "
            f"Score: {source['score']:.4f}"
        )

    print("\nCitation Validation:")
    print(result["citation_validation"])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_manual_pipeline()
