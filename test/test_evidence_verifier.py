from unittest.mock import patch

import pytest

from app.generation.evidence_verifier import verify_claims


@pytest.fixture
def sample_chunks():
    return [
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "page_number": 5,
            "text": "Artificial intelligence enables machines to perform tasks "
                    "that normally require human intelligence."
        }
    ]


@patch("app.generation.evidence_verifier.predict_nli")
def test_supported_claim(mock_nli, sample_chunks):
    mock_nli.return_value = ("entailment", 0.95)

    result = verify_claims(
        "Artificial intelligence enables machines to perform intelligent tasks.",
        sample_chunks
    )

    assert result["supported_count"] == 1
    assert result["claims"][0]["status"] == "supported"


@patch("app.generation.evidence_verifier.predict_nli")
def test_contradicted_claim(mock_nli, sample_chunks):
    mock_nli.return_value = ("contradiction", 0.92)

    result = verify_claims(
        "Artificial intelligence cannot perform intelligent tasks.",
        sample_chunks
    )

    assert result["contradicted_count"] == 1
    assert result["claims"][0]["status"] == "contradicted"


@patch("app.generation.evidence_verifier.predict_nli")
def test_not_supported_claim(mock_nli, sample_chunks):
    mock_nli.return_value = ("neutral", 0.85)

    result = verify_claims(
        "Artificial intelligence was invented in 1950.",
        sample_chunks
    )

    assert result["not_supported_count"] == 1
    assert result["claims"][0]["status"] == "not_supported"


def test_empty_evidence():
    result = verify_claims(
        "Artificial intelligence is useful.",
        []
    )

    assert result["not_supported_count"] == 1


def test_invalid_threshold(sample_chunks):
    with pytest.raises(ValueError):
        verify_claims(
            "Artificial intelligence is useful.",
            sample_chunks,
            threshold=0
        )


def test_empty_answer(sample_chunks):
    with pytest.raises(ValueError):
        verify_claims("", sample_chunks)