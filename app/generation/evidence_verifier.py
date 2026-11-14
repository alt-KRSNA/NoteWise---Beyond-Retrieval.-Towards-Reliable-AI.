
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

DEFAULT_THRESHOLD = 0.70

_model = None
_tokenizer = None
_device = None


def load_nli_model():
    """
    Load the NLI model only when semantic verification
    is requested.
    """

    global _model, _tokenizer, _device

    if _model is None:
        _device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        _model.to(_device)
        _model.eval()

        # Ensure the model exposes meaningful class labels.
        labels = _model.config.id2label.values()

        if not any(
            label.lower() in {"entailment", "contradiction", "neutral"}
            for label in labels
        ):
            raise RuntimeError(
                "The NLI model does not expose the expected class labels."
            )

    return _model, _tokenizer, _device


def predict_nli(premise, hypothesis):
    """
    Compare evidence (premise) with a generated claim (hypothesis).

    Returns the predicted NLI label and its score.
    """

    model, tokenizer, device = load_nli_model()

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    label = model.config.id2label[predicted_index].lower()

    score = float(probabilities[predicted_index].item())

    if label not in {"entailment", "contradiction", "neutral"}:
        raise RuntimeError(f"Unexpected NLI label: {label}")

    return label, score


def split_into_claims(answer):
    """
    Split a generated answer into individual sentences.

    This is a simple baseline sentence splitter.
    """

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be a non-empty string.")

    # Remove citation references before verification.
    answer = re.sub(
        r"\[Source\s+\d+\s*,\s*Page\s+\d+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    claims = []

    for sentence in sentences:
        sentence = sentence.strip()

        # Remove common bullet and numbering prefixes.
        sentence = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", sentence)

        if sentence:
            claims.append(sentence)

    return claims


def verify_claims(
    answer,
    retrieved_chunks,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Verify generated claims against retrieved evidence.

    Each claim is compared against every retrieved chunk.

    A claim is:
    - supported if entailment score reaches the threshold
    - contradicted if contradiction score reaches the threshold
    - not_supported otherwise

    This is an experimental NLI baseline, not a guarantee
    of factual correctness.
    """

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be a non-empty string.")

    if not isinstance(retrieved_chunks, list):
        raise ValueError("Retrieved chunks must be a list.")

    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 < threshold <= 1
    ):
        raise ValueError("Threshold must be greater than 0 and at most 1.")

    claims = split_into_claims(answer)

    results = []

    for claim in claims:

        best_entailment = None
        best_contradiction = None

        # Compare the claim against every retrieved chunk.
        for chunk in retrieved_chunks:

            evidence = chunk.get("text")

            if not isinstance(evidence, str) or not evidence.strip():
                continue

            label, score = predict_nli(
                premise=evidence,
                hypothesis=claim,
            )

            if label == "entailment":
                if (
                    best_entailment is None
                    or score > best_entailment["score"]
                ):
                    best_entailment = {
                        "score": score,
                        "chunk": chunk,
                    }

            elif label == "contradiction":
                if (
                    best_contradiction is None
                    or score > best_contradiction["score"]
                ):
                    best_contradiction = {
                        "score": score,
                        "chunk": chunk,
                    }

        # Decide the claim's verification status.
        if (
            best_entailment is not None
            and best_entailment["score"] >= threshold
        ):
            status = "supported"
            best_match = best_entailment

        elif (
            best_contradiction is not None
            and best_contradiction["score"] >= threshold
        ):
            status = "contradicted"
            best_match = best_contradiction

        else:
            status = "not_supported"

            candidates = [
                item
                for item in [best_entailment, best_contradiction]
                if item is not None
            ]

            best_match = (
                max(candidates, key=lambda item: item["score"])
                if candidates
                else None
            )

        evidence_reference = None

        if best_match is not None:
            chunk = best_match["chunk"]

            evidence_reference = {
                "document_id": chunk.get("document_id"),
                "chunk_id": chunk.get("chunk_id"),
                "page_number": chunk.get("page_number"),
            }

        results.append({
            "claim": claim,
            "status": status,
            "score": (
                best_match["score"]
                if best_match is not None
                else None
            ),
            "evidence": evidence_reference,
        })

    return {
        "model": MODEL_NAME,
        "threshold": threshold,
        "total_claims": len(results),
        "supported_count": sum(
            item["status"] == "supported" for item in results
        ),
        "contradicted_count": sum(
            item["status"] == "contradicted" for item in results
        ),
        "not_supported_count": sum(
            item["status"] == "not_supported" for item in results
        ),
        "claims": results,
    }
