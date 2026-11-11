
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

ANNOTATIONS_PATH = BASE_DIR / "evidence_annotations.json"
RESULTS_PATH = BASE_DIR / "evidence_evaluation_results.json"


def compute_metrics(annotations, results):
    total = len(results["results"])

    if total == 0:
        raise ValueError("Evaluation dataset cannot be empty.")

    top1_hits = 0
    top3_hits = 0
    reciprocal_ranks = []

    label_counts = {
        "direct_support": 0,
        "related_only": 0,
        "no_support": 0,
    }

    for case in results["results"]:
        case_id = str(case["case_id"])

        rank = annotations[case_id]["first_supporting_rank"]
        label = annotations[case_id]["label"]

        # Count evidence categories
        if label not in label_counts:
            raise ValueError(f"Unknown evidence label: {label}")

        label_counts[label] += 1

        # Calculate retrieval metrics
        if rank is not None and rank <= 1:
            top1_hits += 1

        if rank is not None and rank <= 3:
            top3_hits += 1

        reciprocal_ranks.append(
            1 / rank if rank is not None and rank <= 3 else 0
        )

    return {
        "total": total,
        "top1_support_rate": top1_hits / total,
        "top3_support_rate": top3_hits / total,
        "mrr_at_3": sum(reciprocal_ranks) / total,
        "label_counts": label_counts,
    }


def calculate_metrics():
    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    metrics = compute_metrics(annotations, results)

    print("\n" + "=" * 55)
    print("NOTEWISE — RETRIEVAL EVALUATION")
    print("=" * 55)

    print(f"Total questions:       {metrics['total']}")
    print(f"Top-1 support rate:    {metrics['top1_support_rate']:.2%}")
    print(f"Top-3 support rate:    {metrics['top3_support_rate']:.2%}")
    print(f"MRR@3:                 {metrics['mrr_at_3']:.4f}")

    print("\nEvidence category distribution:")

    for label, count in metrics["label_counts"].items():
        percentage = count / metrics["total"]
        print(f"{label:<18} {count:>2} ({percentage:.2%})")

    print("=" * 55)


if __name__ == "__main__":
    calculate_metrics()
