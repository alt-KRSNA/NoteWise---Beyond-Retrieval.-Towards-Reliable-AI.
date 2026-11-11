
import json
from pathlib import Path

from test.evaluation.test_evidence_sufficiency import EVALUATION_CASES
from app.retrieval.retriever import retrieve


DB_PATH = "database/notewise_rebuild.db"
INDEX_PATH = "database/faiss_rebuild.index"
TOP_K = 3

OUTPUT_PATH = Path("test/evaluation/evidence_evaluation_results.json")


def run_evaluation():
    evaluation_results = []

    print("=" * 75)
    print("NOTEWISE — EVIDENCE SUFFICIENCY EVALUATION")
    print("=" * 75)

    for number, case in enumerate(EVALUATION_CASES, start=1):
        query = case["question"]

        results = retrieve(
            query,
            top_k=TOP_K,
            db_path=DB_PATH,
            index_path=INDEX_PATH,
        )

        retrieved_chunks = []

        for rank, result in enumerate(results, start=1):
            retrieved_chunks.append({
                "rank": rank,
                "page_number": result["page_number"],
                "chunk_id": result["chunk_id"],
                "faiss_id": result["faiss_id"],
                "similarity_score": result["score"],
                "text": result["text"],
            })

        evaluation_results.append({
            "case_id": number,
            "question": query,
            "expected_label": case["expected_label"],
            "retrieved_chunks": retrieved_chunks,
        })

        print(
            f"{number:02d}. {query} | "
            f"Retrieved: {len(retrieved_chunks)}"
        )

    report = {
        "dataset_size": len(EVALUATION_CASES),
        "top_k": TOP_K,
        "database": DB_PATH,
        "index": INDEX_PATH,
        "results": evaluation_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETED")
    print("Total questions:", len(evaluation_results))
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    run_evaluation()
