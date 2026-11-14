from app.retrieval.retriever import retrieve

query = "What is machine learning?"

results = retrieve(query, top_k=5)

print("\n===== RETRIEVAL RESULTS =====\n")

for result in results:
    print("Page:", result["page_number"])
    print("Score:", round(result["score"], 4))
    print("Text:", result["text"][:200])
    print("-" * 60)