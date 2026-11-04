# Hybrid Retrieval: Dense Search + BM25

## Overview

Hybrid retrieval combines two different information retrieval approaches:

1. **Dense Retrieval:** Uses vector embeddings to retrieve documents based on semantic similarity.
2. **BM25 Retrieval:** Uses keyword-based scoring to identify documents based on query terms and their frequency.

Combining both approaches helps retrieve relevant information using both semantic meaning and exact keyword matching.

## 1. Dense Retrieval

Dense retrieval converts text into numerical vector representations using an embedding model.

These vectors are stored in a vector database such as FAISS.

When a user submits a query:
- Convert the query into an embedding.
- Search for similar document vectors.
- Retrieve the top-K semantically relevant chunks.

**Advantages:**
- Captures semantic relationships.
- Can retrieve conceptually similar content even when exact words differ.

**Limitations:**
- May miss exact keyword matches.
- Retrieval quality depends on embedding quality.

## 2. BM25 Retrieval

BM25 (Best Matching 25) is a probabilistic ranking algorithm used in information retrieval.

It ranks documents based on:
- Term Frequency (TF)
- Inverse Document Frequency (IDF)
- Document Length Normalization

**Advantages:**
- Effective for exact keyword matching.
- Useful for technical terms, identifiers, and specific phrases.

**Limitations:**
- Does not inherently understand semantic meaning.
- May miss relevant documents that use different terminology.

## 3. Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion combines ranked results from multiple retrieval systems into a unified ranking.

The RRF scoring formula is:

\[
RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}
\]

Where:

- \(d\): Retrieved document.
- \(R\): Set of ranking systems.
- \(rank_r(d)\): Position of document \(d\) in ranking system \(r\).
- \(k\): Ranking constant.

A document appearing near the top of multiple retrieval lists receives a stronger combined score.

**Why RRF?**

Instead of directly combining incompatible similarity scores from dense and keyword retrieval, RRF combines their ranking positions.

## 4. Hybrid Retrieval Workflow

1. Receive a user query.
2. Generate a query embedding.
3. Retrieve relevant chunks using FAISS.
4. Retrieve keyword-matching chunks using BM25.
5. Combine ranked results using RRF.
6. Return the final ranked document chunks.

## 5. Key Takeaways

| Component | Primary Purpose |
|---|---|
| FAISS | Dense vector similarity search |
| BM25 | Keyword-based retrieval |
| RRF | Rank fusion |
| Cross-Encoder | More detailed query-document relevance scoring |

## 6. Further Learning

- Understand embedding models and vector similarity.
- Implement BM25 retrieval.
- Experiment with RRF ranking.
- Compare dense-only and hybrid retrieval.
- Evaluate retrieval using Recall@K and MRR.

**Note:** Hybrid retrieval improves retrieval coverage in many situations, but its actual effectiveness should be validated using a benchmark dataset.