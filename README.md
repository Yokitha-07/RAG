# 🧠 Comparing Sparse retrieval vs Dense retrieval for Semantic Search

This project demonstrates the difference between sparse retrieval (BM25) and dense retrieval (Sentence Transformers) when searching for relevant text documents.
It highlights how traditional keyword-based search models differ from modern semantic models that understand context and meaning.

##  🚀 Project Overview

This notebook compares two popular text retrieval techniques:

1. BM25 (Sparse Retrieval):

Based on keyword matching using term frequency and inverse document frequency (TF-IDF).

Fails to capture synonyms or contextual relationships between words.

 Sentence Transformers (Dense Retrieval):

Uses transformer-based embeddings to represent sentences in a semantic vector space.

Finds documents that are semantically similar to the query, not just keyword matches.

## 💡 Key Takeaways

BM25 → Good for exact term matches, fast and simple.

Sentence Transformers → Understand meaning, ideal for semantic and contextual search.

Dense retrieval methods are essential for modern AI applications like

- Chatbots

- Question Answering

- Retrieval-Augmented Generation (RAG)

## 🧰 Tech Stack

🐍 Python 3

📚 rank_bm25 – Sparse retrieval model

🤗 Sentence Transformers (Hugging Face) – Dense embeddings

🧮 NumPy – Vector operations

💾 FAISS (optional) – Efficient similarity search
