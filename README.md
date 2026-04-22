# 🔎 Semantic Search Engine: Sparse vs Dense Retrieval

This project demonstrates and compares **traditional keyword-based search (BM25)** with **modern semantic search (Sentence Transformers)** for document retrieval.

It shows how dense embeddings outperform sparse methods in understanding meaning and context.

---

## 🚀 Live Demo
👉 Hugging Face Space:  
https://yokitha-semantic-search-demo.hf.space/

---

## 📌 Project Overview

This system implements two retrieval techniques:

### 1️⃣ Sparse Retrieval (BM25)
- Based on keyword matching (TF-IDF style ranking)
- Works well for exact word overlap
- Fails to understand synonyms and context

### 2️⃣ Dense Retrieval (Sentence Transformers)
- Uses transformer-based embeddings
- Captures semantic meaning of text
- Retrieves contextually similar documents even without exact keywords

---

## 💡 Example

Query:
> "capital city of France"

### BM25 Output:
- Works if exact words match (e.g., "France", "capital")

### Dense Retrieval Output:
- Understands meaning → returns "Paris is the French capital"

---

## 🧰 Tech Stack

- Python 🐍
- Rank-BM25 (Sparse Retrieval)
- Sentence Transformers (Hugging Face)
- NumPy (Vector similarity)
- Gradio (Web UI)
- Hugging Face Spaces (Deployment)

---

## 📊 Key Insights

- BM25 is fast but keyword-dependent
- Dense retrieval understands semantics
- Modern AI systems (RAG, Chatbots, QA systems) rely on dense retrieval

---

## 🚀 Applications

- AI Chatbots 🤖
- Question Answering Systems
- Retrieval-Augmented Generation (RAG)
- Semantic Search Engines

---

## 📷 Features

- Compare BM25 vs Dense Retrieval
- Interactive search UI
- Real-time ranking results
- Easy-to-use web interface

---

## 📌 Author
Yokitha R
