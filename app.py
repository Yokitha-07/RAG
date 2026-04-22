import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

# Documents
docs = [
    "The Eiffel Tower is located in Paris, France.",
    "The Leaning Tower of Pisa is in Italy.",
    "Paris is the capital of France, known for art, fashion and history.",
    "Paris is the French capital",
    "The Great Wall of China is visible from space.",
    "France is famous for its cuisine and wines."
]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("🔎 Semantic Search (BM25 vs Dense Retrieval)")

query = st.text_input("Enter your search query")

method = st.selectbox("Choose method", ["BM25", "Dense Retrieval"])

if query:

    if method == "BM25":
        tokenized_docs = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)

        scores = bm25.get_scores(query.lower().split())

        results = sorted(enumerate(scores), key=lambda x: -x[1])

        st.subheader("BM25 Results")
        for i, score in results:
            st.write(f"**{docs[i]}** → Score: {score:.2f}")

    else:
        doc_embeddings = model.encode(docs, normalize_embeddings=True)
        query_embedding = model.encode([query], normalize_embeddings=True)

        scores = np.dot(doc_embeddings, query_embedding.T).squeeze()
        results = sorted(enumerate(scores), key=lambda x: -x[1])

        st.subheader("Dense Retrieval Results")
        for i, score in results:
            st.write(f"**{docs[i]}** → Similarity: {score:.2f}")
