import gradio as gr
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

docs = [
    "The Eiffel Tower is located in Paris, France.",
    "The Leaning Tower of Pisa is in Italy.",
    "Paris is the capital of France, known for art, fashion and history.",
    "Paris is the French capital",
    "The Great Wall of China is visible from space.",
    "France is famous for its cuisine and wines."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, method):

    if method == "BM25":
        tokenized_docs = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)

        scores = bm25.get_scores(query.lower().split())
        results = sorted(enumerate(scores), key=lambda x: -x[1])

        return "\n\n".join([f"{docs[i]} (score={s:.2f})" for i, s in results])

    else:
        doc_embeddings = model.encode(docs, normalize_embeddings=True)
        query_embedding = model.encode([query], normalize_embeddings=True)

        scores = np.dot(doc_embeddings, query_embedding.T).squeeze()
        results = sorted(enumerate(scores), key=lambda x: -x[1])

        return "\n\n".join([f"{docs[i]} (score={s:.2f})" for i, s in results])


demo = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(label="Query"),
        gr.Radio(["BM25", "Dense Retrieval"])
    ],
    outputs="text",
    title="Semantic Search Engine"
)

demo.launch()
