import json
import faiss
import numpy as np

from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBED_MODEL)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

with open("data/medical_manuals.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

texts = [doc["text"] for doc in documents]

index = faiss.read_index("artifacts/faiss_hnsw.index")


def generate_hyde(query):
    prompt = f"""
Convert the following colloquial medical complaint
into a technical medical description:

Complaint:
{query}

Technical description:
"""

    output = generator(
        prompt,
        max_new_tokens=40,
        do_sample=True
    )

    generated = output[0]["generated_text"]

    return generated


def retrieve_documents(hyde_text):
    vector = embedding_model.encode(
        [hyde_text],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        np.array(vector, dtype=np.float32),
        10
    )

    retrieved = [texts[i] for i in indices[0]]

    return retrieved


def rerank(query, retrieved_docs):
    pairs = [[query, doc] for doc in retrieved_docs]

    scores = cross_encoder.predict(pairs)

    ranked = list(zip(retrieved_docs, scores))

    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:3]


def main():
    query = "throbbing headache and light hurts my eyes"

    print("\nUSER QUERY:")
    print(query)

    hyde_document = generate_hyde(query)

    print("\nHYDE DOCUMENT:")
    print(hyde_document)

    retrieved_docs = retrieve_documents(hyde_document)

    print("\nTOP 10 RETRIEVED DOCUMENTS:")
    for doc in retrieved_docs:
        print("-", doc)

    reranked = rerank(query, retrieved_docs)

    print("\nFINAL TOP 3 DOCUMENTS:")
    for doc, score in reranked:
        print(f"\nScore: {score:.4f}")
        print(doc)


if __name__ == "__main__":
    main()