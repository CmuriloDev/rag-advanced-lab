import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(MODEL_NAME)

with open("data/medical_manuals.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

texts = [doc["text"] for doc in documents]

embeddings = embedding_model.encode(texts, convert_to_numpy=True)

dimension = embeddings.shape[1]

index = faiss.IndexHNSWFlat(dimension, 32)

index.hnsw.efConstruction = 200

index.add(np.array(embeddings, dtype=np.float32))

faiss.write_index(index, "artifacts/faiss_hnsw.index")

print("HNSW index created successfully.")