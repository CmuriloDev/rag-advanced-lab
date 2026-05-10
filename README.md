# LABORATÓRIO 09 — Advanced RAG Architecture (HNSW, HyDE and Cross-Encoders)

## Course Information

- Institution: iCEV
- Professor: Dimmy Magalhães
- Student: Carlos Murilo

---

# Project Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) pipeline focused on semantic retrieval in medical manuals.

The system combines:

- Dense embeddings
- HNSW vector indexing
- HyDE query transformation
- Cross-Encoder reranking

The objective is to improve retrieval quality when users describe symptoms using informal language instead of technical medical terminology.

---

# Problem Example

### User Query

```text
throbbing headache and light hurts my eyes
```

### Technical Medical Terms

```text
pulsatile headache and photophobia
```

Traditional cosine similarity may fail because colloquial language differs from technical documentation.

To solve this, the project uses HyDE to generate a hypothetical technical description before retrieval.

---

# Technologies Used

## Libraries

- Python 3.10
- FAISS
- Sentence Transformers
- Transformers
- Torch

## Models

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

### HyDE Generator

```text
distilgpt2
```

### Cross-Encoder

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

# Project Structure

```text
rag-advanced-lab/
│
├── data/
├── indexes/
├── src/
│   ├── build_index.py
│   ├── generate_hyde.py
│   └── rag_pipeline.py
│
├── requirements.txt
└── README.md
```

---

# HNSW Explanation

HNSW (Hierarchical Navigable Small World) is an Approximate Nearest Neighbor (ANN) algorithm optimized for fast vector similarity search.

Instead of comparing all vectors directly like exact KNN, HNSW creates a graph structure for efficient retrieval.

## Hyperparameters

### M

Controls the number of graph connections.

- Higher M → better accuracy, higher RAM usage
- Lower M → lower memory usage, lower retrieval quality

### ef_construction

Controls graph construction quality.

- Higher values → better precision, more memory consumption
- Lower values → faster indexing, lower accuracy

Compared to exact KNN, HNSW is significantly faster and more scalable for production systems.

---

# Pipeline Stages

## 1. Embedding Generation

Medical manual fragments are converted into dense embeddings using Sentence Transformers.

---

## 2. HyDE Query Transformation

The user query is transformed into a hypothetical technical medical description.

Example:

```text
User Query:
throbbing headache and light hurts my eyes
```

```text
Generated Technical Description:
Patient presents pulsatile headache associated with photophobia.
```

This improves semantic alignment between the query and technical documents.

---

## 3. HNSW Retrieval

The generated embedding is used to retrieve the Top-10 most relevant documents from the FAISS HNSW index.

---

## 4. Cross-Encoder Reranking

The Top-10 retrieved documents are reranked using a Cross-Encoder model.

This stage increases semantic precision before selecting the final Top-3 documents.

---

# Installation

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Environment (PowerShell)

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Build HNSW Index

```bash
python src/build_index.py
```

## Execute RAG Pipeline

```bash
python src/rag_pipeline.py
```

---

# Expected Output

The system displays:

- User query
- Generated HyDE document
- Top-10 retrieved documents
- Final Top-3 reranked documents

---

# Conclusion

This project demonstrates how modern RAG systems improve semantic retrieval using:

- HyDE semantic expansion
- HNSW approximate nearest neighbor retrieval
- Cross-Encoder reranking

The architecture simulates real-world production retrieval pipelines.

---

# AI Usage Declaration

Parts of this laboratory were generated/complemented with AI assistance, reviewed and validated by Carlos Murilo.

The AI assistance was limited to:
- repetitive dataset generation,
- auxiliary code templates,
- boilerplate documentation.

All final revisions, tests and validations were performed by Carlos Murilo.

---

# Version

```text
v1.0
```