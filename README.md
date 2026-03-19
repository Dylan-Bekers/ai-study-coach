# RAG Study Assistant

A local Retrieval-Augmented Generation (RAG) pipeline that answers questions about course material (PDFs) using semantic search and a local LLM via Ollama.

## How it works

1. **Ingest** — PDFs in `data/raw/` are loaded and split into overlapping text chunks
2. **Embed** — chunks are encoded into vectors using `paraphrase-multilingual-MiniLM-L12-v2` (supports Dutch + English)
3. **Retrieve** — cosine similarity finds the most relevant chunks for a query
4. **Generate** — a prompt with retrieved context is sent to `llama3.2` via Ollama for an answer

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.2
```

## Usage

Place PDFs in `data/raw/`, then run:

```bash
python main.py
```

## Project structure

```
project/
├── main.py              # interactive CLI
├── src/
│   ├── ingest.py        # load + chunk PDFs
│   ├── embeddings.py    # load model, encode chunks
│   ├── retrieval.py     # semantic search
│   └── generation.py   # prompt building + LLM call
├── data/raw/            # PDF documents (gitignored)
└── notebooks/
    └── 01_project_setup.ipynb
```
