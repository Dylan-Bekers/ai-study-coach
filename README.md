# AI Smart Study Coach

An AI-powered study assistant that lets you load any PDF — lecture slides, course notes, readings — and interact with the content through natural language. Ask questions, generate quizzes, and get summaries, all grounded strictly in your own documents.

Built with a local RAG (Retrieval-Augmented Generation) pipeline using `sentence-transformers` and `llama3.2` via Ollama. No data leaves your machine.

## Features

- **Q&A** — ask questions in Dutch or English, answers are grounded in your PDFs
- **Quiz generation** — `/quiz <topic>` generates 3 multiple-choice questions on any topic, shown one at a time so you answer before seeing the solution
- **Progress tracking** — `/progress` shows your quiz history and flags weak topics to revisit
- **Summary** — `/summary` gives a structured overview of the loaded material
- **Any PDF** — point it at a single file or a whole folder

## How it works

1. **Ingest** — PDFs are loaded and split into overlapping text chunks
2. **Embed** — chunks are encoded into vectors using `paraphrase-multilingual-MiniLM-L12-v2`
3. **Retrieve** — cosine similarity finds the most relevant chunks for your query
4. **Generate** — retrieved context is passed to `llama3.2` via Ollama to produce an answer

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.2
```

## Usage

```bash
# load all PDFs from data/raw/ (default)
python main.py

# load a single PDF
python main.py --pdf path/to/lecture.pdf

# load all PDFs from a folder
python main.py --pdf path/to/folder/
```

**Commands inside the session:**

| Command | Description |
|---------|-------------|
| `<question>` | ask anything about your material |
| `/quiz <topic>` | generate a 3-question quiz on a topic |
| `/progress` | show quiz history and weak topics |
| `/summary` | summarize the loaded material |
| `/help` | show available commands |
| `exit` | quit |

## Project structure

```
project/
├── main.py                      # CLI entrypoint
├── compare.py                   # evaluate retrieval across configs
├── src/
│   ├── ingest.py                # load + chunk PDFs
│   ├── embeddings.py            # load model, encode chunks
│   ├── retrieval.py             # cosine similarity search
│   ├── generation.py            # prompts + LLM calls
│   ├── evaluate.py              # hit rate evaluation
│   └── progress.py              # quiz score tracking + weak topic detection
├── data/raw/                    # your PDFs (gitignored)
└── notebooks/
    └── 01_project_setup.ipynb   # exploration notebook
```

## Evaluation

Retrieval quality was measured across 6 configurations (2 embedding models × 3 chunk sizes) using a 10-question test set. Best results: `paraphrase-multilingual-MiniLM-L12-v2` with `chunk_size=300` (50% hit rate, avg cosine score 0.614).

```bash
python compare.py
python compare.py --verbose   # show missed questions per config
```
