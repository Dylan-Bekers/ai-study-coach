# Technical Report — AI Smart Study Coach

**Course:** Advanced AI
**Student:** Dylan Bekers
**Repository:** [TODO: GitHub link]

---

## 1. Introduction

Students often struggle to efficiently extract information from their own course materials — lecture slides, notes, and readings that accumulate throughout the semester. Generic AI chatbots have no access to these personal documents and tend to produce hallucinated or overly general answers. PDF search is purely keyword-based and requires the student to already know what to look for.

This project builds an **AI-powered Study Coach** that lets a student load any PDF — lecture slides, course notes, or readings — and interact with that material through natural language. The system supports three modes: answering questions, generating topic-based quizzes, and summarizing loaded content. All responses are grounded strictly in the uploaded documents rather than outside knowledge.

The core technique is **Retrieval-Augmented Generation (RAG)**: at query time, the most semantically relevant passages are retrieved from the documents and passed to a language model that generates an answer based only on that context. This prevents hallucination and keeps the study coach focused on the actual course material.

The system runs entirely locally, meaning no documents or queries are sent to external servers — an important consideration for students uploading institutional or personal material.

---

## 2. Data

**Source material:** Three PDF files were used for development and evaluation: lecture slide decks from a PostgreSQL/PL-pgSQL course at UCLL, covering procedural SQL, stored functions and procedures, PL/pgSQL syntax, and query optimization with EXPLAIN. These represent a realistic student use case — exported PowerPoint slides written in Dutch, containing a mix of bullet points, prose, and code examples. The system accepts any PDF a student provides via the `--pdf` argument.

**Preprocessing:** Each PDF is parsed with `pypdf`, extracting raw text page by page. The text is split into overlapping chunks using sentence-boundary detection (splitting on `.`, `!`, `?`). A 100-character overlap between consecutive chunks prevents information from being cut off at boundaries.

**Challenge — slide-based format:** Slide exports are significantly harder to retrieve from than prose documents. Bullet points lack surrounding explanatory context, title slides repeat many topic keywords but contain no actual information, and code blocks appear without natural language descriptions. This structural mismatch between document format and the assumptions of dense retrieval is a central finding of the evaluation.

---

## 3. Model & Methods

**Language model selection — a progression:**
Choosing the right generation model required several iterations. The project started with `t5-v1_1-small`, a lightweight sequence-to-sequence model that could run locally without issue but produced poor answers — it would often repeat fragments of the context rather than synthesizing a coherent response, and lacked the reasoning ability to combine information from multiple retrieved chunks. The next candidate was `google/mt5-base`, Google's multilingual T5 model, which was appealing because the course material is in Dutch. While it handled the language better, mt5-base is designed primarily for translation and summarization tasks, not open-ended question answering — it consistently produced incomplete or grammatically broken outputs when used for generation. The final choice was **llama3.2 via ollama**, which runs entirely on the local machine, produces well-reasoned and fluent answers, correctly handles Dutch content, and reliably follows prompt instructions such as staying within the provided context and matching the question's language.

**Pipeline:** The system consists of four stages:

1. **Ingestion:** PDFs are loaded and chunked with configurable `chunk_size` and `overlap` (`src/ingest.py`). Any file or folder can be passed via `--pdf`.
2. **Embedding:** Chunks are encoded into dense vectors using a `SentenceTransformer` model (`src/embeddings.py`). Two models were evaluated: `paraphrase-multilingual-MiniLM-L12-v2` (multilingual) and `all-MiniLM-L6-v2` (English-only, lighter).
3. **Retrieval:** The student's input is encoded with the same model and the top-k most similar chunks are selected by cosine similarity (`src/retrieval.py`).
4. **Generation:** Retrieved chunks are inserted into a prompt and passed to llama3.2 via ollama. The prompt enforces strict grounding in the context and language matching (`src/generation.py`).

Beyond Q&A, the system offers `/quiz <topic>` (generates 3 multiple-choice questions from retrieved chunks on that topic) and `/summary` (summarizes a random sample of loaded chunks).

---

## 4. Results & Evaluation

**Evaluation method:** A test set of 10 Dutch questions was constructed from the PostgreSQL course material, each paired with expected keywords that must appear in at least one retrieved chunk to count as a hit. Six configurations were evaluated by varying chunk size (300, 500, 700 characters) and embedding model.

| Model                                 | Chunk size | Hit rate | Avg score |
| ------------------------------------- | ---------- | -------- | --------- |
| paraphrase-multilingual-MiniLM-L12-v2 | 300        | **50%**  | **0.614** |
| paraphrase-multilingual-MiniLM-L12-v2 | 500        | 40%      | 0.566     |
| paraphrase-multilingual-MiniLM-L12-v2 | 700        | 40%      | 0.566     |
| all-MiniLM-L6-v2                      | 300        | 40%      | 0.560     |
| all-MiniLM-L6-v2                      | 500        | 40%      | 0.523     |
| all-MiniLM-L6-v2                      | 700        | **60%**  | 0.490     |

**Hit rate vs. confidence trade-off:** `all-MiniLM-L6-v2` with `chunk_size=700` achieves the highest hit rate (60%) but the lowest cosine score (0.49). Larger chunks contain more text, increasing the chance a keyword appears incidentally rather than because the chunk is semantically relevant. `paraphrase-multilingual-MiniLM-L12-v2` with `chunk_size=300` achieves the highest confidence (0.614) at 50% hit rate — tighter, more precise matches. For a study coach where answer quality matters over raw recall, the multilingual model with smaller chunks is preferable.

**Consistent failure — SECURITY DEFINER:** The question about security execution modes failed in all six configurations (scores 0.27–0.37). Manual inspection showed that `SECURITY DEFINER` appears only as a bare keyword inside a code block with no surrounding explanatory prose. This illustrates a core limitation of dense retrieval: if the document does not explain a concept in natural language, the semantic gap between a student's question and the relevant passage is too large to bridge with embedding similarity alone.

**Slide titles as false positives:** Several misses were caused by a high-level overview slide listing many topic keywords but containing no actual content. This slide consistently ranked in the top-3 for unrelated questions across all configurations, acting as a systematic noise source.

**Manual investigation of misses:**

To understand why certain questions failed, the `--verbose` flag was used to inspect the actual chunks retrieved for each missed question. Three patterns emerged:

*Pattern 1 — Keyword only in code, no prose context.* The question "Welke beveiligingsmodus voert een functie uit met de rechten van de eigenaar?" (SECURITY DEFINER) failed across all six configurations with cosine scores between 0.27 and 0.37 — far below any hit. Inspection showed that `SECURITY DEFINER` appears in the PDFs only as a keyword inside a code block, with no surrounding sentence that explains what it does. The embedding model encodes the *meaning* of the question ("security mode, owner's rights, caller") but the matching code block carries almost none of that meaning in its vector. This is a fundamental mismatch between how dense retrieval works and how technical documentation is written.

*Pattern 2 — High-frequency title slide as noise.* Multiple missed questions had the same chunk ranked first: `"Procedurele SQL met plpgsql Wim.bertels@ucll.be — Objecten op de server: Stored procedures, Stored functions, Triggers..."`. This is a title/overview slide that contains many topic keywords but no actual information. Because it mentions nearly every concept in the course, its embedding lands close to almost any query. It acted as a systematic false positive across all configurations and chunk sizes.

*Pattern 3 — Right content, wrong chunk boundary.* The question about `PERFORM` ("alternatief voor SELECT") retrieved an EXPLAIN-related chunk as the top result (score 0.61) — from a completely different PDF. The chunk that actually contains the answer (`"PERFORM alternatief voor SELECT waarbij het resultaat niet wordt opgevangen"`) existed in the index but ranked lower. With `chunk_size=300` the relevant sentence was isolated in its own chunk, making it retrievable for some configs but not others depending on the model. This shows that chunking boundaries directly affect whether a key sentence is retrievable at all.

**Quiz generation:** The `/quiz` feature demonstrated that retrieval quality directly affects generation quality. When asked for a quiz on `EXPLAIN`, one of the five retrieved chunks was a PL/pgSQL chunk that was only marginally relevant, causing the model to generate an off-topic question about functions vs. procedures. This confirms that improving retrieval would benefit all three features of the study coach, not just Q&A.

---

## 5. Contributions

**Designed and built independently:**

- The full RAG pipeline architecture — deciding to separate ingestion, embedding, retrieval, and generation into distinct modules
- Sentence-boundary chunking with configurable size and overlap, chosen over fixed-character splitting to avoid cutting sentences mid-thought
- The evaluation framework: defining the test set of 10 Dutch questions, choosing hit rate and cosine score as metrics, and writing the comparison script that runs all 6 configurations automatically
- The quiz and summary features, including prompt engineering to make the model generate structured multiple-choice questions grounded in the retrieved context
- The `--pdf` CLI argument allowing any file or folder to be used as input

**Research and experimentation:**

- Read through the RAG literature to understand the trade-offs between chunk size, overlap, and retrieval quality before implementing
- Tested three language models (`t5-v1_1-small`, `google/mt5-base`, `llama3.2`) and two embedding models to find combinations that work well for Dutch technical content
- Investigated why certain questions consistently failed by manually inspecting retrieved chunks, leading to the finding about low-context code blocks and false-positive title slides

**Adapted from documentation and existing libraries:**

- `sentence-transformers` encoding pipeline from the official library documentation
- `ollama` Python API usage from its README
- `pypdf` page extraction from its documentation
- Cosine similarity retrieval using `sklearn.metrics.pairwise`, following standard RAG examples

**Used generative AI for:**

- Helping structure and proofread this report
- Explaining error messages and debugging issues during development
- Discussing design decisions (e.g. chunking strategy, prompt structure)

---

## 6. Challenges & Future Work

**Challenges encountered:**

- Selecting a suitable language model required testing multiple options. A Google model produced better outputs but could not be used due to privacy and cost constraints. This led to the choice of ollama, which provides a good balance of quality, local execution, and ease of use.
- Slide-exported PDFs are poorly suited to RAG due to low information density per chunk and keyword-heavy title slides acting as false positives.
- Technical keywords embedded in code blocks without prose context (e.g. `SECURITY DEFINER`) are nearly unretrievable with dense embeddings alone.

**Future improvements:**

- **Hybrid search:** Combine dense retrieval with BM25 keyword matching to handle exact technical terms that semantic models miss.
- **Slide-aware chunking:** Chunk per slide using page breaks and prepend the slide title to each chunk for richer context.
- **Persistent embeddings:** Save embeddings to disk so re-indexing is not required on every run.
- **Web interface:** Replace the CLI with a Streamlit UI where students can upload PDFs through a browser without using the command line.
- **Adaptive learning:** Track which questions a student answers incorrectly and prioritize quizzes on those topics in future sessions.
