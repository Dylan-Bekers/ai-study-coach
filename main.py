"""
AI Smart Study Coach
Usage:
    python main.py                          # loads PDFs from data/raw/
    python main.py --pdf path/to/file.pdf   # single PDF
    python main.py --pdf path/to/folder/    # all PDFs in a folder

Commands while running:
    <question>         ask a question about your material
    /quiz <topic>      generate a 3-question quiz on a topic
    /summary           summarize the loaded material
    exit               quit
"""

import argparse
from pathlib import Path

from src.ingest import load_all_pdfs_from_folder, extract_text_from_pdf, chunk_text
from src.embeddings import load_embedding_model, encode_chunks
from src.generation import answer_question, generate_quiz, generate_summary


def build_chunks(documents, chunk_size=500, overlap=100):
    all_chunks = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)):
            all_chunks.append({"filename": doc["filename"], "chunk_id": i, "text": text})
    return all_chunks


def load_documents(pdf_path):
    path = Path(pdf_path)

    if path.is_file() and path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path)
        return [{"filename": path.name, "text": text}]

    if path.is_dir():
        docs = load_all_pdfs_from_folder(path)
        if not docs:
            print(f"No PDFs found in '{path}'.")
        return docs

    print(f"'{pdf_path}' is not a valid PDF file or folder.")
    return []


def print_help():
    print("\nCommands:")
    print("  <question>       ask a question about your material")
    print("  /quiz <topic>    generate a quiz on a topic")
    print("  /summary         summarize the loaded material")
    print("  /help            show this message")
    print("  exit             quit\n")


def main():
    parser = argparse.ArgumentParser(description="AI Smart Study Coach")
    parser.add_argument("--pdf", default="data/raw", help="Path to a PDF file or folder (default: data/raw)")
    args = parser.parse_args()

    print(f"\nLoading documents from '{args.pdf}'...")
    documents = load_documents(args.pdf)
    if not documents:
        return

    print(f"Loaded {len(documents)} document(s): {[d['filename'] for d in documents]}")

    print("Chunking...")
    all_chunks = build_chunks(documents)
    print(f"{len(all_chunks)} chunks created.")

    print("Loading embedding model...")
    model = load_embedding_model()
    chunk_embeddings = encode_chunks(all_chunks, model)
    print("Ready.\n")

    print_help()

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            break

        if query == "/help":
            print_help()
            continue

        if query == "/summary":
            print("\nGenerating summary...\n")
            summary = generate_summary(all_chunks)
            print(summary)
            print()
            continue

        if query.startswith("/quiz"):
            topic = query[5:].strip()
            if not topic:
                print("Usage: /quiz <topic>  (e.g. /quiz stored functions)\n")
                continue
            print(f"\nGenerating quiz on '{topic}'...\n")
            quiz = generate_quiz(topic, all_chunks, chunk_embeddings, model)
            print(quiz)
            print()
            continue

        response = answer_question(query, all_chunks, chunk_embeddings, model)
        print(f"\nAnswer: {response['answer']}")
        print("\nSources:")
        for s in response["sources"]:
            print(f"  - {s['filename']} | chunk {s['chunk_id']} | score {s['score']:.4f}")
        print()


if __name__ == "__main__":
    main()
