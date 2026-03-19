"""
compare.py — evaluate retrieval quality across chunk sizes and embedding models.

Usage:
    python compare.py
    python compare.py --verbose     # show missed questions per config
"""

import sys
from src.ingest import load_all_pdfs_from_folder, chunk_text
from src.embeddings import load_embedding_model, encode_chunks
from src.evaluate import evaluate_config, TEST_SET

DATA_FOLDER = "data/raw"
TOP_K = 3

CHUNK_SIZES = [300, 500, 700]
EMBEDDING_MODELS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
]


def build_chunks(documents, chunk_size):
    all_chunks = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc["text"], chunk_size=chunk_size, overlap=100)):
            all_chunks.append({"filename": doc["filename"], "chunk_id": i, "text": text})
    return all_chunks


def print_table(results):
    header = f"{'Model':<45} {'Chunk':>6} {'Hits':>6} {'Hit rate':>10} {'Avg score':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<45} {r['chunk_size']:>6} {r['hits']:>4}/{r['total']:<2}"
            f" {r['hit_rate']:>9.0%} {r['avg_top1_score']:>10.4f}"
        )
    print()


def print_misses(metrics, model_name, chunk_size):
    misses = [d for d in metrics["details"] if not d["hit"]]
    if not misses:
        return
    print(f"\n  Misses for {model_name} | chunk_size={chunk_size}:")
    for d in misses:
        print(f"    Q: {d['question']}")
        print(f"       top score: {d['top_score']:.4f}")
        for i, chunk in enumerate(d["top_chunks"], 1):
            preview = chunk["text"][:120].replace("\n", " ")
            print(f"       [{i}] ({chunk['score']:.4f}) {preview}...")
    print()


def main():
    verbose = "--verbose" in sys.argv

    if not TEST_SET:
        print("ERROR: TEST_SET is empty. Add your questions to src/evaluate.py first.")
        return

    print(f"Loading documents from '{DATA_FOLDER}'...")
    documents = load_all_pdfs_from_folder(DATA_FOLDER)
    print(f"Loaded {len(documents)} document(s). Running {len(CHUNK_SIZES) * len(EMBEDDING_MODELS)} configs...\n")

    results = []
    loaded_models = {}

    for model_name in EMBEDDING_MODELS:
        print(f"Loading model: {model_name}")
        loaded_models[model_name] = load_embedding_model(model_name)

    for chunk_size in CHUNK_SIZES:
        all_chunks = build_chunks(documents, chunk_size)

        for model_name in EMBEDDING_MODELS:
            print(f"  chunk_size={chunk_size} | model={model_name} | {len(all_chunks)} chunks", end=" ... ")
            embedding_model = loaded_models[model_name]
            chunk_embeddings = encode_chunks(all_chunks, embedding_model)

            metrics = evaluate_config(all_chunks, chunk_embeddings, embedding_model, top_k=TOP_K)
            print(f"hit rate={metrics['hit_rate']:.0%}  avg_score={metrics['avg_top1_score']:.4f}")

            if verbose:
                print_misses(metrics, model_name, chunk_size)

            results.append({
                "model": model_name,
                "chunk_size": chunk_size,
                **metrics,
            })

    print_table(results)


if __name__ == "__main__":
    main()
