import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_relevant_chunks(query, all_chunks, chunk_embeddings, model, top_k=3):
    """Return the top_k most semantically similar chunks to the query."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {
            "filename": all_chunks[idx]["filename"],
            "chunk_id": all_chunks[idx]["chunk_id"],
            "score": similarities[idx],
            "text": all_chunks[idx]["text"],
        }
        for idx in top_indices
    ]


def build_context(results):
    """Format retrieved chunks into a single context string."""
    context = ""
    for i, r in enumerate(results, 1):
        context += f"[Source {i} - {r['filename']} | chunk {r['chunk_id']}]\n"
        context += r["text"] + "\n\n"
    return context
