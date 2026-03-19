from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """Load and return the sentence transformer model."""
    return SentenceTransformer(model_name)


def encode_chunks(chunks, model):
    """Encode a list of chunk dicts into numpy embeddings."""
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, convert_to_numpy=True)
