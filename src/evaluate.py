from src.retrieval import retrieve_relevant_chunks


# --- Test set ---
# Fill in questions from your PDFs and keywords that MUST appear in the retrieved chunk.
# Keywords are checked case-insensitively; a hit = at least one keyword found in any retrieved chunk.
TEST_SET = [
    {"question": "Wat is de naam van de ISO-standaard voor procedurele SQL?", "expected_keywords": ["PSM", "Persistent Stored Modules"]},
    {"question": "Welke programmeertaal is de standaard procedurele taal in PostgreSQL?", "expected_keywords": ["PL/pgSQL"]},
    {"question": "Wat is het verschil tussen een trusted en untrusted taal in PostgreSQL?", "expected_keywords": ["trusted", "untrusted"]},
    {"question": "Welk sleutelwoord gebruik je voor een functie die altijd hetzelfde resultaat geeft bij dezelfde invoer?", "expected_keywords": ["IMMUTABLE"]},
    {"question": "Welke variabele wordt op TRUE gezet na een succesvolle PERFORM of SELECT INTO?", "expected_keywords": ["FOUND"]},
    {"question": "Wat is het alternatief voor SELECT wanneer je het resultaat van een query niet wil opvangen?", "expected_keywords": ["PERFORM"]},
    {"question": "Welk sleutelwoord gebruik je om uitzonderingen af te handelen in PL/pgSQL?", "expected_keywords": ["EXCEPTION"]},
    {"question": "Welk SQL-commando gebruik je om een stored procedure op te roepen in PostgreSQL?", "expected_keywords": ["CALL"]},
    {"question": "Welke beveiligingsmodus voert een functie uit met de rechten van de eigenaar in plaats van de aanroeper?", "expected_keywords": ["DEFINER", "SECURITY DEFINER"]},
    {"question": "Welke clausule gebruik je binnen een functie om een queryresultaat op te slaan in een variabele?", "expected_keywords": ["INTO", "SELECT INTO"]},
]


def evaluate_config(all_chunks, chunk_embeddings, embedding_model, top_k=3, test_set=None):
    """
    Run the test set against a given index configuration.

    Returns a dict with:
      - hit_rate: fraction of questions where at least one retrieved chunk
                  contains an expected keyword
      - avg_score: mean cosine similarity of the top-1 result across all questions
    """
    if test_set is None:
        test_set = TEST_SET

    if not test_set:
        raise ValueError("TEST_SET is empty — add your questions to src/evaluate.py")

    hits = 0
    top_scores = []
    details = []

    for item in test_set:
        question = item["question"]
        keywords = [kw.lower() for kw in item["expected_keywords"]]

        results = retrieve_relevant_chunks(
            question, all_chunks, chunk_embeddings, embedding_model, top_k=top_k
        )

        top_score = results[0]["score"] if results else 0.0
        top_scores.append(top_score)

        retrieved_text = " ".join(r["text"].lower() for r in results)
        hit = any(kw in retrieved_text for kw in keywords)
        if hit:
            hits += 1

        details.append({
            "question": question,
            "hit": hit,
            "top_score": top_score,
            "top_chunks": results,
        })

    return {
        "hit_rate": hits / len(test_set),
        "hits": hits,
        "total": len(test_set),
        "avg_top1_score": sum(top_scores) / len(top_scores),
        "details": details,
    }
