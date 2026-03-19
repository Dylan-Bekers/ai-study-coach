import json
import random
import ollama
from src.retrieval import retrieve_relevant_chunks, build_context


def build_prompt(query, context):
    return f"""You are a study assistant that answers questions strictly from the provided course material.

Rules:
- Use ONLY the information in the context below. Do not use any outside knowledge.
- Write 2 to 3 clear, complete sentences.
- Always answer in the same language as the question.
- If the answer cannot be found in the context, respond with exactly: "This is not covered in the provided course material." or "Dit komt niet voor in het cursusmateriaal." depending on the language of the question.

Context:
{context}

Question: {query}

Answer:"""


def build_quiz_prompt(context):
    return f"""You are a study coach creating a short quiz based on course material.

Using ONLY the content below, generate exactly 3 multiple-choice questions.

Return ONLY a valid JSON array — no explanation, no markdown, just the JSON.

Format:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "A",
    "explanation": "..."
  }}
]

Use the same language as the source material.

Course material:
{context}"""


def build_summary_prompt(context):
    return f"""You are a study coach helping a student understand their course material.

Write a clear, structured summary of the following content.
- Use bullet points for key concepts
- Keep it concise but complete
- Use the same language as the source material

Content:
{context}

Summary:"""


def generate_answer(prompt, model_name="llama3.2"):
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def answer_question(query, all_chunks, chunk_embeddings, embedding_model, top_k=3):
    results = retrieve_relevant_chunks(query, all_chunks, chunk_embeddings, embedding_model, top_k=top_k)
    context = build_context(results)
    prompt = build_prompt(query, context)
    answer = generate_answer(prompt)

    return {
        "query": query,
        "answer": answer,
        "sources": results,
    }


def generate_quiz(topic, all_chunks, chunk_embeddings, embedding_model, top_k=5):
    results = retrieve_relevant_chunks(topic, all_chunks, chunk_embeddings, embedding_model, top_k=top_k)
    context = build_context(results)
    prompt = build_quiz_prompt(context)
    raw = generate_answer(prompt)

    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def generate_summary(all_chunks, model_name="llama3.2", sample_size=10):
    sampled = random.sample(all_chunks, min(sample_size, len(all_chunks)))
    context = "\n\n".join(c["text"] for c in sampled)
    prompt = build_summary_prompt(context)
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
