"""Embedding utilities — compute and compare embeddings via OpenAI."""

import math

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


def unit_to_embedding_text(unit: dict) -> str:
    """Build embedding text from a knowledge unit dict.

    Includes title, knowledge type, content, tags, intent tags,
    and reusable questions to maximize retrieval coverage.
    """
    parts = []
    if unit.get("title"):
        parts.append(unit["title"])
    if unit.get("knowledge_type"):
        parts.append(f"[{unit['knowledge_type']}]")
    if unit.get("content"):
        parts.append(unit["content"])
    if unit.get("topic_tags"):
        parts.append(" | ".join(unit["topic_tags"]))
    if unit.get("question_intent_tags"):
        parts.append(" | ".join(unit["question_intent_tags"]))
    if unit.get("reusable_for_questions"):
        parts.append(" ".join(unit["reusable_for_questions"]))
    if unit.get("method_name"):
        parts.append(f"Method: {unit['method_name']}")
    if unit.get("field"):
        parts.append(f"Field: {unit['field']}")
    if unit.get("keywords"):
        parts.append("Keywords: " + " | ".join(unit["keywords"]))
    if unit.get("problem_it_solves"):
        parts.append(f"Problem: {unit['problem_it_solves']}")
    if unit.get("related_methods"):
        parts.append("Related: " + " | ".join(unit["related_methods"]))
    return "\n".join(parts) if parts else ""


def keywords_to_text(method: dict) -> str:
    """Legacy helper — kept for backward compatibility."""
    return unit_to_embedding_text(method)


def compute_embedding(text: str, api_key: str) -> list[float]:
    """Compute embedding vector for a text string."""
    client = OpenAI(api_key=api_key)
    return client.embeddings.create(model=EMBEDDING_MODEL, input=text).data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _score_methods(
    query_embedding: list[float],
    methods: list[dict],
) -> list[tuple[float, dict]]:
    """Score all units by cosine similarity, sorted descending."""
    scored = []
    for m in methods:
        emb = m.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(query_embedding, emb)
        scored.append((round(score, 4), m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def retrieve_top_k(
    query_embedding: list[float],
    methods: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Find the top-K most similar knowledge units by cosine similarity."""
    results = []
    for score, m in _score_methods(query_embedding, methods)[:top_k]:
        m_copy = dict(m)
        m_copy["_similarity"] = score
        results.append(m_copy)
    return results


def score_all_methods(
    query_embedding: list[float],
    methods: list[dict],
) -> list[tuple[float, str]]:
    """Score ALL units and return (similarity, title) pairs, sorted descending."""
    return [(score, m.get("title", "unknown")) for score, m in _score_methods(query_embedding, methods)]
