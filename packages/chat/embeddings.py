"""Embedding utilities — compute and compare embeddings via OpenAI."""

import logging
import math

from openai import OpenAI

logger = logging.getLogger(__name__)

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
    try:
        return client.embeddings.create(model=EMBEDDING_MODEL, input=text).data[0].embedding
    except Exception as e:
        logger.warning(f"[Embeddings] Failed to compute embedding: {e}")
        return []


def compute_embeddings_batch(
    texts: list[str], api_key: str
) -> list[list[float]]:
    """Compute embeddings for a batch of texts in a single API call.

    Returns a list of embedding vectors in the same order as *texts*.
    Empty strings are skipped (returned as empty lists).
    """
    if not texts:
        return []

    # Map non-empty texts to their original indices
    index_map: list[int] = []
    non_empty: list[str] = []
    for i, t in enumerate(texts):
        if t.strip():
            index_map.append(i)
            non_empty.append(t)

    results: list[list[float]] = [[] for _ in texts]
    if not non_empty:
        return results

    client = OpenAI(api_key=api_key)
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=non_empty)
        for item in resp.data:
            orig_idx = index_map[item.index]
            results[orig_idx] = item.embedding
    except Exception as e:
        logger.warning(f"[Embeddings] Batch embedding failed: {e}")

    return results


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
    domain: str | None = None,
) -> list[tuple[float, dict]]:
    """Score units by cosine similarity, optionally filtered by domain first."""
    pool = methods
    if domain:
        domain_lower = domain.lower()
        filtered = [m for m in methods if (m.get("_domain") or "").lower() == domain_lower]
        if filtered:  # only use filter if it found results
            pool = filtered
    scored = []
    for m in pool:
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


# ---------------------------------------------------------------------------
# Hybrid search (pgvector + tsvector)
# ---------------------------------------------------------------------------

RRF_K = 60  # Reciprocal Rank Fusion constant


def _has_cjk(text: str) -> bool:
    """Return True if text contains CJK characters (Chinese/Japanese/Korean)."""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:  # CJK Unified Ideographs
            return True
        if 0x3040 <= cp <= 0x30FF:  # Hiragana / Katakana
            return True
    return False


def _rrf_score(rank: int) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-indexed)."""
    return 1.0 / (RRF_K + rank)


def hybrid_search(
    db,
    vector_queries: list[str],
    keyword_terms: list[str],
    api_key: str,
    top_k: int = 50,
    boost_ids: set[int] | None = None,
    method_boost_ids: set[int] | None = None,
) -> list[tuple[float, dict]]:
    """Run dual-path retrieval: pgvector ANN + tsvector full-text, merged via RRF.

    Args:
        db: SQLAlchemy Session.
        vector_queries: Search queries for embedding-based retrieval.
        keyword_terms: Terms for full-text search (method names, aliases, keywords).
        api_key: OpenAI API key for computing query embeddings.
        top_k: Max results per search path.
        boost_ids: KU IDs to boost (taxonomy match) — score *= 1.3.
        method_boost_ids: KU IDs to boost (method match) — score *= 1.5.

    Returns:
        Sorted list of (score, ku_dict) tuples, highest score first.
    """
    from sqlalchemy import text as sa_text

    # --- Vector search: one query per search string, merge best rank per KU ---
    vector_ranked: dict[int, int] = {}  # ku_id -> best rank (1-indexed)
    for query_str in vector_queries:
        if not query_str.strip():
            continue
        emb = compute_embedding(query_str, api_key)
        if not emb:
            continue
        emb_str = "[" + ",".join(str(x) for x in emb) + "]"
        rows = db.execute(
            sa_text("""
                SELECT id, embedding <=> :emb AS distance
                FROM knowledge_units
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :emb
                LIMIT :k
            """),
            {"emb": emb_str, "k": top_k},
        ).fetchall()
        for rank, row in enumerate(rows, 1):
            ku_id = row[0]
            if ku_id not in vector_ranked or rank < vector_ranked[ku_id]:
                vector_ranked[ku_id] = rank

    # --- Full-text search: combine all keyword terms with OR ---
    text_ranked: dict[int, int] = {}  # ku_id -> rank (1-indexed)
    all_terms = list(keyword_terms)
    # Also add the original vector queries as keyword search terms
    for q in vector_queries:
        if q.strip():
            all_terms.append(q.strip())

    if all_terms:
        search_text = " or ".join(t.strip() for t in all_terms if t.strip())
        if search_text:
            if _has_cjk(search_text):
                # CJK text: skip pg_bigm full-text search (no indexes, causes timeout)
                # Vector search handles CJK fine via embeddings
                logger.info("[Hybrid] CJK detected, skipping text search (vector only)")
            else:
                # Standard tsvector full-text search for non-CJK queries
                rows = db.execute(
                    sa_text("""
                        SELECT id, ts_rank_cd(search_vector, websearch_to_tsquery('english', :query), 32) AS rank
                        FROM knowledge_units
                        WHERE search_vector @@ websearch_to_tsquery('english', :query)
                        ORDER BY rank DESC
                        LIMIT :k
                    """),
                    {"query": search_text, "k": top_k},
                ).fetchall()
                for rank, row in enumerate(rows, 1):
                    text_ranked[row[0]] = rank

    # --- RRF merge ---
    all_ids = set(vector_ranked.keys()) | set(text_ranked.keys())
    if not all_ids:
        return []

    rrf_scores: dict[int, float] = {}
    for ku_id in all_ids:
        score = 0.0
        if ku_id in vector_ranked:
            score += _rrf_score(vector_ranked[ku_id])
        if ku_id in text_ranked:
            score += _rrf_score(text_ranked[ku_id])
        # Apply boosts
        if boost_ids and ku_id in boost_ids:
            score *= 1.3
        if method_boost_ids and ku_id in method_boost_ids:
            score *= 1.5
        rrf_scores[ku_id] = round(score, 6)

    # --- Load full KU data for top results ---
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    top_ids = sorted_ids[:top_k]

    from models import KnowledgeUnit, Paper
    kus = db.query(KnowledgeUnit).filter(KnowledgeUnit.id.in_(top_ids)).all()
    ku_map = {ku.id: ku for ku in kus}

    # Load paper metadata
    paper_ids = {ku.paper_id for ku in kus if ku.paper_id}
    papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all() if paper_ids else []
    paper_map = {p.id: p for p in papers}

    # Build result dicts (matching _load_unit_dicts format in chat.py)
    _UNIT_FIELDS = (
        "title", "source_type", "section", "knowledge_type",
        "content", "evidence_span", "limitations", "confidence",
        "method_name", "field", "problem_it_solves", "model_assumption",
        "input_format", "output_format",
    )
    _LIST_FIELDS = (
        "topic_tags", "question_intent_tags", "dependencies",
        "reusable_for_questions", "keywords", "typical_questions",
        "related_methods",
    )

    results: list[tuple[float, dict]] = []
    for ku_id in top_ids:
        ku = ku_map.get(ku_id)
        if not ku:
            continue
        d = {"id": ku.id}
        d.update({c: getattr(ku, c) for c in _UNIT_FIELDS})
        d.update({c: getattr(ku, c) or [] for c in _LIST_FIELDS})
        if ku.paper_id:
            d["paper_id"] = ku.paper_id
        if ku.paper_id and ku.paper_id in paper_map:
            paper = paper_map[ku.paper_id]
            d["_domain"] = paper.domain
            d["_paper_title"] = paper.title
            d["_paper_authors"] = paper.authors
            d["_paper_year"] = paper.year
            d["_paper_doi"] = paper.doi
        score = rrf_scores[ku_id]
        d["_similarity"] = score
        results.append((score, d))

    results.sort(key=lambda x: x[0], reverse=True)
    return results
