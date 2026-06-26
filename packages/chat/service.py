"""Chat service - routes questions to skills, retrieves relevant knowledge units, generates responses.

Response strategy:
  - direct_answer: One unit matched with high confidence -> LLM answers grounded in that unit
  - comparison:    Multiple units matched -> LLM synthesizes across units
  - llm_only:     No good match in knowledge base -> LLM answers from its own knowledge
"""

import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

from .skill_loader import load_skills
from .router import classify_question
from .embeddings import compute_embedding, cosine_similarity, _score_methods, hybrid_search
from .method_skills import format_skills_for_selection

_skills = load_skills()

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
MODEL_HEAVY = "gpt-4o"       # reranker + final generation (quality-critical)
MODEL_LIGHT = "gpt-4o-mini"  # routing, classification, rewriting (structured tasks)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
SCORE_DIRECT = 0.75   # single high-confidence hit -> direct answer
SCORE_FLOOR = 0.50    # minimum score to be considered relevant
MAX_COMPARE = 5       # cap for comparison mode

# ---------------------------------------------------------------------------
# Strategy-specific system prompts
# ---------------------------------------------------------------------------
DIRECT_ANSWER_PROMPT = """You are a statistical research assistant providing expert-level answers.

## How to respond
- Lead with a clear, direct answer to the user's question.
- Include mathematical formulas ($...$) when the topic involves a method.
- Describe algorithm steps when relevant.
- Give a concrete example or scenario when it helps understanding.
- Use the matched knowledge units as your primary evidence, supplementing with your own expertise when needed.

## Formatting
- Use markdown naturally: headers, bullet points, equations, code blocks.
- Let the content dictate the structure — do NOT force a rigid numbered template.
- Do NOT use comparison tables unless the user is comparing multiple methods.
- Keep the response focused and proportional to the question's complexity.

## Important
- Users are researchers who need actionable, technical detail — not just conceptual overviews.
- Respond in the same language the user uses. Only use Traditional Chinese (繁體中文) or English.
"""

FOLLOWUP_PROMPT = """You are a statistical research assistant in an ongoing conversation.

You are continuing a multi-turn discussion. Respond like a knowledgeable colleague — direct, natural, and building on what was already said.

## Rules
- Do NOT restate the user's question. Just answer it.
- Do NOT use the rigid numbered structure (no "1. Restate the question, 2. Give recommendation...").
- Do NOT produce a comparison table unless the user explicitly asks to compare methods.
- Build on the previous discussion — don't repeat information already given.
- If the user asks for clarification, clarify directly and concisely.
- If the user asks a new but related question, answer it naturally.
- Use markdown formatting (headers, bullet points, equations) where helpful, but let the content dictate the structure.
- Include formulas ($...$) when they add clarity, but don't force them in.
- If the user's message contains pronouns or references like "it", "that", "this method", "the above", resolve them from the conversation history before answering. Do NOT ask the user to clarify what they meant — you have the history.
- If the message is brief (e.g., "Give me code", "What are the assumptions?", "How does it compare?"), infer what the user is referring to from the previous turn and answer it directly.

## Depth
- Match the depth to what the user is asking. A simple "yes/no" question deserves a short answer, not a dissertation.
- If they ask for more detail, go deep. If they ask a quick follow-up, keep it brief.

## Knowledge base
- The conversation history establishes the topic. Knowledge units below are supplementary evidence.
- If the knowledge units are about a different method or topic than what is being discussed in the history, prioritize your own knowledge of the conversation topic over the knowledge units.
- If matched knowledge units are directly relevant, use them as evidence and cite inline with [1], [2] etc.

## Language
- Respond in the same language the user uses. Only use Traditional Chinese (繁體中文) or English.
"""

COMPARISON_PROMPT = """You are a statistical research assistant providing expert-level answers.

## How to respond
- Give your recommendation upfront: which approach best fits the user's situation and why.
- Compare the matched methods on dimensions relevant to the question: formulas, assumptions, when to use each, strengths, limitations.
- Include mathematical formulations ($...$) to show HOW methods differ technically.
- Use a concrete scenario to illustrate trade-offs when helpful.

## Formatting
- Use markdown naturally: headers, bullet points, equations.
- A comparison table is appropriate ONLY when the user is explicitly comparing 2-3 methods. For general questions that happen to match multiple knowledge units, just answer the question directly.
- Keep the response focused. Not every answer needs to be exhaustive.

## Important
- Lead with the answer, not the knowledge descriptions.
- If the knowledge base does not cover a method, clearly state this.
- Respond in the same language the user uses. Only use Traditional Chinese (繁體中文) or English.
"""

LLM_ONLY_PROMPT = """\
You are a statistical research assistant providing expert-level answers.

## How to respond
- Give a direct, thorough answer grounded in established statistical theory and practice.
- Include relevant formulas ($...$), algorithmic steps, and concrete examples where appropriate.
- If the question is about choosing a method, ask clarifying questions but still provide a tentative recommendation.
- Be honest about uncertainty.

## Formatting
- Use markdown naturally: headers, bullet points, equations.
- Let the content dictate the structure — do NOT force a rigid template.
- Do NOT use comparison tables unless the user is explicitly comparing methods.

## Important
- No matching knowledge was found in the curated knowledge library. Briefly note this at the end.
- Even without matched knowledge, provide technical depth.
- Respond in the same language the user uses. Only use Traditional Chinese (繁體中文) or English.
"""

CLARIFY_PROMPT = """You are a statistical research assistant. The user's question is too vague to give a useful answer.

Based on the conversation context, ask ONE focused clarifying question to understand what they need.

Rules:
- Ask about the most important missing piece (data type? goal? specific method?)
- Keep it short and conversational
- Suggest 2-3 concrete options when possible (e.g., "Are you looking for A, B, or C?")
- Respond in the same language the user uses. Only use Traditional Chinese (繁體中文) or English.
"""

CITATION_INSTRUCTION = """
## Citation Rules
- When using information from the knowledge base, cite the source using [1], [2], etc.
- The numbers correspond to the knowledge units listed below.
- Place citations inline where the information is used, e.g. "The method achieves O(n log n) convergence [1]."
- You may cite multiple sources: [1][3] or [1, 3].
- Only cite sources you actually use. Do not cite all sources just to be thorough.
- If you supplement with your own knowledge, do not add a citation for that part.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_knowledge_context(units: list[dict]) -> str:
    """Build a formatted knowledge base string from knowledge unit dicts."""
    if not units:
        return ""

    sections = []
    for i, u in enumerate(units, 1):
        method = u.get("method_name") or u.get("title", "Unknown")
        ktype = u.get("knowledge_type", "")
        # Add source info from paper metadata
        source_parts = []
        if u.get("_paper_authors"):
            source_parts.append(u["_paper_authors"].split(",")[0].strip() + " et al.")
        if u.get("_paper_year"):
            source_parts.append(str(u["_paper_year"]))
        source = f" — {', '.join(source_parts)}" if source_parts else ""

        parts = [f"### [{i}] {method} [{ktype}]{source}"]
        if u.get("field"):
            parts.append(f"**Field:** {u['field']}")
        if u.get("problem_it_solves"):
            parts.append(f"**Problem:** {u['problem_it_solves']}")
        parts.append(f"**Content:** {u['content']}")
        if u.get("model_assumption"):
            parts.append(f"**Assumptions:** {u['model_assumption']}")
        if u.get("input_format"):
            parts.append(f"**Input:** {u['input_format']}")
        if u.get("output_format"):
            parts.append(f"**Output:** {u['output_format']}")
        if u.get("limitations"):
            parts.append(f"**Limitations:** {u['limitations']}")
        if u.get("evidence_span"):
            parts.append(f"**Evidence:** {u['evidence_span']}")
        if u.get("confidence"):
            parts.append(f"**Confidence:** {u['confidence']}")
        if u.get("related_methods"):
            parts.append(f"**Related methods:** {', '.join(u['related_methods'])}")
        if u.get("_similarity") is not None:
            parts.append(f"*(similarity: {u['_similarity']})*")
        sections.append("\n".join(parts))

    return "\n\n---\n\n".join(sections)

# ---------------------------------------------------------------------------
# Step 2.75: Scoped Section Retrieval
# ---------------------------------------------------------------------------

SECTION_SELECT_PROMPT = """You are selecting which paper sections to include as reference material for answering a research question.

## User question
{question}

## Available sections
{section_list}

## Task
Select the 2-3 sections most likely to contain information needed to answer the question.
Return ONLY a JSON array of section numbers, e.g. [1, 3, 5].
If no section is relevant, return [].
"""


def _select_paper_sections(
    question: str,
    matched_units: list[dict],
    db,
    api_key: str,
) -> list[dict]:
    """Retrieve relevant paper sections from papers linked to matched KUs.

    Steps:
    1. Collect paper_ids from matched KUs
    2. Query paper_sections for those papers (summary + section_type + paper title)
    3. LLM selects 2-3 most relevant sections based on summaries
    4. Fetch full content of selected sections, truncate to ~3000 chars each

    Returns list of dicts with keys: paper_title, section_type, content
    """
    if not matched_units or db is None:
        return []

    # Step 1: Collect paper_ids
    paper_ids = set()
    for u in matched_units:
        pid = u.get("paper_id")
        if pid is not None:
            paper_ids.add(pid)

    if not paper_ids:
        return []

    # Step 2: Query sections and paper titles
    from models import PaperSection, Paper

    sections = (
        db.query(PaperSection, Paper.title)
        .join(Paper, PaperSection.paper_id == Paper.id)
        .filter(PaperSection.paper_id.in_(paper_ids))
        .order_by(PaperSection.paper_id, PaperSection.section_index)
        .all()
    )

    if not sections:
        return []

    # Build section list for LLM
    section_items = []
    for i, (sec, paper_title) in enumerate(sections, 1):
        section_items.append({
            "index": i,
            "id": sec.id,
            "paper_title": paper_title,
            "section_type": sec.section_type,
            "summary": sec.summary,
            "content": sec.content,
        })

    # If 3 or fewer sections, just use all of them (no need for LLM selection)
    if len(section_items) <= 3:
        selected = section_items
    else:
        # Step 3: LLM selection
        lines = []
        for item in section_items:
            lines.append(f"[{item['index']}] Paper: \"{item['paper_title']}\" — Section: {item['section_type']} — {item['summary']}")
        section_list_text = "\n".join(lines)

        prompt = SECTION_SELECT_PROMPT.replace("{question}", question).replace("{section_list}", section_list_text)

        client = OpenAI(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                raw = raw.rsplit("```", 1)[0]
            chosen_indices = json.loads(raw)
            if not isinstance(chosen_indices, list):
                chosen_indices = []
        except Exception as e:
            logger.warning(f"[Chat] Section selection failed: {e}")
            # Fallback: take top 3 by order
            chosen_indices = [item["index"] for item in section_items[:3]]

        index_set = set(chosen_indices)
        selected = [item for item in section_items if item["index"] in index_set]
        if not selected:
            selected = section_items[:3]

    # Step 4: Build result with truncated content
    MAX_SECTION_CHARS = 3000
    result = []
    for item in selected[:3]:
        content = item["content"]
        if len(content) > MAX_SECTION_CHARS:
            content = content[:MAX_SECTION_CHARS] + "\n... (truncated)"
        result.append({
            "paper_title": item["paper_title"],
            "section_type": item["section_type"],
            "content": content,
        })

    return result


def _build_section_context(sections: list[dict]) -> str:
    """Build formatted context string from selected paper sections."""
    if not sections:
        return ""

    parts = ["## Original Paper Sections\nThe following are relevant sections from matched papers, provided as additional reference.\n"]
    for sec in sections:
        parts.append(f"### Paper: \"{sec['paper_title']}\" — {sec['section_type']}\n{sec['content']}")

    return "\n\n".join(parts)


SCORE_FLOOR_FILTERED = 0.40  # lower threshold when method pre-filtering already narrowed scope


def _select_strategy(
    scored: list[tuple[float, dict]],
    pre_filtered: bool = False,
    hybrid: bool = False,
) -> tuple[str, list[dict], str]:
    """Determine response strategy based on similarity scores.

    Returns (strategy_name, selected_units, system_prompt).
    When pre_filtered=True, uses a lower score floor since the search pool
    is already scoped to relevant methods.
    When hybrid=True, scores are RRF values (0-0.05 range) instead of cosine
    similarity (0-1 range), so much lower thresholds are used.
    """
    if hybrid:
        # RRF scores with K=60: rank-1 in one path ≈ 0.016, rank-1 in both ≈ 0.033
        floor = 0.005
        direct_threshold = 0.025
    else:
        floor = SCORE_FLOOR_FILTERED if pre_filtered else SCORE_FLOOR
        direct_threshold = SCORE_DIRECT

    high = [(s, m) for s, m in scored if s >= floor]

    if not high:
        return "llm_only", [], LLM_ONLY_PROMPT

    if len(high) == 1 and high[0][0] >= direct_threshold:
        unit = dict(high[0][1])
        unit["_similarity"] = high[0][0]
        return "direct_answer", [unit], DIRECT_ANSWER_PROMPT

    # Multiple relevant units (or single above floor but below direct threshold)
    selected = []
    for s, m in high[:MAX_COMPARE]:
        mc = dict(m)
        mc["_similarity"] = s
        selected.append(mc)
    return "comparison", selected, COMPARISON_PROMPT


# ---------------------------------------------------------------------------
# Taxonomy helpers
# ---------------------------------------------------------------------------

def _taxonomy_locate(
    query_embedding: list[float],
    taxonomy_nodes: list[dict],
    top_k: int = 3,
    db=None,
) -> set[int]:
    """Locate the most relevant taxonomy branches and return linked knowledge unit IDs.

    When db is provided and is PostgreSQL, uses pgvector for efficient DB-side
    similarity search instead of loading all embeddings into Python.

    Falls back to in-memory cosine similarity when db is not available (tests).
    """
    if not query_embedding:
        return set()

    # --- Fast path: pgvector DB-side search ---
    if db is not None:
        _is_pg = getattr(getattr(db, "bind", None), "dialect", None) is not None and db.bind.dialect.name == "postgresql"
        if _is_pg:
            return _taxonomy_locate_pg(query_embedding, db, top_k)

    # --- Fallback: in-memory cosine similarity ---
    if not taxonomy_nodes:
        return set()

    node_by_id: dict[int, dict] = {n["id"]: n for n in taxonomy_nodes}

    scored: list[tuple[float, dict]] = []
    for node in taxonomy_nodes:
        emb = node.get("embedding")
        if not emb:
            continue
        sim = cosine_similarity(query_embedding, emb)
        scored.append((sim, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_nodes = scored[:top_k]

    boost_ids: set[int] = set()

    def _collect_descendant_kus(node_id: int) -> None:
        node = node_by_id.get(node_id)
        if node is None:
            return
        for ku_id in node.get("knowledge_unit_ids", []):
            boost_ids.add(ku_id)
        for child_id in node.get("children_ids", []):
            _collect_descendant_kus(child_id)

    for _score, matched_node in top_nodes:
        _collect_descendant_kus(matched_node["id"])

    return boost_ids


def _taxonomy_locate_pg(
    query_embedding: list[float],
    db,
    top_k: int = 3,
) -> set[int]:
    """Use pgvector to find top-K taxonomy nodes, then collect descendant KU IDs via SQL."""
    from sqlalchemy import text as sa_text

    emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    # Find top-K taxonomy nodes by vector similarity
    top_node_rows = db.execute(
        sa_text("""
            SELECT id FROM method_taxonomy
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :emb
            LIMIT :k
        """),
        {"emb": emb_str, "k": top_k},
    ).fetchall()

    if not top_node_rows:
        return set()

    seed_ids = [row[0] for row in top_node_rows]

    # Recursively collect all descendant node IDs using a CTE
    placeholders = ",".join(str(i) for i in seed_ids)
    descendant_rows = db.execute(
        sa_text(f"""
            WITH RECURSIVE tree AS (
                SELECT id FROM method_taxonomy WHERE id IN ({placeholders})
                UNION ALL
                SELECT mt.id FROM method_taxonomy mt
                JOIN tree t ON mt.parent_id = t.id
            )
            SELECT DISTINCT kun.knowledge_unit_id
            FROM tree
            JOIN knowledge_unit_nodes kun ON kun.method_node_id = tree.id
        """),
    ).fetchall()

    return {row[0] for row in descendant_rows}


def _get_sibling_methods(
    matched_units: list[dict],
    taxonomy_nodes: list[dict],
) -> list[dict]:
    """Find sibling methods for the matched knowledge units.

    For each matched unit, find its taxonomy node(s), then find sibling nodes
    (nodes sharing the same parent_id). Returns a unique list of sibling info.

    Each taxonomy node dict may have:
      - id, name, node_type, parent_id
      - knowledge_unit_ids: list[int]
    """
    if not matched_units or not taxonomy_nodes:
        return []

    node_by_id: dict[int, dict] = {n["id"]: n for n in taxonomy_nodes}

    # Build reverse mapping: knowledge_unit_id -> list of node IDs
    ku_to_nodes: dict[int, list[int]] = {}
    for node in taxonomy_nodes:
        for ku_id in node.get("knowledge_unit_ids", []):
            ku_to_nodes.setdefault(ku_id, []).append(node["id"])

    # Collect node IDs associated with matched units
    matched_node_ids: set[int] = set()
    for unit in matched_units:
        uid = unit.get("id")
        if uid is not None and uid in ku_to_nodes:
            matched_node_ids.update(ku_to_nodes[uid])

    # Find parent IDs of matched nodes
    parent_ids: set[int] = set()
    for nid in matched_node_ids:
        node = node_by_id.get(nid)
        if node and node.get("parent_id") is not None:
            parent_ids.add(node["parent_id"])

    # Collect sibling nodes (same parent, but not in matched set)
    seen: set[int] = set()
    siblings: list[dict] = []
    for node in taxonomy_nodes:
        if node.get("parent_id") in parent_ids and node["id"] not in matched_node_ids:
            if node["id"] not in seen:
                seen.add(node["id"])
                siblings.append({
                    "name": node.get("name", "unknown"),
                    "node_type": node.get("node_type", "unknown"),
                })

    return siblings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _multi_query_search(
    queries: list[str],
    method_context: list[dict],
    api_key: str,
    domain: str | None = None,
    boost_ids: set[int] | None = None,
) -> list[tuple[float, dict]]:
    """Run embedding search for each query, merge results keeping best score per unit.

    When boost_ids is provided, units whose 'id' is in the set receive a 30%
    taxonomy boost (score *= 1.3) to surface taxonomy-relevant results.
    """
    best: dict[int, tuple[float, dict]] = {}  # unit id -> (best_score, unit)

    for q in queries:
        if not q.strip():
            continue
        emb = compute_embedding(q, api_key)
        scored = _score_methods(emb, method_context, domain=domain)
        for score, unit in scored:
            # Apply taxonomy boost if this unit is in a matched taxonomy branch
            if boost_ids and unit.get("id") in boost_ids:
                score = round(score * 1.3, 4)
            uid = id(unit)  # use object id as key since units are same dicts
            if uid not in best or score > best[uid][0]:
                best[uid] = (score, unit)

    merged = sorted(best.values(), key=lambda x: x[0], reverse=True)
    return merged



# ---------------------------------------------------------------------------
# Step 1.5: Question analysis + method selection
# ---------------------------------------------------------------------------

QUESTION_ANALYSIS_PROMPT = """You are a research question analyzer.

Given a user question about statistical methods, extract structured information.

Return ONLY a JSON object:
{
  "methods_mentioned": ["any method names, abbreviations, or technique names mentioned"],
  "intent": "what the user wants (explain, compare, apply, troubleshoot, choose)",
  "domain": "statistical field if identifiable (e.g. regression, causal inference, spatial statistics)",
  "key_concepts": ["key technical concepts or criteria mentioned"]
}

Rules:
- Extract ALL method names, even partial ones (e.g. "HDIC" from "HDIC vs BIC").
- Include abbreviations AND full names if both appear.
- Be generous with method extraction - if it could be a method name, include it.
- Return ONLY valid JSON, no extra text."""


METHOD_MATCH_PROMPT = """You are a method selection assistant for a statistical research Q&A system.

## Question analysis:
{question_analysis}

## Available methods:
{method_catalog}

## Rules:
- Match the question analysis to the available methods.
- A method is relevant if:
  1. Any of methods_mentioned appears in its name or aliases, OR
  2. Its field matches the question domain, OR
  3. Its purpose relates to the user intent + key_concepts
- Return a JSON array of method names (use the exact name from the catalog).
- Select 1-3 methods maximum.
- If NO method is relevant, return an empty array [].
- Return ONLY the JSON array, no extra text.

Example: ["OGA+HDIC+Trim", "Double Machine Learning (DML)"]"""


def _analyze_question(
    message: str,
    api_key: str,
    history: list[dict[str, str]] | None = None,
) -> dict:
    """Analyze user question into structured form (methods, intent, domain, concepts)."""
    client = OpenAI(api_key=api_key)
    msgs: list[dict[str, str]] = [{"role": "system", "content": QUESTION_ANALYSIS_PROMPT}]
    if history:
        for h in history[-4:]:
            msgs.append(h)
    msgs.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0,
            max_tokens=200,
        )
    except Exception as e:
        logger.warning(f"[Chat] Question analysis failed: {e}")
        return {"methods_mentioned": [], "intent": "", "domain": "", "key_concepts": []}
    raw = (resp.choices[0].message.content or "").strip()

    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return {"methods_mentioned": [], "intent": "", "domain": "", "key_concepts": []}


def _select_methods(
    message: str,
    method_skills: list[dict],
    api_key: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[list[str], dict]:
    """Analyze question then select relevant methods from skill catalog.

    Returns (selected_method_names, question_analysis).
    """
    if not method_skills:
        return [], {}

    # Step A: Analyze the question
    analysis = _analyze_question(message, api_key, history)

    # Step B: Use analysis to match methods
    catalog = format_skills_for_selection(method_skills)
    analysis_text = json.dumps(analysis, ensure_ascii=False)
    system = METHOD_MATCH_PROMPT.replace("{question_analysis}", analysis_text)
    system = system.replace("{method_catalog}", catalog)

    client = OpenAI(api_key=api_key)
    msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    msgs.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0,
            max_tokens=150,
        )
    except Exception as e:
        logger.warning(f"[Chat] Method selection failed: {e}")
        return [], analysis
    raw = (resp.choices[0].message.content or "").strip()

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(m) for m in result], analysis
    except (json.JSONDecodeError, ValueError):
        pass
    return [], analysis


def _filter_units_by_methods(
    units: list[dict],
    selected_methods: list[str],
    method_skills: list[dict],
) -> list[dict]:
    """Filter knowledge units to those belonging to selected methods.

    Matching strategy:
    1. Name match: unit method_name matches canonical name or alias (substring)
    2. Field match: unit field matches the selected method's field
       (catches sub-concepts like Neyman Orthogonality -> DML via causal inference)
    """
    if not selected_methods:
        return units

    # Build allowed names and fields for selected methods
    allowed_names: set[str] = set()
    allowed_fields: set[str] = set()
    for ms in method_skills:
        if ms.get("method") in selected_methods:
            allowed_names.add(ms["method"].lower())
            for alias in ms.get("aliases", []):
                allowed_names.add(alias.lower())
            field = (ms.get("field") or "").lower().strip()
            if field:
                allowed_fields.add(field)

    for m in selected_methods:
        allowed_names.add(m.lower())

    filtered = []
    for u in units:
        name = (u.get("method_name") or u.get("title") or "").lower()
        field = (u.get("field") or "").lower().strip()
        # Match by name (substring in either direction)
        if any(a in name or name in a for a in allowed_names if a):
            filtered.append(u)
        # Match by field (sub-concepts share the same field as primary method)
        elif field and field in allowed_fields:
            filtered.append(u)

    return filtered if filtered else units


def _expand_queries_with_skills(
    search_queries: list[str],
    method_skills: list[dict],
) -> tuple[list[str], list[str]]:
    """Expand search queries using MethodSkill aliases and related methods.

    Returns:
        (vector_queries, keyword_terms)
        - vector_queries: original search queries (for embedding search)
        - keyword_terms: extra method names/aliases (for full-text search only)
    """
    vector_queries = [q for q in search_queries if q.strip()]
    keyword_terms: list[str] = []
    seen_terms: set[str] = set()

    for query in vector_queries:
        query_lower = query.lower()
        for skill in method_skills:
            method = skill.get("method", "")
            aliases = skill.get("aliases", [])
            related = skill.get("related_methods", [])
            field = skill.get("field", "")

            # Check if any skill name/alias/field appears in the query
            all_names = [method] + aliases + [field]
            matched = any(
                name.lower() in query_lower or query_lower in name.lower()
                for name in all_names if name
            )
            if not matched:
                continue

            # Collect aliases and related methods as keyword terms
            for term in [method] + aliases + related:
                term = term.strip()
                if term and term.lower() not in seen_terms:
                    seen_terms.add(term.lower())
                    keyword_terms.append(term)

    return vector_queries, keyword_terms


def _compute_method_boost_ids(
    db,
    selected_methods: list[str],
    method_skills: list[dict],
) -> set[int]:
    """Return KU IDs matching selected methods, for score boosting (not filtering).

    Uses SQL ILIKE to filter in the database instead of loading all KU rows.
    """
    if not selected_methods:
        return set()

    from sqlalchemy import text as sa_text

    # Build allowed names and fields from skills
    allowed_names: set[str] = set()
    allowed_fields: set[str] = set()
    for ms in method_skills:
        if ms.get("method") in selected_methods:
            allowed_names.add(ms["method"].lower())
            for alias in ms.get("aliases", []):
                allowed_names.add(alias.lower())
            field = (ms.get("field") or "").lower().strip()
            if field:
                allowed_fields.add(field)
    for m in selected_methods:
        allowed_names.add(m.lower())

    # Build SQL conditions
    conditions = []
    params: dict = {}
    for i, name in enumerate(allowed_names):
        if name:
            conditions.append(f"LOWER(method_name) LIKE :n{i}")
            params[f"n{i}"] = f"%{name}%"
    for i, field in enumerate(allowed_fields):
        if field:
            conditions.append(f"LOWER(field) = :f{i}")
            params[f"f{i}"] = field

    if not conditions:
        return set()

    where = " OR ".join(conditions)
    rows = db.execute(
        sa_text(f"SELECT id FROM knowledge_units WHERE {where}"),
        params,
    ).fetchall()

    return {row[0] for row in rows}


def _call_dify_workflow(
    question: str,
    knowledge_context: str,
    history: str,
    strategy: str,
    dify_api_key: str,
    dify_base_url: str = "https://api.dify.ai/v1",
) -> str:
    """Call Dify workflow API to generate the final answer."""
    import httpx

    url = f"{dify_base_url}/workflows/run"
    headers = {
        "Authorization": f"Bearer {dify_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": {
            "question": question,
            "knowledge_context": knowledge_context or "",
            "history": history or "[]",
            "strategy": strategy,
        },
        "response_mode": "blocking",
        "user": "fastapi-backend",
    }

    logger.info(f"[Chat] Calling Dify workflow (strategy={strategy})...")
    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    resp.raise_for_status()

    data = resp.json()
    # Dify returns {"data": {"outputs": {"answer": "..."}}}
    outputs = data.get("data", {}).get("outputs", {})
    answer = outputs.get("answer", "")
    if not answer:
        # Try alternative output format
        answer = outputs.get("text", data.get("data", {}).get("text", "No response from Dify."))
    return answer


REWRITE_PROMPT = """You are a query rewriter for a statistical research assistant.

Given the conversation history (if any) and the user's latest message, rewrite the message into a complete, self-contained question.

Rules:
- Resolve all pronouns and references (e.g., "that method" -> the actual method name from history)
- ALWAYS translate the final question into English, regardless of the input language
- Keep the rewritten question concise
- If the message is already self-contained and in English, return it unchanged
- Return ONLY the rewritten question in English, no explanation"""


def _rewrite_query(
    message: str,
    history: list[dict[str, str]],
    api_key: str,
) -> str:
    """Rewrite a follow-up question into a self-contained English query."""
    # Skip rewrite only if no history AND message is already English
    if not history and message.isascii():
        return message

    client = OpenAI(api_key=api_key)
    msgs: list[dict[str, str]] = [{"role": "system", "content": REWRITE_PROMPT}]
    # Include last 6 messages of history for context
    for h in (history or [])[-6:]:
        msgs.append(h)
    msgs.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0,
            max_tokens=200,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten if rewritten else message
    except Exception:
        return message


# ---------------------------------------------------------------------------
# Step 2.5: LLM Reranker
# ---------------------------------------------------------------------------

RERANK_PROMPT = """\
You are a relevance judge for a statistical research Q&A system.

## User question
{question}

## Candidate knowledge units
{candidates}

## Task
Rate how relevant each candidate is to the user's question.
- 10 = directly answers the question or describes the exact method asked about
- 7-9 = highly relevant (same method family, directly applicable concept)
- 4-6 = somewhat relevant (related field, might provide useful context)
- 1-3 = marginally relevant (tangential connection)
- 0 = not relevant at all

Return ONLY a JSON array of objects, one per candidate, in the same order:
[{"id": 1, "score": 8, "reason": "brief reason"}, ...]

Important:
- Be strict: generic statistical concepts that don't specifically address the question should score low
- If the user asks about a specific method, only rate highly units that discuss THAT method
- If the user asks a broad question, rate highly units that cover the relevant topic area
- Return valid JSON only, no extra text"""


RERANK_CANDIDATES = 20  # how many hybrid results to send to the reranker
RERANK_TOP_K = 5        # how many to keep after reranking


def _rerank_with_llm(
    question: str,
    scored: list[tuple[float, dict]],
    api_key: str,
    n_candidates: int = RERANK_CANDIDATES,
    top_k: int = RERANK_TOP_K,
) -> list[tuple[float, dict]]:
    """Rerank hybrid search results using an LLM relevance judge.

    Takes the top *n_candidates* from scored results, sends abbreviated
    info to gpt-4o-mini for relevance scoring, then returns the top *top_k*
    reranked results with normalised scores.

    Falls back to the original scored list (truncated) on any error.
    """
    if not scored:
        return []

    candidates = scored[:n_candidates]

    # Build compact summaries for the prompt
    lines = []
    for i, (rrf_score, u) in enumerate(candidates, 1):
        title = u.get("title") or "Untitled"
        method = u.get("method_name") or ""
        ktype = u.get("knowledge_type") or ""
        field = u.get("field") or ""
        content = (u.get("content") or "")[:800]
        tags = ", ".join((u.get("topic_tags") or [])[:5])
        evidence = (u.get("evidence_span") or "")[:200]
        problem = u.get("problem_it_solves") or ""

        parts = [f"[{i}] **{title}**"]
        if method:
            parts.append(f"Method: {method}")
        if ktype:
            parts.append(f"Type: {ktype}")
        if field:
            parts.append(f"Field: {field}")
        if problem:
            parts.append(f"Problem: {problem}")
        if tags:
            parts.append(f"Tags: {tags}")
        parts.append(f"Content: {content}")
        if evidence:
            parts.append(f"Evidence: {evidence}")
        lines.append(" | ".join(parts))

    candidate_text = "\n".join(lines)
    system = RERANK_PROMPT.replace("{question}", question).replace("{candidates}", candidate_text)

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=MODEL_HEAVY,
            messages=[{"role": "system", "content": system}],
            temperature=0,
            max_tokens=800,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # Parse JSON — handle markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]

        rankings = json.loads(raw)
        if not isinstance(rankings, list):
            raise ValueError("Expected JSON array")

        # Map back: ranking index (1-based) -> LLM score
        llm_scores: dict[int, float] = {}
        for item in rankings:
            idx = item.get("id", 0)
            score = item.get("score", 0)
            if isinstance(idx, int) and 1 <= idx <= len(candidates):
                llm_scores[idx] = float(score)

        # Build reranked list sorted by LLM score descending
        reranked = []
        for i, (rrf_score, u) in enumerate(candidates, 1):
            llm_score = llm_scores.get(i, 0)
            if llm_score >= 4:  # only keep if somewhat relevant
                u_copy = dict(u)
                u_copy["_llm_relevance"] = llm_score
                # Use normalised LLM score (0-1) as the new score
                reranked.append((llm_score / 10.0, u_copy))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k] if reranked else scored[:top_k]

    except Exception as e:
        logger.warning(f"[Chat] LLM rerank failed: {e!r}")
        return scored[:top_k]


def generate_response(
    message: str,
    api_key: str,
    db=None,
    history: list[dict[str, str]] | None = None,
    method_context: list[dict] | None = None,
    method_skills: list[dict] | None = None,
    taxonomy_nodes: list[dict] | None = None,
    dify_api_key: str | None = None,
    dify_base_url: str = "https://api.dify.ai/v1",
) -> tuple[str, str, list[dict], str]:
    """Classify question -> retrieve knowledge -> select strategy -> generate response."""
    result = _prepare_generation_context(
        message, api_key, db=db, history=history,
        method_context=method_context, method_skills=method_skills,
        taxonomy_nodes=taxonomy_nodes,
    )

    # Clarify mode
    if result[1] == "clarify":
        _, _, clarify_text, debug_lines, matched_units = result
        return clarify_text, chr(10).join(debug_lines), matched_units, "clarify"

    msgs, strategy, knowledge_section, debug_lines, matched_units = result

    # Try Dify first
    if dify_api_key:
        try:
            history_text = json.dumps(history or [], ensure_ascii=False)
            answer = _call_dify_workflow(
                question=message,
                knowledge_context=knowledge_section,
                history=history_text,
                strategy=strategy,
                dify_api_key=dify_api_key,
                dify_base_url=dify_base_url,
            )
            debug_lines.append("LLM backend: **Dify Workflow**")
            return answer, chr(10).join(debug_lines), matched_units, strategy
        except Exception as e:
            logger.warning(f"[Chat] Dify call failed: {e}, falling back to OpenAI")
            debug_lines.append(f"LLM backend: **Dify FAILED ({e}), fell back to OpenAI**")

    # Fallback: direct OpenAI
    gen_params = {
        "direct_answer": {"temperature": 0.3, "max_tokens": 1500},
        "comparison":    {"temperature": 0.5, "max_tokens": 2000},
        "llm_only":      {"temperature": 0.7, "max_tokens": 1000},
    }.get(strategy, {"temperature": 0.7, "max_tokens": 1000})
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=MODEL_HEAVY, messages=msgs, **gen_params)
    answer = resp.choices[0].message.content or "No response generated."
    debug_lines.append("LLM backend: **Direct OpenAI**")

    logger.info("[Chat] Done.")
    return answer, chr(10).join(debug_lines), matched_units, strategy


def _prepare_generation_context(
    message: str,
    api_key: str,
    db=None,
    history: list[dict[str, str]] | None = None,
    method_context: list[dict] | None = None,
    method_skills: list[dict] | None = None,
    taxonomy_nodes: list[dict] | None = None,
) -> tuple:
    """Run the classification/retrieval pipeline, return everything needed for generation.

    Returns either:
    - (None, "clarify", clarify_text, debug_lines, []) for clarification mode
    - (msgs, strategy, knowledge_section, debug_lines, matched_units) for normal generation
    """
    effective_message = _rewrite_query(message, history, api_key)
    if effective_message != message:
        logger.info(f"[Chat] Step 0: Rewrote query: '{message}' -> '{effective_message}'")

    logger.info("[Chat] Step 1: Classifying question...")
    route = classify_question(effective_message, _skills, api_key, history)

    queries = getattr(route, "search_queries", []) or [route.search_query]
    queries = [q for q in queries if q.strip()]

    # Fallback: if classifier returned no search queries, use the message itself
    # so we always attempt knowledge base retrieval (except for obvious greetings)
    if not queries and len(effective_message.split()) > 2:
        queries = [effective_message]
        logger.info("[Chat] No search queries from classifier, using message as fallback query")

    debug_lines = [
        f"Skill: **{route.skill}**",
        f"Search queries: *{queries or '(none)'}*",
        f"Confidence: **{route.confidence}**",
    ]
    if effective_message != message:
        debug_lines.append(f"Rewritten query: *{effective_message}*")

    CONFIDENCE_THRESHOLD = 0.5
    if route.confidence < CONFIDENCE_THRESHOLD and not history:
        logger.info(f"[Chat] Low confidence ({route.confidence}), generating clarification...")
        client = OpenAI(api_key=api_key)
        clarify_msgs: list[dict[str, str]] = [{"role": "system", "content": CLARIFY_PROMPT}]
        if history:  # include recent context for better clarification
            for h in history[-2:]:
                clarify_msgs.append(h)
        clarify_msgs.append({"role": "user", "content": message})
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=clarify_msgs, temperature=0.5, max_tokens=200
        )
        clarify_text = resp.choices[0].message.content or "Could you provide more details?"
        debug_lines.append(f"Confidence below {CONFIDENCE_THRESHOLD}, clarifying")
        return None, "clarify", clarify_text, debug_lines, []

    strategy = "llm_only"
    system_prompt = LLM_ONLY_PROMPT
    knowledge_section = ""
    matched_units: list[dict] = []

    # Step 1.5: Method selection (soft boost)
    logger.info("[Chat] Step 1.5: Selecting methods...")
    selected_methods: list[str] = []
    q_analysis: dict = {}
    method_boost_ids: set[int] = set()
    if method_skills:
        selected_methods, q_analysis = _select_methods(effective_message, method_skills, api_key, history)
        if selected_methods:
            debug_lines.append(f"Question analysis: {q_analysis}")
            debug_lines.append(f"Selected methods: **{selected_methods}**")
            # Soft boost instead of hard filter
            if db is not None:
                method_boost_ids = _compute_method_boost_ids(db, selected_methods, method_skills)
                debug_lines.append(f"Method boost: {len(method_boost_ids)} KUs")
        else:
            debug_lines.append(f"Question analysis: {q_analysis}")
            debug_lines.append("Method selection: no specific method matched, searching all KUs")

    # Append method purpose to queries (same as before)
    if selected_methods and method_skills:
        for ms in method_skills:
            if ms.get("method") in selected_methods:
                purpose = ms.get("purpose", "")
                method_name = ms.get("method", "")
                field = ms.get("field", "")
                if purpose:
                    queries.append(f"{method_name} {field} {purpose}")

    # Step 1.6: Query expansion with skill aliases
    vector_queries = queries
    keyword_terms: list[str] = []
    if method_skills:
        vector_queries, keyword_terms = _expand_queries_with_skills(queries, method_skills)
        if keyword_terms:
            debug_lines.append(f"Expanded keywords: {keyword_terms[:10]}")

    # Step 1.75: Taxonomy-based boosting
    boost_ids: set[int] | None = None
    if vector_queries and taxonomy_nodes:
        logger.info("[Chat] Step 1.75: Taxonomy locating...")
        first_query_emb = compute_embedding(vector_queries[0], api_key)
        if first_query_emb:
            boost_ids = _taxonomy_locate(first_query_emb, taxonomy_nodes, db=db)
            debug_lines.append(f"Taxonomy boost: {len(boost_ids)} units from matched branches")
        else:
            debug_lines.append("Taxonomy boost: skipped (embedding failed)")

    # Step 2: Retrieval
    logger.info("[Chat] Step 2: Retrieving knowledge units...")
    _is_pg = db is not None and getattr(getattr(db, "bind", None), "dialect", None) is not None and db.bind.dialect.name == "postgresql"
    use_hybrid = _is_pg and bool(vector_queries)
    if use_hybrid:
        # New hybrid search path (pgvector + tsvector + RRF)
        scored = hybrid_search(
            db=db,
            vector_queries=vector_queries,
            keyword_terms=keyword_terms,
            api_key=api_key,
            top_k=50,
            boost_ids=boost_ids,
            method_boost_ids=method_boost_ids if method_boost_ids else None,
        )
        debug_lines.append(f"Hybrid search: {len(scored)} results")
    elif method_context and queries:
        # Legacy in-memory path (fallback for tests or when db is not passed)
        scored = _multi_query_search(
            queries, method_context, api_key,
            boost_ids=boost_ids,
        )
    else:
        scored = []

    if scored:
        debug_lines.append("")
        debug_lines.append("**Knowledge unit scores (top results):**")
        for score, u in scored[:15]:
            title = u.get("title", "unknown")
            ktype = u.get("knowledge_type", "")
            tags = u.get("topic_tags", [])
            tag_str = f" -- tags: {', '.join(tags[:5])}" if tags else ""
            debug_lines.append(f"- {title} [{ktype}]: {score}{tag_str}")

        # Step 2.5: LLM reranking
        if use_hybrid and len(scored) > 3:
            logger.info("[Chat] Step 2.5: LLM reranking...")
            reranked = _rerank_with_llm(effective_message, scored, api_key)
            debug_lines.append("")
            debug_lines.append("**After LLM reranking:**")
            for score, u in reranked:
                title = u.get("title", "unknown")
                ktype = u.get("knowledge_type", "")
                llm_rel = u.get("_llm_relevance", "?")
                debug_lines.append(f"- {title} [{ktype}]: relevance={llm_rel}/10")

            if reranked:
                # Use reranked results directly (scores are already 0-1 normalised)
                matched_units = [u for _, u in reranked]
                knowledge_section = _build_knowledge_context(matched_units)
                # Pick strategy based on reranked scores
                top_score = reranked[0][0]
                if top_score >= 0.7 and len(reranked) == 1:
                    strategy = "direct_answer"
                    system_prompt = DIRECT_ANSWER_PROMPT
                elif reranked:
                    strategy = "comparison"
                    system_prompt = COMPARISON_PROMPT
            else:
                strategy = "llm_only"
                system_prompt = LLM_ONLY_PROMPT
        else:
            strategy, selected_units, system_prompt = _select_strategy(
                scored, pre_filtered=bool(selected_methods), hybrid=use_hybrid,
            )
            knowledge_section = _build_knowledge_context(selected_units)
            matched_units = selected_units

    # Step 2.75: Scoped section retrieval
    section_context = ""
    if matched_units and db is not None:
        logger.info("[Chat] Step 2.75: Retrieving paper sections...")
        paper_sections = _select_paper_sections(effective_message, matched_units, db, api_key)
        if paper_sections:
            section_context = _build_section_context(paper_sections)
            sec_info = [f"{s['paper_title']}/{s['section_type']}" for s in paper_sections]
            debug_lines.append(f"Paper sections included: {sec_info}")

    # Sibling method recommendations
    if matched_units and taxonomy_nodes:
        siblings = _get_sibling_methods(matched_units, taxonomy_nodes)
        if siblings:
            sibling_names = [f"{s['name']} ({s['node_type']})" for s in siblings]
            debug_lines.append(f"Related methods (taxonomy siblings): {', '.join(sibling_names)}")

    debug_lines.insert(1, f"Strategy: **{strategy}**")

    # Follow-up: override strategy prompt with conversational style
    if history:
        full_system = FOLLOWUP_PROMPT
        debug_lines.insert(2, "Mode: **follow-up** (conversational)")
    else:
        full_system = system_prompt
    if knowledge_section:
        full_system += CITATION_INSTRUCTION
        full_system += chr(10)*2 + "## Knowledge Base - Matched Units" + chr(10)*2 + knowledge_section
    if section_context:
        full_system += chr(10)*2 + section_context

    msgs: list[dict[str, str]] = [{"role": "system", "content": full_system}]
    if history:
        msgs.extend(history)
    # Use original message so LLM sees the user's language for response matching;
    # append the English translation as context if it differs
    if effective_message != message:
        user_content = message + chr(10) + chr(10) + "[English: " + effective_message + "]"
    else:
        user_content = message
    msgs.append({"role": "user", "content": user_content})

    return msgs, strategy, knowledge_section, debug_lines, matched_units


def generate_response_stream(
    message: str,
    api_key: str,
    db=None,
    history: list[dict[str, str]] | None = None,
    method_context: list[dict] | None = None,
    method_skills: list[dict] | None = None,
    taxonomy_nodes: list[dict] | None = None,
    dify_api_key: str | None = None,
    dify_base_url: str = "https://api.dify.ai/v1",
):
    """Streaming version of generate_response. Yields (event_type, data) tuples.

    Event types: "token", "debug", "references", "done", "error"
    """

    result = _prepare_generation_context(
        message, api_key, db=db, history=history,
        method_context=method_context, method_skills=method_skills,
        taxonomy_nodes=taxonomy_nodes,
    )

    # Clarify mode - no streaming needed
    if result[1] == "clarify":
        _, _, clarify_text, debug_lines, matched_units = result
        yield ("token", clarify_text)
        yield ("debug", chr(10).join(debug_lines))
        yield ("references", matched_units)
        yield ("done", (clarify_text, "clarify"))
        return

    msgs, strategy, knowledge_section, debug_lines, matched_units = result

    # Try Dify first (blocking, then emit as single chunk)
    if dify_api_key:
        try:
            history_text = json.dumps(history or [], ensure_ascii=False)
            answer = _call_dify_workflow(
                question=message,
                knowledge_context=knowledge_section,
                history=history_text,
                strategy=strategy,
                dify_api_key=dify_api_key,
                dify_base_url=dify_base_url,
            )
            debug_lines.append("LLM backend: **Dify Workflow**")
            yield ("token", answer)
            yield ("debug", chr(10).join(debug_lines))
            yield ("references", matched_units)
            yield ("done", (answer, strategy))
            return
        except Exception as e:
            logger.warning(f"[Chat] Dify call failed: {e}, falling back to OpenAI streaming")
            debug_lines.append(f"LLM backend: **Dify FAILED ({e}), fell back to OpenAI streaming**")

    # OpenAI streaming
    gen_params = {
        "direct_answer": {"temperature": 0.3, "max_tokens": 1500},
        "comparison":    {"temperature": 0.5, "max_tokens": 2000},
        "llm_only":      {"temperature": 0.7, "max_tokens": 1000},
    }.get(strategy, {"temperature": 0.7, "max_tokens": 1000})

    client = OpenAI(api_key=api_key)
    debug_lines.append("LLM backend: **OpenAI Streaming**")

    try:
        stream = client.chat.completions.create(
            model=MODEL_HEAVY, messages=msgs, stream=True, **gen_params
        )
        full_answer = []
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_answer.append(delta.content)
                yield ("token", delta.content)

        answer = "".join(full_answer) or "No response generated."
        yield ("debug", chr(10).join(debug_lines))
        yield ("references", matched_units)
        yield ("done", (answer, strategy))
    except Exception as e:
        logger.error(f"[Chat] OpenAI streaming failed: {e}")
        yield ("error", str(e))
