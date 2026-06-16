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
# Thresholds
# ---------------------------------------------------------------------------
SCORE_DIRECT = 0.75   # single high-confidence hit -> direct answer
SCORE_FLOOR = 0.50    # minimum score to be considered relevant
MAX_COMPARE = 5       # cap for comparison mode

# ---------------------------------------------------------------------------
# Strategy-specific system prompts
# ---------------------------------------------------------------------------
DIRECT_ANSWER_PROMPT = """You are a statistical research assistant.

## Response structure
Follow this order strictly:
1. **First, restate the user's core question or problem in one sentence** to confirm you understand what they are asking.
2. **Then, give a direct answer to their question.** If they asked "how to handle X", tell them how. If they asked "what is X", explain it. If they asked "should I use X", give a recommendation with reasoning.
3. **Use the matched knowledge below as supporting evidence** - connect the knowledge to the user's specific situation.
4. If the user's question goes beyond what the knowledge covers, you may supplement with your own knowledge but clearly mark it as such.

## Important
- The user's question comes FIRST. The knowledge is a tool to answer it, not the topic.
- Do NOT just describe the knowledge - explain how it solves the user's problem.
- If the knowledge doesn't fully address their question, say so and provide additional guidance.
- Use clear markdown formatting.
- If your answer draws on general knowledge beyond the matched units, clearly mark those parts as supplementary. Explicitly reference which knowledge unit you are drawing from.
- If the knowledge base lacks information on a topic, clearly state this limitation.
- Respond in the same language the user uses. Only use Traditional Chinese (繁體中文) or English. If the user writes in any other language, respond in English.

If your answer relies on general knowledge or lacks specific details from the knowledge base, clearly state this limitation and suggest resources for further research.
"""

FOLLOW_UP_ADDENDUM = """
## Conversation continuation
You are in a multi-turn conversation. The user is following up on a previous exchange.
- Do NOT restate the question or repeat the rigid structure above.
- Respond naturally, as a knowledgeable colleague would in an ongoing discussion.
- Build on what was already discussed — don't repeat information already given.
- If the user asks for clarification, just clarify directly.
- If the user asks a new but related question, answer it concisely without preamble.
- Keep using markdown formatting, but skip the numbered structure.
"""

COMPARISON_PROMPT = """You are a statistical research assistant.

## Response structure
Follow this order strictly:
1. **First, restate the user's core question or problem in one sentence** to confirm you understand what they need.
2. **Give your recommendation upfront**: which approach best fits their situation and why. Do NOT make the user read through all the comparisons before getting to the answer.
3. **Then provide supporting comparison** of the matched knowledge, focused on the dimensions that matter for the user's specific situation - not a generic feature dump.
4. If the user hasn't provided enough context, state what you'd need to know, but still give a tentative recommendation based on what you do know.

## Important
- Lead with the answer, not the knowledge descriptions.
- Compare only on dimensions relevant to the user's question.
- Use clear markdown formatting.
- If the knowledge base does not cover a method being compared, clearly state this limitation and avoid speculative comparisons.
- Respond in the same language the user uses. Only use Traditional Chinese (繁體中文) or English. If the user writes in any other language, respond in English.
"""

LLM_ONLY_PROMPT = """\
You are a statistical research assistant.

## Response structure
Follow this order strictly:
1. **First, restate the user's core question or problem in one sentence** to confirm \
you understand what they are asking.
2. **Give a direct answer** grounded in established statistical theory and practice.
3. If the question is about choosing a method, ask clarifying questions about \
their data and goals, but still provide a tentative recommendation.
4. Be honest about uncertainty - if the answer depends on context you don't have, say so.

## Important
- This answer is based on general statistical knowledge, as no matching knowledge was found \
in the project's curated knowledge library. Briefly note this at the end.
- Use clear markdown formatting.
- Respond in the same language the user uses. Only use Traditional Chinese (繁體中文) or English. If the user writes in any other language, respond in English.
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

SCORE_FLOOR_FILTERED = 0.40  # lower threshold when method pre-filtering already narrowed scope


def _select_strategy(
    scored: list[tuple[float, dict]],
    pre_filtered: bool = False,
) -> tuple[str, list[dict], str]:
    """Determine response strategy based on similarity scores.

    Returns (strategy_name, selected_units, system_prompt).
    When pre_filtered=True, uses a lower score floor since the search pool
    is already scoped to relevant methods.
    """
    floor = SCORE_FLOOR_FILTERED if pre_filtered else SCORE_FLOOR
    high = [(s, m) for s, m in scored if s >= floor]

    if not high:
        return "llm_only", [], LLM_ONLY_PROMPT

    if len(high) == 1 and high[0][0] >= SCORE_DIRECT:
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
) -> set[int]:
    """Locate the most relevant taxonomy branches and return linked knowledge unit IDs.

    Each taxonomy node dict has:
      - id: int
      - embedding: list[float]
      - children_ids: list[int]  (direct child node IDs)
      - knowledge_unit_ids: list[int]  (KU IDs linked to this node)

    Returns the union of knowledge_unit_ids from the top-K matched nodes
    and all their descendants (resolved via children_ids).
    """
    if not query_embedding or not taxonomy_nodes:
        return set()

    # Build a lookup for fast traversal
    node_by_id: dict[int, dict] = {n["id"]: n for n in taxonomy_nodes}

    # Score each node by cosine similarity
    scored: list[tuple[float, dict]] = []
    for node in taxonomy_nodes:
        emb = node.get("embedding")
        if not emb:
            continue
        sim = cosine_similarity(query_embedding, emb)
        scored.append((sim, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_nodes = scored[:top_k]

    # Collect KU IDs from matched nodes and all descendants
    boost_ids: set[int] = set()

    def _collect_descendant_kus(node_id: int) -> None:
        """Recursively collect knowledge_unit_ids from a node and its descendants."""
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

    Uses the same matching logic as the old _filter_units_by_methods but queries
    the DB directly and returns IDs instead of filtering a list.
    """
    if not selected_methods:
        return set()

    from models import KnowledgeUnit

    # Build allowed names and fields
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

    # Query all KU ids with method_name or field
    kus = db.query(KnowledgeUnit.id, KnowledgeUnit.method_name, KnowledgeUnit.field).all()
    boost_ids: set[int] = set()
    for ku_id, method_name, field in kus:
        name = (method_name or "").lower()
        f = (field or "").lower().strip()
        if any(a in name or name in a for a in allowed_names if a):
            boost_ids.add(ku_id)
        elif f and f in allowed_fields:
            boost_ids.add(ku_id)

    return boost_ids


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

Given the conversation history and the user's latest message, rewrite the message into a complete, self-contained question.

Rules:
- Resolve all pronouns and references (e.g., "that method" -> the actual method name from history)
- Keep the rewritten question concise and in the same language as the user's message
- If the message is already self-contained, return it unchanged
- Return ONLY the rewritten question, no explanation"""


def _rewrite_query(
    message: str,
    history: list[dict[str, str]],
    api_key: str,
) -> str:
    """Rewrite a follow-up question into a self-contained query using conversation history."""
    if not history:
        return message

    client = OpenAI(api_key=api_key)
    msgs: list[dict[str, str]] = [{"role": "system", "content": REWRITE_PROMPT}]
    # Include last 6 messages of history for context
    for h in history[-6:]:
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
) -> tuple[str, str, list[dict]]:
    """Classify question -> retrieve knowledge -> select strategy -> generate response."""
    result = _prepare_generation_context(
        message, api_key, db=db, history=history,
        method_context=method_context, method_skills=method_skills,
        taxonomy_nodes=taxonomy_nodes,
    )

    # Clarify mode
    if result[1] == "clarify":
        _, _, clarify_text, debug_lines, matched_units = result
        return clarify_text, chr(10).join(debug_lines), matched_units

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
            return answer, chr(10).join(debug_lines), matched_units
        except Exception as e:
            logger.warning(f"[Chat] Dify call failed: {e}, falling back to OpenAI")
            debug_lines.append(f"LLM backend: **Dify FAILED ({e}), fell back to OpenAI**")

    # Fallback: direct OpenAI
    gen_params = {
        "direct_answer": {"temperature": 0.3, "max_tokens": 600},
        "comparison":    {"temperature": 0.5, "max_tokens": 800},
        "llm_only":      {"temperature": 0.7, "max_tokens": 400},
    }[strategy]
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, **gen_params)
    answer = resp.choices[0].message.content or "No response generated."
    debug_lines.append("LLM backend: **Direct OpenAI**")

    logger.info("[Chat] Done.")
    return answer, chr(10).join(debug_lines), matched_units


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
    effective_message = message
    if history:
        effective_message = _rewrite_query(message, history, api_key)
        if effective_message != message:
            logger.info(f"[Chat] Step 0: Rewrote query: '{message}' -> '{effective_message}'")

    logger.info("[Chat] Step 1: Classifying question...")
    route = classify_question(effective_message, _skills, api_key, history)

    queries = getattr(route, "search_queries", []) or [route.search_query]
    queries = [q for q in queries if q.strip()]

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
            boost_ids = _taxonomy_locate(first_query_emb, taxonomy_nodes)
            debug_lines.append(f"Taxonomy boost: {len(boost_ids)} units from matched branches")
        else:
            debug_lines.append("Taxonomy boost: skipped (embedding failed)")

    # Step 2: Retrieval
    logger.info("[Chat] Step 2: Retrieving knowledge units...")
    use_hybrid = db is not None and vector_queries
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

        strategy, selected_units, system_prompt = _select_strategy(
            scored, pre_filtered=bool(selected_methods)
        )
        knowledge_section = _build_knowledge_context(selected_units)
        matched_units = selected_units

    # Sibling method recommendations
    if matched_units and taxonomy_nodes:
        siblings = _get_sibling_methods(matched_units, taxonomy_nodes)
        if siblings:
            sibling_names = [f"{s['name']} ({s['node_type']})" for s in siblings]
            debug_lines.append(f"Related methods (taxonomy siblings): {', '.join(sibling_names)}")

    debug_lines.insert(1, f"Strategy: **{strategy}**")

    full_system = system_prompt
    if history:
        full_system += FOLLOW_UP_ADDENDUM
    if knowledge_section:
        full_system += CITATION_INSTRUCTION
        full_system += chr(10)*2 + "## Knowledge Base - Matched Units" + chr(10)*2 + knowledge_section

    msgs: list[dict[str, str]] = [{"role": "system", "content": full_system}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": message})

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
        yield ("done", clarify_text)
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
            yield ("done", answer)
            return
        except Exception as e:
            logger.warning(f"[Chat] Dify call failed: {e}, falling back to OpenAI streaming")
            debug_lines.append(f"LLM backend: **Dify FAILED ({e}), fell back to OpenAI streaming**")

    # OpenAI streaming
    gen_params = {
        "direct_answer": {"temperature": 0.3, "max_tokens": 600},
        "comparison":    {"temperature": 0.5, "max_tokens": 800},
        "llm_only":      {"temperature": 0.7, "max_tokens": 400},
    }[strategy]

    client = OpenAI(api_key=api_key)
    debug_lines.append("LLM backend: **OpenAI Streaming**")

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini", messages=msgs, stream=True, **gen_params
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
        yield ("done", answer)
    except Exception as e:
        logger.error(f"[Chat] OpenAI streaming failed: {e}")
        yield ("error", str(e))
