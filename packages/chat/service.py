"""Chat service - routes questions to skills, retrieves relevant knowledge units, generates responses.

Response strategy:
  - direct_answer: One unit matched with high confidence -> LLM answers grounded in that unit
  - comparison:    Multiple units matched -> LLM synthesizes across units
  - llm_only:     No good match in knowledge base -> LLM answers from its own knowledge
"""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

from .skill_loader import load_skills
from .router import classify_question
from .embeddings import compute_embedding, _score_methods
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_knowledge_context(units: list[dict]) -> tuple[str, str]:
    """Build a formatted knowledge base string from knowledge unit dicts."""
    if not units:
        return ""

    sections = []
    for u in units:
        method = u.get("method_name") or u.get("title", "Unknown")
        ktype = u.get("knowledge_type", "")
        parts = [f"### {method} [{ktype}]"]
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
# Main entry point
# ---------------------------------------------------------------------------

def _multi_query_search(
    queries: list[str],
    method_context: list[dict],
    api_key: str,
) -> list[tuple[float, dict]]:
    """Run embedding search for each query, merge results keeping best score per unit."""
    best: dict[int, tuple[float, dict]] = {}  # unit id -> (best_score, unit)

    for q in queries:
        if not q.strip():
            continue
        emb = compute_embedding(q, api_key)
        scored = _score_methods(emb, method_context)
        for score, unit in scored:
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

    import json
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
    import json
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
    import json as _json

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
    history: list[dict[str, str]] | None = None,
    method_context: list[dict] | None = None,
    method_skills: list[dict] | None = None,
    dify_api_key: str | None = None,
    dify_base_url: str = "https://api.dify.ai/v1",
) -> str:
    """Classify question -> retrieve knowledge -> select strategy -> generate response."""
    # Step 0: Rewrite query for multi-turn context
    effective_message = message
    if history:
        effective_message = _rewrite_query(message, history, api_key)
        if effective_message != message:
            logger.info(f"[Chat] Step 0: Rewrote query: '{message}' -> '{effective_message}'")

    # Step 1: Classify + expand queries
    logger.info("[Chat] Step 1: Classifying question...")
    route = classify_question(effective_message, _skills, api_key, history)
    skill = _skills[route.skill]

    queries = getattr(route, "search_queries", []) or [route.search_query]
    queries = [q for q in queries if q.strip()]

    debug_lines = [
        f"Skill: **{route.skill}**",
        f"Search queries: *{queries or '(none)'}*",
        f"Knowledge units in DB: {len(method_context) if method_context else 0}",
        f"Confidence: **{route.confidence}**",
    ]

    if effective_message != message:
        debug_lines.append(f"Rewritten query: *{effective_message}*")

    # Step 1.1: Check if clarification is needed
    CONFIDENCE_THRESHOLD = 0.5
    if route.confidence < CONFIDENCE_THRESHOLD and not history:
        # First message and vague intent -> ask for clarification
        logger.info(f"[Chat] Low confidence ({route.confidence}), generating clarification...")
        client = OpenAI(api_key=api_key)
        clarify_msgs: list[dict[str, str]] = [{"role": "system", "content": CLARIFY_PROMPT}]
        clarify_msgs.append({"role": "user", "content": message})
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=clarify_msgs, temperature=0.5, max_tokens=200
        )
        clarify_text = resp.choices[0].message.content or "Could you provide more details?"
        debug_lines.append(f"Confidence: **{route.confidence}** (below {CONFIDENCE_THRESHOLD}, clarifying)")
        debug_text = chr(10).join(debug_lines)
        return clarify_text, debug_text

    strategy = "llm_only"
    system_prompt = LLM_ONLY_PROMPT
    knowledge_section = ""

    # Step 1.5: Method skill pre-filtering
    logger.info("[Chat] Step 1.5: Selecting methods...")
    selected_methods: list[str] = []
    search_pool = method_context or []
    if method_skills and method_context:
        selected_methods, q_analysis = _select_methods(effective_message, method_skills, api_key, history)
        if selected_methods:
            search_pool = _filter_units_by_methods(method_context, selected_methods, method_skills)
            debug_lines.append(f"Question analysis: {q_analysis}")
            debug_lines.append(f"Selected methods: **{selected_methods}**")
            debug_lines.append(f"Filtered KUs: {len(search_pool)} / {len(method_context)}")
        else:
            debug_lines.append(f"Question analysis: {q_analysis}")
            debug_lines.append("Method selection: no specific method matched, searching all KUs")

    # Step 1.7: Inject skill-bridge queries when method was selected
    #   Converts user's colloquial language to academic terms via skill summary
    if selected_methods and method_skills:
        for ms in method_skills:
            if ms.get("method") in selected_methods:
                purpose = ms.get("purpose", "")
                method_name = ms.get("method", "")
                field = ms.get("field", "")
                if purpose:
                    queries.append(f"{method_name} {field} {purpose}")

    # Step 2: Multi-query retrieve & score
    logger.info("[Chat] Step 2: Retrieving knowledge units...")
    if search_pool and queries:
        scored = _multi_query_search(queries, search_pool, api_key)

        debug_lines.append("")
        debug_lines.append("**Knowledge unit similarity scores (best across queries):**")
        for score, u in scored:
            title = u.get("title", "unknown")
            ktype = u.get("knowledge_type", "")
            tags = u.get("topic_tags", [])
            tag_str = f" -- tags: {', '.join(tags[:5])}" if tags else ""
            debug_lines.append(f"- {title} [{ktype}]: {score}{tag_str}")

        # Step 2.5: Select strategy based on score distribution
        strategy, selected_units, system_prompt = _select_strategy(scored, pre_filtered=bool(selected_methods))
        knowledge_section = _build_knowledge_context(selected_units)

    debug_lines.insert(1, f"Strategy: **{strategy}**")

    # Step 3+4: Generate response via Dify workflow or direct OpenAI
    import json as _json
    history_text = _json.dumps(history or [], ensure_ascii=False)

    if dify_api_key:
        try:
            answer = _call_dify_workflow(
                question=message,
                knowledge_context=knowledge_section,
                history=history_text,
                strategy=strategy,
                dify_api_key=dify_api_key,
                dify_base_url=dify_base_url,
            )
            debug_lines.append("LLM backend: **Dify Workflow**")
        except Exception as e:
            logger.warning(f"[Chat] Dify call failed: {e}, falling back to OpenAI")
            dify_api_key = None  # trigger OpenAI fallback below
            debug_lines.append(f"LLM backend: **Dify FAILED ({e}), fell back to OpenAI**")
    if not dify_api_key:
        # Fallback: direct OpenAI call (original behavior)
        full_system = system_prompt
        if history:
            full_system += FOLLOW_UP_ADDENDUM
        if knowledge_section:
            full_system += chr(10)*2 + "## Knowledge Base - Matched Units" + chr(10)*2 + knowledge_section
        gen_params = {
            "direct_answer": {"temperature": 0.3, "max_tokens": 600},
            "comparison":    {"temperature": 0.5, "max_tokens": 800},
            "llm_only":      {"temperature": 0.7, "max_tokens": 400},
        }[strategy]
        client = OpenAI(api_key=api_key)
        msgs: list[dict[str, str]] = [{"role": "system", "content": full_system}]
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": message})
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, **gen_params)
        answer = resp.choices[0].message.content or "No response generated."
        debug_lines.append("LLM backend: **Direct OpenAI**")

    logger.info("[Chat] Done.")
    debug_text = chr(10).join(debug_lines)
    return answer, debug_text


def _prepare_generation_context(
    message: str,
    api_key: str,
    history: list[dict[str, str]] | None = None,
    method_context: list[dict] | None = None,
    method_skills: list[dict] | None = None,
) -> tuple:
    """Run the classification/retrieval pipeline, return everything needed for generation.

    Returns either:
    - (None, "clarify", clarify_text, debug_lines) for clarification mode
    - (msgs, strategy, knowledge_section, debug_lines) for normal generation
    """
    effective_message = message
    if history:
        effective_message = _rewrite_query(message, history, api_key)
        if effective_message != message:
            logger.info(f"[Chat] Step 0: Rewrote query: '{message}' -> '{effective_message}'")

    logger.info("[Chat] Step 1: Classifying question...")
    route = classify_question(effective_message, _skills, api_key, history)
    skill = _skills[route.skill]

    queries = getattr(route, "search_queries", []) or [route.search_query]
    queries = [q for q in queries if q.strip()]

    debug_lines = [
        f"Skill: **{route.skill}**",
        f"Search queries: *{queries or '(none)'}*",
        f"Knowledge units in DB: {len(method_context) if method_context else 0}",
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
        return None, "clarify", clarify_text, debug_lines

    strategy = "llm_only"
    system_prompt = LLM_ONLY_PROMPT
    knowledge_section = ""

    logger.info("[Chat] Step 1.5: Selecting methods...")
    selected_methods: list[str] = []
    search_pool = method_context or []
    if method_skills and method_context:
        selected_methods, q_analysis = _select_methods(effective_message, method_skills, api_key, history)
        if selected_methods:
            search_pool = _filter_units_by_methods(method_context, selected_methods, method_skills)
            debug_lines.append(f"Question analysis: {q_analysis}")
            debug_lines.append(f"Selected methods: **{selected_methods}**")
            debug_lines.append(f"Filtered KUs: {len(search_pool)} / {len(method_context)}")
        else:
            debug_lines.append(f"Question analysis: {q_analysis}")
            debug_lines.append("Method selection: no specific method matched, searching all KUs")

    if selected_methods and method_skills:
        for ms in method_skills:
            if ms.get("method") in selected_methods:
                purpose = ms.get("purpose", "")
                method_name = ms.get("method", "")
                field = ms.get("field", "")
                if purpose:
                    queries.append(f"{method_name} {field} {purpose}")

    logger.info("[Chat] Step 2: Retrieving knowledge units...")
    if search_pool and queries:
        scored = _multi_query_search(queries, search_pool, api_key)
        debug_lines.append("")
        debug_lines.append("**Knowledge unit similarity scores (best across queries):**")
        for score, u in scored:
            title = u.get("title", "unknown")
            ktype = u.get("knowledge_type", "")
            tags = u.get("topic_tags", [])
            tag_str = f" -- tags: {', '.join(tags[:5])}" if tags else ""
            debug_lines.append(f"- {title} [{ktype}]: {score}{tag_str}")
        strategy, selected_units, system_prompt = _select_strategy(scored, pre_filtered=bool(selected_methods))
        knowledge_section = _build_knowledge_context(selected_units)

    debug_lines.insert(1, f"Strategy: **{strategy}**")

    full_system = system_prompt
    if history:
        full_system += FOLLOW_UP_ADDENDUM
    if knowledge_section:
        full_system += chr(10)*2 + "## Knowledge Base - Matched Units" + chr(10)*2 + knowledge_section

    msgs: list[dict[str, str]] = [{"role": "system", "content": full_system}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": message})

    return msgs, strategy, knowledge_section, debug_lines


def generate_response_stream(
    message: str,
    api_key: str,
    history: list[dict[str, str]] | None = None,
    method_context: list[dict] | None = None,
    method_skills: list[dict] | None = None,
    dify_api_key: str | None = None,
    dify_base_url: str = "https://api.dify.ai/v1",
):
    """Streaming version of generate_response. Yields (event_type, data) tuples.

    Event types: "token", "debug", "done", "error"
    """
    import json as _json

    result = _prepare_generation_context(
        message, api_key, history, method_context, method_skills,
    )

    # Clarify mode - no streaming needed
    if result[1] == "clarify":
        _, _, clarify_text, debug_lines = result
        yield ("token", clarify_text)
        yield ("debug", chr(10).join(debug_lines))
        yield ("done", clarify_text)
        return

    msgs, strategy, knowledge_section, debug_lines = result

    # Try Dify first (blocking, then emit as single chunk)
    if dify_api_key:
        try:
            history_text = _json.dumps(history or [], ensure_ascii=False)
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
        yield ("done", answer)
    except Exception as e:
        logger.error(f"[Chat] OpenAI streaming failed: {e}")
        yield ("error", str(e))
