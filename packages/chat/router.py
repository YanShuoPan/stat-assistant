"""Question router \u2014 classifies user questions, expands queries for retrieval."""

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)

from .skill_loader import Skill

ROUTER_PROMPT = """You are a question classifier and query expander for a statistical research assistant.

Given a user question, do THREE things:

1. **Classify** it into exactly ONE skill:
{skill_list}

2. **Identify the statistical problem domain.** Map vague or colloquial descriptions to precise statistical concepts. Common mappings:
   - "many variables / too many variables / high-dimensional" -> high-dimensional variable selection, model selection, sparse regression
   - "variable selection / model selection / feature importance" -> variable selection, feature selection, model selection
   - "overfitting / model too complex" -> regularization, model selection, cross-validation, information criterion
   - "prediction / forecast" -> prediction risk, regression, supervised learning
   - "compare methods / which method is better" -> method comparison
   - "convergence / theoretical properties" -> convergence rate, asymptotic theory, consistency
   - "stepwise / greedy" -> stepwise regression, orthogonal greedy algorithm, forward selection
   - "penalty / regularization" -> penalized regression, LASSO, ridge, regularization
   These are examples. Apply the same logic to any vague question.

3. **Generate 1-3 search queries** for finding relevant methods in a knowledge base:
   - query_1: directly matches the user's intent in precise statistical English
   - query_2: expands to related method names and technical terms (e.g., OGA, HDIC, LASSO, stepwise)
   - query_3 (optional): broader concept query if the topic spans multiple areas
   All queries must be in English.

Return ONLY a JSON object:
{{"skill": "skill_name", "search_queries": ["query_1", "query_2", "query_3"], "confidence": 0.8}}

confidence (0.0-1.0): How clear is the user's intent?
- 1.0: Crystal clear, specific question
- 0.7-0.9: Clear enough to answer
- 0.4-0.6: Vague, could mean multiple things
- 0.0-0.3: Very vague, need clarification

Examples:
- User: "I have too many variables, what should I do?"
  {{"skill": "method_recommendation", "search_queries": ["how to handle high-dimensional data with many variables", "variable selection model selection high-dimensional sparse regression", "OGA HDIC stepwise regression penalized method"]}}
- User: "What is OGA?"
  {{"skill": "method_explanation", "search_queries": ["orthogonal greedy algorithm OGA definition and procedure", "OGA stepwise regression high-dimensional variable selection"]}}
- User: "My data has 500 variables but only 100 observations"
  {{"skill": "method_recommendation", "search_queries": ["regression with more variables than observations p greater than n", "high-dimensional sparse model selection variable selection", "OGA LASSO penalized regression small sample"]}}

If the question is general and does not need method retrieval (e.g. greetings), set search_queries to [""].
No extra text."""


@dataclass
class RouteResult:
    skill: str
    search_query: str  # primary query (first of search_queries), kept for backward compat
    search_queries: list[str] = field(default_factory=list)
    confidence: float = 1.0


def classify_question(
    message: str,
    skills: dict[str, Skill],
    api_key: str,
    history: list[dict[str, str]] | None = None,
) -> RouteResult:
    """Classify a user message and generate expanded search queries.

    Returns a RouteResult with skill name and 1-3 search queries.
    """
    skill_list = "\n".join(
        f'- "{s.name}": {s.description}' for s in skills.values()
    )
    system = ROUTER_PROMPT.format(skill_list=skill_list)

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if history:
        for h in history[-4:]:
            messages.append(h)
    messages.append({"role": "user", "content": message})

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=250,
        )
    except Exception as e:
        logger.warning(f"[Router] OpenAI call failed: {e}, falling back to general_stats")
        return RouteResult(skill="general_stats", search_query="", search_queries=[""])

    raw = (resp.choices[0].message.content or "").strip()

    try:
        result = json.loads(raw)
        skill_name = result.get("skill", "general_stats")
        if skill_name not in skills:
            skill_name = "general_stats"

        # Support both old "search_query" and new "search_queries" format
        queries = result.get("search_queries", [])
        if not queries:
            legacy = result.get("search_query", "")
            queries = [legacy] if legacy else [""]

        # Filter out empty strings (except if all are empty)
        queries = [q for q in queries if q.strip()] or [""]

        confidence = float(result.get("confidence", 1.0))

        return RouteResult(
            skill=skill_name,
            search_query=queries[0],
            search_queries=queries,
            confidence=confidence,
        )
    except (json.JSONDecodeError, AttributeError):
        return RouteResult(skill="general_stats", search_query="", search_queries=[""])
