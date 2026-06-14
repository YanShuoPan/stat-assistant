"""Auto-classification of KnowledgeUnits into a method taxonomy tree.

The taxonomy has three levels:
  1. Problem Category  (e.g. "Variable Selection", "Causal Inference")
  2. Method Family     (e.g. "Greedy Methods", "Penalization Methods")
  3. Method / Variant  (e.g. "OGA", "OGA+HDIC+Trim")

This module classifies methods extracted from KnowledgeUnits into the tree
using LLM reasoning + embedding similarity for deduplication.
"""

import json
import logging
from collections import defaultdict

from openai import OpenAI
from sqlalchemy.orm import Session

from models import KnowledgeUnit, KnowledgeUnitNode, MethodNode
from .embeddings import compute_embedding, cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for taxonomy classification
# ---------------------------------------------------------------------------

TAXONOMY_CLASSIFY_PROMPT = """\
You are a statistical methods taxonomy classifier.

Given a list of method names and an existing taxonomy tree, classify each method into the taxonomy.

The taxonomy has three levels:
1. Problem Category — the type of statistical problem (e.g., "Variable Selection", "Causal Inference", "Missing Data")
2. Method Family — the methodological approach (e.g., "Greedy Methods", "Penalization Methods", "Bayesian Methods")
3. Method / Variant — the specific method or variant (e.g., "OGA", "OGA+HDIC+Trim")

Rules:
- PREFER mapping to existing nodes whenever possible
- Only suggest new nodes for genuinely new categories, families, or methods
- Use the exact name from the existing taxonomy when mapping to existing nodes
- A method_name like "OGA+HDIC" is likely a variant of "OGA" — set parent_method accordingly
- If unsure about the problem_category or method_family, make your best inference from the method name
- For combined/pipeline methods (X+Y+Z), the longest combination is the method, shorter parts may be parent_method

Return a JSON array:
[
  {
    "method_name": "the input method name",
    "problem_category": "existing or new problem category name",
    "method_family": "existing or new method family name",
    "parent_method": null or "parent method name for variants",
    "node_type": "method" or "variant",
    "aliases": ["alternative names"],
    "description": "one sentence description",
    "is_new_category": true/false,
    "is_new_family": true/false,
    "is_new_method": true/false
  }
]

Return ONLY valid JSON, no markdown fences."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_units_to_taxonomy(
    db: Session,
    units: list,
    api_key: str,
) -> dict:
    """Classify KnowledgeUnit objects into the method taxonomy.

    Args:
        db: SQLAlchemy session.
        units: List of KnowledgeUnit ORM objects to classify.
        api_key: OpenAI API key.

    Returns:
        {"classified": int, "new_nodes": int}
    """
    # 1. Extract distinct method names from the units
    method_names = _extract_method_names(units)
    if not method_names:
        logger.info("No method names found in units — nothing to classify")
        return {"classified": 0, "new_nodes": 0}

    logger.info("Classifying %d distinct method(s) into taxonomy", len(method_names))

    # 2. Load current taxonomy tree
    tree_json = _get_taxonomy_tree_json(db)

    # 3. Ask LLM to classify each method
    classifications = _classify_methods_llm(method_names, tree_json, api_key)
    if not classifications:
        logger.warning("LLM returned no classifications")
        return {"classified": 0, "new_nodes": 0}

    # 4. Match or insert nodes, then link units
    classified_count = 0
    new_node_count = 0

    for entry in classifications:
        method_name = entry.get("method_name", "").strip()
        if not method_name:
            continue

        try:
            # Ensure problem category node exists
            category_node = _match_or_create_node(
                db,
                name=entry.get("problem_category", "Uncategorized"),
                node_type="problem_category",
                parent_id=None,
                aliases=[],
                description=None,
                api_key=api_key,
            )
            if category_node._is_new:
                new_node_count += 1

            # Ensure method family node exists
            family_node = _match_or_create_node(
                db,
                name=entry.get("method_family", "Other"),
                node_type="method_family",
                parent_id=category_node.id,
                aliases=[],
                description=None,
                api_key=api_key,
            )
            if family_node._is_new:
                new_node_count += 1

            # Determine parent for variant methods
            parent_method_name = entry.get("parent_method")
            method_parent_id = family_node.id

            if parent_method_name:
                parent_method_node = _match_or_create_node(
                    db,
                    name=parent_method_name,
                    node_type="method",
                    parent_id=family_node.id,
                    aliases=[],
                    description=None,
                    api_key=api_key,
                )
                if parent_method_node._is_new:
                    new_node_count += 1
                method_parent_id = parent_method_node.id

            # Create the method/variant node itself
            node_type = entry.get("node_type", "method")
            method_node = _match_or_create_node(
                db,
                name=method_name,
                node_type=node_type,
                parent_id=method_parent_id,
                aliases=entry.get("aliases", []),
                description=entry.get("description"),
                api_key=api_key,
            )
            if method_node._is_new:
                new_node_count += 1

            # 5. Link matching units to this node
            linked = _link_units_to_node(db, units, method_name, method_node.id)
            classified_count += linked

        except Exception:
            logger.exception("Failed to classify method '%s'", method_name)
            continue

    db.commit()
    logger.info(
        "Taxonomy classification complete: %d units classified, %d new nodes",
        classified_count,
        new_node_count,
    )
    return {"classified": classified_count, "new_nodes": new_node_count}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_method_names(units: list) -> list[str]:
    """Extract distinct, non-empty method_name values from KnowledgeUnit objects."""
    names: set[str] = set()
    for u in units:
        name = getattr(u, "method_name", None)
        if name and name.strip():
            names.add(name.strip())
    return sorted(names)


def _get_taxonomy_tree_json(db: Session) -> list[dict]:
    """Load all MethodNode rows and format as a nested JSON tree.

    Returns a list of dicts representing the tree hierarchy, suitable for
    including in the LLM prompt.
    """
    nodes = db.query(MethodNode).all()
    if not nodes:
        return []

    # Build lookup structures
    by_id: dict[int, MethodNode] = {n.id: n for n in nodes}
    children: dict[int | None, list[MethodNode]] = defaultdict(list)
    for n in nodes:
        children[n.parent_id].append(n)

    def _build(parent_id: int | None) -> list[dict]:
        result = []
        for node in children.get(parent_id, []):
            entry = {
                "name": node.name,
                "type": node.node_type,
                "aliases": node.aliases or [],
            }
            kids = _build(node.id)
            if kids:
                entry["children"] = kids
            result.append(entry)
        return result

    return _build(None)


def _classify_methods_llm(
    method_names: list[str],
    tree_json: list[dict],
    api_key: str,
) -> list[dict]:
    """Call GPT-4o-mini to classify method names into the taxonomy.

    Returns a list of classification dicts, one per method.
    """
    tree_text = json.dumps(tree_json, indent=2) if tree_json else "(empty — no existing taxonomy)"

    user_message = (
        "## Existing Taxonomy\n"
        f"{tree_text}\n\n"
        "## Methods to classify\n"
        f"{json.dumps(method_names)}"
    )

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TAXONOMY_CLASSIFY_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
    except Exception:
        logger.exception("LLM call failed for taxonomy classification")
        return []

    raw = (resp.choices[0].message.content or "").strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM taxonomy output: %s", raw[:200])
        return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        logger.warning("LLM taxonomy output is not a list")
        return []

    return parsed


def _match_or_create_node(
    db: Session,
    name: str,
    node_type: str,
    parent_id: int | None,
    aliases: list[str],
    description: str | None,
    api_key: str,
) -> MethodNode:
    """Find an existing taxonomy node or create a new one.

    Matching strategy (in order):
      1. Exact name match (case-insensitive) within the same node_type
      2. Alias match — check if ``name`` appears in any node's aliases
      3. Embedding similarity > 0.90

    If no match is found, a new node is created with auto_generated=True.

    The returned node has a transient ``_is_new`` attribute indicating
    whether it was freshly created.
    """
    name_lower = name.lower().strip()

    # --- 1. Exact name match (same node_type) ---
    existing = (
        db.query(MethodNode)
        .filter(
            MethodNode.node_type == node_type,
            MethodNode.name.ilike(name_lower),
        )
        .first()
    )
    if existing:
        _maybe_add_aliases(existing, aliases)
        existing._is_new = False  # type: ignore[attr-defined]
        return existing

    # --- 2. Alias match across all nodes of same type ---
    candidates = db.query(MethodNode).filter(MethodNode.node_type == node_type).all()
    for node in candidates:
        node_aliases = node.aliases or []
        if any(a.lower().strip() == name_lower for a in node_aliases):
            _maybe_add_aliases(node, aliases)
            node._is_new = False  # type: ignore[attr-defined]
            return node

    # --- 3. Embedding similarity ---
    embedding = _compute_node_embedding(name, description, api_key)
    if embedding:
        best_score = 0.0
        best_node: MethodNode | None = None
        for node in candidates:
            if not node.embedding:
                continue
            score = cosine_similarity(embedding, node.embedding)
            if score > best_score:
                best_score = score
                best_node = node
        if best_node and best_score > 0.90:
            logger.info(
                "Matched '%s' to existing node '%s' (similarity=%.3f)",
                name,
                best_node.name,
                best_score,
            )
            _maybe_add_aliases(best_node, [name] + aliases)
            best_node._is_new = False  # type: ignore[attr-defined]
            return best_node

    # --- 4. Create new node ---
    new_node = MethodNode(
        name=name,
        node_type=node_type,
        parent_id=parent_id,
        aliases=aliases or [],
        description=description,
        auto_generated=True,
        embedding=embedding or None,
    )
    db.add(new_node)
    db.flush()  # assign id
    logger.info("Created new %s node: '%s' (id=%d)", node_type, name, new_node.id)
    new_node._is_new = True  # type: ignore[attr-defined]
    return new_node


def _maybe_add_aliases(node: MethodNode, new_aliases: list[str]) -> None:
    """Add aliases to a node if they are not already present."""
    if not new_aliases:
        return
    current = set((a.lower().strip()) for a in (node.aliases or []))
    current.add(node.name.lower().strip())
    added = False
    updated = list(node.aliases or [])
    for alias in new_aliases:
        if alias.lower().strip() and alias.lower().strip() not in current:
            updated.append(alias)
            current.add(alias.lower().strip())
            added = True
    if added:
        node.aliases = updated


def _link_units_to_node(
    db: Session,
    units: list,
    method_name: str,
    node_id: int,
) -> int:
    """Create KnowledgeUnitNode rows linking units to a taxonomy node.

    Only links units whose ``method_name`` matches (case-insensitive).
    Skips links that already exist.

    Returns the number of new links created.
    """
    method_lower = method_name.lower().strip()
    linked = 0

    for u in units:
        u_method = getattr(u, "method_name", None)
        if not u_method or u_method.lower().strip() != method_lower:
            continue

        unit_id = u.id

        # Check if link already exists
        exists = (
            db.query(KnowledgeUnitNode)
            .filter(
                KnowledgeUnitNode.knowledge_unit_id == unit_id,
                KnowledgeUnitNode.method_node_id == node_id,
            )
            .first()
        )
        if exists:
            continue

        link = KnowledgeUnitNode(
            knowledge_unit_id=unit_id,
            method_node_id=node_id,
        )
        db.add(link)
        linked += 1

    if linked:
        db.flush()
        logger.info(
            "Linked %d unit(s) with method_name='%s' to node %d",
            linked,
            method_name,
            node_id,
        )

    return linked


def _compute_node_embedding(
    name: str,
    description: str | None,
    api_key: str,
) -> list[float]:
    """Compute embedding for a taxonomy node.

    Text is ``"name: description"`` if description is provided,
    otherwise just ``name``.
    """
    text = f"{name}: {description}" if description else name
    return compute_embedding(text, api_key)
