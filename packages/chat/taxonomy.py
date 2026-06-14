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
You are a statistical / machine-learning methods taxonomy classifier.

## Your task
Given a list of method names (each with a short context snippet), classify each
into a three-level taxonomy.

## Taxonomy levels
1. **Problem Category** — the fundamental problem the method addresses.
   Use categories that reflect the method's OWN purpose, NOT the paper it
   appeared in.  Examples: "Variable Selection", "Causal Inference",
   "Spatial Statistics", "Experimental Design", "Missing Data",
   "Dimensionality Reduction", "Density Estimation", "Time Series",
   "Hypothesis Testing", "Regression", "Classification", "Clustering",
   "Optimization", "Uncertainty Quantification".
2. **Method Family** — the methodological approach within that category.
   Examples: "Greedy Methods", "Penalization Methods", "Semiparametric Methods",
   "Kernel Methods", "Deep Learning Methods", "Bayesian Optimization".
3. **Method / Variant** — the specific method or variant (e.g. "OGA",
   "OGA+HDIC+Trim").

## Critical rules

### Deduplication — MOST IMPORTANT
Before classifying, first identify groups of names that refer to the SAME
method.  Common patterns:
- Abbreviation vs full name: "DML" = "Double/Debiased Machine Learning"
- With/without slash or spaces: "LASSO" = "Lasso"
- Numbered variants: "DML1" is a variant OF "DML", not a separate method

When you find synonyms, pick ONE canonical name and list the others as aliases.
Output only ONE entry per unique method (with all input names as aliases or the
canonical name).  The "method_name" field should be set to the canonical name;
the "input_names" field should list ALL original input names that map to this entry.

### Classification accuracy
- Classify based on what the METHOD fundamentally does, not which paper it
  appeared in.  A spatial prediction method is "Spatial Statistics" even if
  it appeared in a causal inference paper.
- If a name describes a theoretical concept/property rather than an algorithm
  (e.g. "Conditional Convergence", "Asymptotic Normality"), set
  "is_concept": true.  These will be attached to their parent method rather
  than becoming standalone taxonomy nodes.

### Descriptions
- Descriptions MUST be factually accurate.  Expand acronyms correctly.
- Do not guess — if unsure, say "A statistical method for [general area]".

### Variants
- A method_name like "OGA+HDIC" is likely a variant of "OGA" — set parent_method
- For combined/pipeline methods (X+Y+Z), the longest combination is the variant,
  shorter parts may be parent_method

## Output format
Return a JSON array (ONLY valid JSON, no markdown fences):
[
  {
    "method_name": "canonical method name",
    "input_names": ["all input names that map to this method"],
    "problem_category": "category name",
    "method_family": "family name",
    "parent_method": null or "parent method name for variants",
    "node_type": "method" or "variant",
    "aliases": ["alternative names / abbreviations"],
    "description": "one accurate sentence describing what this method does",
    "is_concept": false,
    "is_new_category": true/false,
    "is_new_family": true/false,
    "is_new_method": true/false
  }
]

IMPORTANT: EVERY input method name must appear in exactly one entry's
"input_names" array.  Do not drop any input method."""


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
    # 1. Extract distinct method names and context snippets from the units
    method_names = _extract_method_names(units)
    if not method_names:
        logger.info("No method names found in units — nothing to classify")
        return {"classified": 0, "new_nodes": 0}

    method_contexts = _extract_method_contexts(units)

    logger.info("Classifying %d distinct method(s) into taxonomy", len(method_names))

    # 2. Classify and process in batches of BATCH_SIZE
    #    Each batch: LLM classify → create nodes → link units → flush
    #    Next batch sees the updated tree from previous batches.
    BATCH_SIZE = 15
    all_classifications: list[dict] = []
    counts = {"classified": 0, "new_nodes": 0}

    total_batches = (len(method_names) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(method_names), BATCH_SIZE):
        batch = method_names[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info("Classifying batch %d/%d (%d methods)", batch_num, total_batches, len(batch))

        current_tree = _get_taxonomy_tree_json(db)
        batch_result = _classify_methods_llm(batch, current_tree, api_key, method_contexts)
        if batch_result:
            _process_classification_entries(batch_result, db, units, api_key, counts)
            all_classifications.extend(batch_result)
            db.flush()
        else:
            logger.warning("Batch %d returned no classifications", batch_num)

    if not all_classifications:
        logger.warning("LLM returned no classifications across all batches")
        return {"classified": 0, "new_nodes": 0}

    # 3. Re-query any methods missed across all batches
    covered = set()
    for entry in all_classifications:
        for n in entry.get("input_names", [entry.get("method_name", "")]):
            covered.add(n.strip().lower())
        for a in entry.get("aliases", []):
            covered.add(a.strip().lower())
        covered.add(entry.get("method_name", "").strip().lower())
    missing = [m for m in method_names if m.lower() not in covered]
    if missing:
        logger.info("Re-querying %d methods missed in all batches: %s", len(missing), missing)
        tree_json_updated = _get_taxonomy_tree_json(db)
        extra = _classify_methods_llm(missing, tree_json_updated, api_key, method_contexts)
        if extra:
            _process_classification_entries(extra, db, units, api_key, counts)

    db.commit()
    logger.info(
        "Taxonomy classification complete: %d units classified, %d new nodes",
        counts["classified"],
        counts["new_nodes"],
    )
    return counts


def _process_classification_entries(
    entries: list[dict],
    db: Session,
    units: list,
    api_key: str,
    counts: dict,
) -> None:
    """Process a list of LLM classification entries: create nodes and link units."""
    for entry in entries:
        method_name = entry.get("method_name", "").strip()
        if not method_name:
            continue

        # Skip concepts — they are not standalone methods
        if entry.get("is_concept"):
            logger.info("Skipping concept (not a method): '%s'", method_name)
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
                counts["new_nodes"] += 1

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
                counts["new_nodes"] += 1

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
                    counts["new_nodes"] += 1
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
                counts["new_nodes"] += 1

            # Link matching units to this node
            link_names = set()
            for n in entry.get("input_names", [method_name]):
                link_names.add(n.strip())
            link_names.add(method_name)
            for a in entry.get("aliases", []):
                link_names.add(a.strip())
            for iname in link_names:
                linked = _link_units_to_node(db, units, iname, method_node.id)
                counts["classified"] += linked

        except Exception:
            logger.exception("Failed to classify method '%s'", method_name)
            continue


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


def _extract_method_contexts(units: list) -> dict[str, str]:
    """Extract a short context snippet per distinct method_name.

    Returns {method_name: context_snippet} where the snippet is a
    concatenation of title + content (truncated) from the first matching unit.
    """
    contexts: dict[str, str] = {}
    for u in units:
        name = getattr(u, "method_name", None)
        if not name or not name.strip():
            continue
        name = name.strip()
        if name in contexts:
            continue
        title = getattr(u, "title", "") or ""
        content = getattr(u, "content", "") or ""
        field = getattr(u, "field", "") or ""
        problem = getattr(u, "problem_it_solves", "") or ""
        snippet_parts = []
        if title:
            snippet_parts.append(f"Title: {title}")
        if field:
            snippet_parts.append(f"Field: {field}")
        if problem:
            snippet_parts.append(f"Problem: {problem[:200]}")
        if content:
            snippet_parts.append(f"Content: {content[:300]}")
        contexts[name] = "\n".join(snippet_parts)
    return contexts


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
    method_contexts: dict[str, str] | None = None,
) -> list[dict]:
    """Call GPT-4o-mini to classify method names into the taxonomy.

    Returns a list of classification dicts, one per method.
    """
    tree_text = json.dumps(tree_json, indent=2) if tree_json else "(empty — no existing taxonomy)"

    # Build methods section with context snippets
    methods_section = "## Methods to classify\n"
    for name in method_names:
        methods_section += f"\n### {name}\n"
        ctx = (method_contexts or {}).get(name)
        if ctx:
            methods_section += ctx + "\n"
        else:
            methods_section += "(no context available)\n"

    user_message = (
        "## Existing Taxonomy\n"
        f"{tree_text}\n\n"
        f"{methods_section}"
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
            max_tokens=4000,
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
