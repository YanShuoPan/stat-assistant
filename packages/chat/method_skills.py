"""Generate and manage per-method summary skills from knowledge units."""

import json
from collections import defaultdict

from openai import OpenAI

SUMMARIZE_PROMPT = """You are a research method cataloging assistant.

Given a set of knowledge units about the SAME statistical method, produce a single
concise method skill card.

Return ONLY a JSON object with these fields:
{
  "method": "the COMPLETE pipeline name as proposed by the authors (e.g. OGA+HDIC+Trim, not just OGA)",
  "field": "statistical sub-field (e.g. high-dimensional regression, causal inference)",
  "aliases": ["component names and variant names (e.g. OGA, HDIC, DML1, DML2)"],
  "purpose": "one sentence: what problem does the COMPLETE pipeline solve?",
  "summary": "2-4 sentence technical summary of the full pipeline: what each stage does and how they connect",
  "pipeline_steps": ["Step 1 name: what it does", "Step 2 name: what it does", "..."],
  "assumptions": ["key assumptions the method requires"],
  "typical_questions": ["3-5 realistic questions a researcher would ask about this method"],
  "related_methods": ["alternative or complementary methods"]
}

Rules:
- "method" must be the FULL pipeline name the authors proposed, not just the most frequent component.
  If the paper proposes a multi-step procedure (e.g. selection + criterion + trimming), the name
  should reflect ALL stages (e.g. "OGA+HDIC+Trim").
- "pipeline_steps" is REQUIRED. Describe each stage of the authors' complete algorithm in order.
  Even single-step methods should have at least one step.
- "aliases" should include individual component names and all variant names seen in the knowledge units.
- "summary" should describe the full end-to-end workflow, not just one component.
- "typical_questions" should cover: what is it, when to use it, how it works, limitations.
- Be precise and technical. No filler.

Return ONLY valid JSON, no markdown fences, no extra text."""


def _group_units_by_method(units: list[dict]) -> dict[str, list[dict]]:
    """Group knowledge units by their canonical method.

    Uses method_name as the grouping key. Units without method_name
    are grouped under their title.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for u in units:
        key = (u.get("method_name") or u.get("title") or "unknown").strip()
        groups[key].append(u)
    return dict(groups)


def _merge_related_groups(groups: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Merge groups that are clearly variants of the same method.

    Merging rules:
    - A+B, A+B+C patterns: shorter component merges into longest combined name
    - "noiseless A", "generalized A" merge into A
    - Exact abbreviation expansions (e.g. DML <-> Double/debiased machine learning)
    - Simple prefix match: if name B starts with name A followed by a
      non-alpha char or end-of-string, B merges into A.
    """
    import re

    # Known abbreviation expansions (add more as needed)
    ABBREV_MAP = {
        "dml": {"double/debiased machine learning", "double machine learning",
                "debiased machine learning"},
        "bo-mocu": {"bayesian optimization with mocu"},
        "mocu": {"mean objective cost of uncertainty (mocu)",
                 "mean objective cost of uncertainty"},
    }

    names = sorted(groups.keys(), key=len)  # shorter names first
    merged: dict[str, list[dict]] = {}
    absorbed: set[str] = set()

    def _is_variant(canonical: str, other: str) -> bool:
        c = canonical.lower()
        o = other.lower()
        # 1. Known abbreviation match
        if c in ABBREV_MAP and o in ABBREV_MAP[c]:
            return True
        # 2. Plus-suffix pattern: "A+B" merges into "A"
        if o.startswith(c + "+"):
            return True
        # 3. Modifier prefix: "noiseless A", "generalized A"
        modifiers = ("noiseless ", "generalized ", "modified ")
        for mod in modifiers:
            if o.startswith(mod) and o[len(mod):].strip() == c:
                return True
        # 4. Version suffix: "A1", "A2", "A IV"
        if o.startswith(c) and len(o) > len(c):
            suffix = o[len(c):].strip()
            if suffix in ("1", "2", "iv", "v", "1a", "2a"):
                return True
        return False

    # First pass: find clusters of related names
    clusters: list[list[str]] = []
    clustered: set[str] = set()
    for name in names:
        if name in clustered:
            continue
        cluster = [name]
        clustered.add(name)
        for other in names:
            if other in clustered:
                continue
            # Check both directions for variant relationship
            if _is_variant(name, other) or _is_variant(other, name):
                cluster.append(other)
                clustered.add(other)
        clusters.append(cluster)

    # For each cluster, pick the LONGEST name as canonical (= most complete pipeline)
    for cluster in clusters:
        canonical = max(cluster, key=len)
        merged[canonical] = []
        for name in cluster:
            merged[canonical].extend(groups[name])
            if name != canonical:
                absorbed.add(name)

    return merged


def _build_unit_summary_text(units: list[dict], max_units: int = 30) -> str:
    """Build a text block summarizing a group of knowledge units for the LLM."""
    lines = []
    for u in units[:max_units]:
        method = u.get("method_name") or u.get("title", "")
        ktype = u.get("knowledge_type", "")
        content = u.get("content", "")
        field = u.get("field", "")
        problem = u.get("problem_it_solves", "")
        assumptions = u.get("model_assumption", "")
        parts = [f"[{method}] [{ktype}]"]
        if field:
            parts.append(f"Field: {field}")
        if problem:
            parts.append(f"Problem: {problem}")
        parts.append(content[:500])
        if assumptions:
            parts.append(f"Assumptions: {assumptions}")
        lines.append(" | ".join(parts))
    return "\n---\n".join(lines)


def generate_method_skill(units: list[dict], api_key: str) -> dict | None:
    """Generate a method skill card from a group of KUs about the same method."""
    if not units:
        return None

    summary_text = _build_unit_summary_text(units)
    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUMMARIZE_PROMPT},
            {"role": "user", "content": summary_text},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    raw = (resp.choices[0].message.content or "").strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _identify_primary_methods(
    groups: dict[str, list[dict]],
    primary_threshold: int = 10,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Identify primary methods and absorb sub-concept groups into them.

    Primary methods are those with >= primary_threshold KUs (i.e., the paper's
    main contribution). Smaller groups are sub-concepts that get absorbed into
    the primary method sharing the same field.

    Returns (primary_groups, absorbed_names).
    """
    # Separate primary vs sub-concept groups
    primary: dict[str, list[dict]] = {}
    subconcepts: dict[str, list[dict]] = {}
    for name, units in groups.items():
        if len(units) >= primary_threshold:
            primary[name] = list(units)
        else:
            subconcepts[name] = units

    # Build field -> primary method mapping
    field_to_primary: dict[str, str] = {}
    for name, units in primary.items():
        fields = defaultdict(int)
        for u in units:
            f = (u.get("field") or "").lower().strip()
            if f:
                fields[f] += 1
        if fields:
            dominant_field = max(fields, key=fields.get)
            field_to_primary[dominant_field] = name

    # Absorb sub-concepts into primary methods by field match
    absorbed: list[str] = []
    for name, units in subconcepts.items():
        fields = defaultdict(int)
        for u in units:
            f = (u.get("field") or "").lower().strip()
            if f:
                fields[f] += 1
        if not fields:
            continue
        dominant_field = max(fields, key=fields.get)
        if dominant_field in field_to_primary:
            target = field_to_primary[dominant_field]
            primary[target].extend(units)
            absorbed.append(name)

    return primary, absorbed


def generate_all_method_skills(
    units: list[dict], api_key: str, primary_threshold: int = 10,
) -> list[dict]:
    """Generate method skill cards for primary methods only.

    Each primary method represents a complete pipeline proposed by the paper's
    authors. Sub-concepts (Neyman Orthogonality, Cross-Fitting, etc.) are
    absorbed into their parent method based on field overlap.
    """
    groups = _group_units_by_method(units)
    groups = _merge_related_groups(groups)

    primary_groups, absorbed = _identify_primary_methods(groups, primary_threshold)

    skills = []
    for canonical_name, group_units in primary_groups.items():
        skill = generate_method_skill(group_units, api_key)
        if skill:
            skills.append(skill)
    return skills


def format_skills_for_selection(skills: list[dict]) -> str:
    """Format method skills into a compact text for LLM method selection."""
    lines = []
    for i, s in enumerate(skills, 1):
        aliases = ", ".join(s.get("aliases", []))
        alias_str = f" (aka {aliases})" if aliases else ""
        lines.append(
            f"{i}. **{s.get('method', '?')}**{alias_str} "
            f"[{s.get('field', '?')}]: {s.get('purpose', '?')}"
        )
    return "\n".join(lines)
