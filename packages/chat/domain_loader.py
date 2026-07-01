"""Load domain hint definitions from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DOMAIN_HINTS_DIR = Path(__file__).parent / "domain_hints"


@dataclass
class DomainHint:
    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    prompt_hint: str = ""
    concept_keywords: list[str] = field(default_factory=list)


def load_domain_hints() -> dict[str, DomainHint]:
    """Load all .yaml domain hint files from the domain_hints directory."""
    hints: dict[str, DomainHint] = {}
    if not DOMAIN_HINTS_DIR.exists():
        return hints
    for path in DOMAIN_HINTS_DIR.glob("*.yaml"):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        hint = DomainHint(
            name=data["name"],
            description=data.get("description", ""),
            keywords=[k.lower() for k in data.get("keywords", [])],
            prompt_hint=data.get("prompt_hint", "").strip(),
            concept_keywords=[k.lower() for k in data.get("concept_keywords", [])],
        )
        hints[hint.name] = hint
    return hints


def detect_domain(message: str, hints: dict[str, DomainHint]) -> DomainHint | None:
    """Detect the statistical domain from a message using keyword matching.

    Returns the best-matching DomainHint, or None if no domain matches.
    Uses a simple scoring: count how many keywords appear in the message.
    """
    if not hints:
        return None

    msg_lower = message.lower()
    best_hint = None
    best_score = 0

    for hint in hints.values():
        score = sum(1 for kw in hint.keywords if kw in msg_lower)
        if score > best_score:
            best_score = score
            best_hint = hint

    # Require at least 1 keyword match
    return best_hint if best_score >= 1 else None


def get_all_concept_keywords(hints: dict[str, DomainHint]) -> list[str]:
    """Return a deduplicated list of all concept keywords across domains."""
    seen: set[str] = set()
    result: list[str] = []
    for hint in hints.values():
        for kw in hint.concept_keywords:
            if kw not in seen:
                seen.add(kw)
                result.append(kw)
    return result


def match_concept_keywords(
    text: str,
    hints: dict[str, DomainHint],
    domain_names: list[str] | None = None,
) -> list[str]:
    """Extract concept keywords that appear in the given text.

    Args:
        text: The text to search in.
        hints: Loaded domain hints.
        domain_names: If provided, only consider keywords from these domains.
            An empty list means no domains are selected (returns []).
            None means all domains.

    Returns:
        A deduplicated list of matching concept keywords (lowercased).
    """
    if domain_names is not None:
        filtered = {k: v for k, v in hints.items() if k in domain_names}
    else:
        filtered = hints

    all_kw = get_all_concept_keywords(filtered)
    text_lower = text.lower()
    return [kw for kw in all_kw if kw in text_lower]
