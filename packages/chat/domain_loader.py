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
