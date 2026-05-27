"""Load skill definitions from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).parent / "skills"


@dataclass
class RetrievalConfig:
    enabled: bool = False
    top_k: int = 3
    include_code: bool = False


@dataclass
class Skill:
    name: str
    description: str
    system_prompt: str
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


def load_skills() -> dict[str, Skill]:
    """Load all .yaml skill files from the skills directory."""
    skills: dict[str, Skill] = {}
    for path in SKILLS_DIR.glob("*.yaml"):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        ret = data.get("retrieval", {})
        skill = Skill(
            name=data["name"],
            description=data["description"],
            system_prompt=data["system_prompt"].strip(),
            retrieval=RetrievalConfig(
                enabled=ret.get("enabled", False),
                top_k=ret.get("top_k", 3),
                include_code=ret.get("include_code", False),
            ),
        )
        skills[skill.name] = skill
    return skills
