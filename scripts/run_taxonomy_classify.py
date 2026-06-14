"""Run taxonomy classification on a random sample of KnowledgeUnits.

Usage:
    python scripts/run_taxonomy_classify.py [--limit N]

Defaults to 20 random units. Results are printed as a summary table and
the full taxonomy tree is dumped as JSON for review.
"""

import argparse
import json
import logging
import os
import sys
import random

# Make project packages importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Set a dummy JWT secret so config doesn't exit — not needed for classification
if not os.environ.get("JWT_SECRET_KEY"):
    os.environ["JWT_SECRET_KEY"] = "script-only-dummy-key-not-for-production"

from database import Base, engine, SessionLocal
from config import settings
from models import KnowledgeUnit, MethodNode, KnowledgeUnitNode

# Create tables if they don't exist yet
Base.metadata.create_all(bind=engine)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="Number of random units to classify")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    db = SessionLocal()

    try:
        # 1. Check how many KnowledgeUnits exist
        total_units = db.query(KnowledgeUnit).count()
        logger.info("Total KnowledgeUnits in DB: %d", total_units)
        if total_units == 0:
            logger.error("No KnowledgeUnits found. Nothing to classify.")
            return

        # 2. Get all units and pick a random sample
        all_units = db.query(KnowledgeUnit).all()
        sample_size = min(args.limit, len(all_units))
        sampled = random.sample(all_units, sample_size)

        # Show what we're classifying
        method_names = set()
        logger.info("=== Sampled %d KnowledgeUnits ===", sample_size)
        for u in sampled:
            mn = u.method_name or "(none)"
            method_names.add(mn)
            logger.info("  KU %d: method_name=%s | title=%s", u.id, mn, u.title[:80])

        distinct_methods = sorted(m for m in method_names if m != "(none)")
        logger.info("Distinct method names in sample: %d — %s", len(distinct_methods), distinct_methods)

        # 3. Check existing taxonomy before
        nodes_before = db.query(MethodNode).count()
        links_before = db.query(KnowledgeUnitNode).count()
        logger.info("Taxonomy before: %d nodes, %d links", nodes_before, links_before)

        # 4. Run classification
        from chat.taxonomy import classify_units_to_taxonomy
        result = classify_units_to_taxonomy(db, sampled, settings.OPENAI_API_KEY)
        logger.info("Classification result: %s", result)

        # 5. Check taxonomy after
        nodes_after = db.query(MethodNode).count()
        links_after = db.query(KnowledgeUnitNode).count()
        logger.info("Taxonomy after: %d nodes (+%d), %d links (+%d)",
                     nodes_after, nodes_after - nodes_before,
                     links_after, links_after - links_before)

        # 6. Dump full taxonomy tree
        nodes = db.query(MethodNode).all()
        from collections import defaultdict
        children_map = defaultdict(list)
        node_by_id = {}
        for n in nodes:
            node_by_id[n.id] = n
            children_map[n.parent_id].append(n)

        # Count linked KUs per node
        ku_counts = {}
        for link in db.query(KnowledgeUnitNode).all():
            ku_counts[link.method_node_id] = ku_counts.get(link.method_node_id, 0) + 1

        def build_tree(parent_id=None, depth=0):
            lines = []
            for node in sorted(children_map.get(parent_id, []), key=lambda n: n.name):
                indent = "  " * depth
                kus = ku_counts.get(node.id, 0)
                aliases_str = f" (aliases: {node.aliases})" if node.aliases else ""
                desc_str = f" — {node.description}" if node.description else ""
                lines.append(f"{indent}[{node.node_type}] {node.name} (id={node.id}, KUs={kus}){aliases_str}{desc_str}")
                lines.extend(build_tree(node.id, depth + 1))
            return lines

        tree_lines = build_tree()
        logger.info("\n=== TAXONOMY TREE ===")
        for line in tree_lines:
            print(line)

        # 7. Show units that were NOT linked (method_name was empty or didn't match)
        unlinked = []
        for u in sampled:
            has_link = db.query(KnowledgeUnitNode).filter(
                KnowledgeUnitNode.knowledge_unit_id == u.id
            ).first()
            if not has_link:
                unlinked.append(u)

        if unlinked:
            logger.info("\n=== UNLINKED UNITS (%d) ===", len(unlinked))
            for u in unlinked:
                logger.info("  KU %d: method_name=%s | title=%s",
                            u.id, u.method_name or "(none)", u.title[:80])

    finally:
        db.close()


if __name__ == "__main__":
    main()
