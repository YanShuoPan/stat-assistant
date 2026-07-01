"""Backfill concept keywords on existing KnowledgeUnits.

Usage:
    python scripts/backfill_concept_keywords.py --dry-run --limit 5
    python scripts/backfill_concept_keywords.py --db-url sqlite:///./test.db
    python scripts/backfill_concept_keywords.py  # runs against DATABASE_URL from .env
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from models import KnowledgeUnit
from chat.domain_loader import load_domain_hints, match_concept_keywords


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Backfill concept keywords on KUs")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "sqlite:///./test.db"))
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--limit", type=int, default=0, help="Process only N units (0=all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N units")
    args = parser.parse_args()

    engine = create_engine(args.db_url)
    Session = sessionmaker(bind=engine)
    db = Session()

    hints = load_domain_hints()
    print(f"Loaded {len(hints)} domain hints")

    query = db.query(KnowledgeUnit).order_by(KnowledgeUnit.id)
    if args.offset:
        query = query.offset(args.offset)
    if args.limit:
        query = query.limit(args.limit)

    units = query.all()
    print(f"Processing {len(units)} knowledge units...")

    updated = 0
    for ku in units:
        # Build text to match against
        text = f"{ku.title or ''} {ku.content or ''}"

        # Search all domains for concept keyword matches
        matched = match_concept_keywords(text, hints, domain_names=None)
        if not matched:
            continue

        existing = ku.keywords or []
        existing_lower = {k.lower() for k in existing}
        new_kw = [kw for kw in matched if kw.lower() not in existing_lower]

        if not new_kw:
            continue

        merged = existing + new_kw
        if args.dry_run:
            print(f"  KU {ku.id} ({ku.title}): +{len(new_kw)} keywords: {new_kw}")
        else:
            ku.keywords = merged
        updated += 1

    if not args.dry_run:
        db.commit()
        print(f"Committed {updated} updates")
    else:
        print(f"[DRY RUN] Would update {updated} units")

    db.close()


if __name__ == "__main__":
    main()
