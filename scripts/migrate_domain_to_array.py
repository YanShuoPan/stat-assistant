"""Migrate Paper.domain from string to JSON array.

Converts existing single-string domain values (e.g. "statistics") into
JSON arrays (e.g. ["statistics"]).  Rows that already contain a JSON
array are silently skipped.

Usage:
    python scripts/migrate_domain_to_array.py              # uses DATABASE_URL from .env
    python scripts/migrate_domain_to_array.py --db-url "postgresql://..."
    python scripts/migrate_domain_to_array.py --dry-run    # preview without committing
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def migrate(db_url: str, *, dry_run: bool = False) -> None:
    engine = create_engine(db_url)
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, domain FROM papers")).fetchall()
        updated = 0
        skipped = 0
        for row in rows:
            paper_id, domain = row[0], row[1]

            # Already a list (JSON array) — skip
            if isinstance(domain, list):
                skipped += 1
                continue

            # String that looks like a JSON array — try parsing
            if isinstance(domain, str) and domain.startswith("["):
                try:
                    parsed = json.loads(domain)
                    if isinstance(parsed, list):
                        skipped += 1
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass

            # Convert plain string to single-element array
            new_domain = json.dumps([domain] if isinstance(domain, str) else [str(domain)])

            if dry_run:
                print(f"  [DRY RUN] paper {paper_id}: {domain!r} -> {new_domain}")
            else:
                conn.execute(
                    text("UPDATE papers SET domain = :domain WHERE id = :id"),
                    {"domain": new_domain, "id": paper_id},
                )
            updated += 1

        print(f"Done. Updated: {updated}, Skipped (already array): {skipped}")
        if dry_run:
            print("(dry run — no changes committed)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Paper.domain string -> JSON array")
    parser.add_argument("--db-url", default=None, help="Database URL (default: DATABASE_URL from .env)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without committing")
    args = parser.parse_args()

    db_url = args.db_url
    if not db_url:
        # Load .env from project root
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        load_dotenv(env_path)
        db_url = os.environ.get("DATABASE_URL")

    if not db_url:
        print("ERROR: No DATABASE_URL found. Pass --db-url or set DATABASE_URL in .env", file=sys.stderr)
        sys.exit(1)

    print(f"Migrating Paper.domain to JSON array ...")
    print(f"Database: {db_url.split('@')[-1] if '@' in db_url else '(local)'}")
    migrate(db_url, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
