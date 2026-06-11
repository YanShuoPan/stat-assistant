#!/usr/bin/env python3
"""
DB Import Script for Parsed Paper Knowledge Units

Reads parsed JSON output files and imports Paper + KnowledgeUnit records
into the database.

Usage:
    python scripts/import_parsed.py --list
    python scripts/import_parsed.py --domain missing_data --dry-run
    python scripts/import_parsed.py --domain missing_data
    python scripts/import_parsed.py --domain all --skip-embeddings
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup - allow importing from apps/api and packages
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))
sys.path.insert(0, str(PROJECT_ROOT / "packages"))

os.environ.setdefault("DATABASE_URL", "sqlite:///model_bridge.db")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("JWT_SECRET_KEY", "unused")

from database import engine, SessionLocal, Base  # noqa: E402
from models import Paper, KnowledgeUnit  # noqa: E402
from chat.embeddings import compute_embedding, unit_to_embedding_text  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPERS_DIR = PROJECT_ROOT / "papers"
CONFIG_PATH = PAPERS_DIR / "download_config.json"
METADATA_DIR = PAPERS_DIR / "metadata"
PARSED_DIR = PAPERS_DIR / "parsed"


def load_config() -> dict:
    """Load download_config.json."""
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(domain: str) -> list[dict]:
    """Load metadata for a domain. Returns empty list if file missing."""
    meta_path = METADATA_DIR / f"{domain}.json"
    if not meta_path.exists():
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_parsed_files(domain: str) -> list[Path]:
    """Get all parsed output JSON files for a domain."""
    output_dir = PARSED_DIR / domain / "output"
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*.json"))


def build_metadata_index(metadata: list[dict]) -> dict:
    """Build lookup dicts by DOI and by title (lowercased) for matching."""
    by_doi = {}
    by_title = {}
    for entry in metadata:
        doi = (entry.get("doi") or "").strip().lower()
        if doi:
            by_doi[doi] = entry
        title = (entry.get("title") or "").strip().lower()
        if title:
            by_title[title] = entry
    return {"doi": by_doi, "title": by_title}


def match_metadata(parsed: dict, meta_index: dict) -> dict | None:
    """Find matching metadata entry for a parsed file."""
    doi = (parsed.get("paper_doi") or "").strip().lower()
    if doi and doi in meta_index["doi"]:
        return meta_index["doi"][doi]
    title = (parsed.get("paper_title") or "").strip().lower()
    if title and title in meta_index["title"]:
        return meta_index["title"][title]
    return None


def list_domains(config: dict) -> None:
    """Print all domains with parsed file counts."""
    header = f"{'Domain':<40} {'Parsed':>8}  {'Metadata':>10}"
    print(header)
    print("-" * 65)
    for domain in sorted(config.keys()):
        parsed_files = get_parsed_files(domain)
        meta_path = METADATA_DIR / f"{domain}.json"
        meta_status = "no"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_status = f"yes ({len(json.load(f))})"
        print(f"{domain:<40} {len(parsed_files):>8}  {meta_status:>10}")


def import_domain(
    domain: str,
    config: dict,
    dry_run: bool = False,
    skip_embeddings: bool = False,
) -> dict:
    """Import all parsed files for a domain. Returns summary stats."""
    stats = {
        "domain": domain,
        "files_found": 0,
        "papers_created": 0,
        "papers_skipped": 0,
        "units_created": 0,
        "units_skipped_no_content": 0,
        "embeddings_computed": 0,
        "errors": [],
    }

    parsed_files = get_parsed_files(domain)
    stats["files_found"] = len(parsed_files)

    if not parsed_files:
        print(f"  No parsed files found for '{domain}'.")
        return stats

    metadata = load_metadata(domain)
    meta_index = build_metadata_index(metadata)
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not dry_run:
        Base.metadata.create_all(bind=engine)

    db = None if dry_run else SessionLocal()

    try:
        for pf in parsed_files:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                msg = f"Error reading {pf.name}: {e}"
                print(f"  SKIP: {msg}")
                stats["errors"].append(msg)
                continue

            paper_filename = pf.stem + ".pdf"

            # Check for existing paper (dedup by filename + domain)
            if not dry_run:
                existing = (
                    db.query(Paper)
                    .filter(Paper.filename == paper_filename, Paper.domain == domain)
                    .first()
                )
                if existing:
                    print(f"  SKIP (exists): {paper_filename}")
                    stats["papers_skipped"] += 1
                    continue

            # Find metadata match
            meta = match_metadata(parsed, meta_index)

            # Build Paper fields - prefer metadata, fallback to parsed JSON
            paper_title = (
                (meta.get("title") if meta else None)
                or parsed.get("paper_title")
                or pf.stem
            )
            paper_authors = (meta.get("authors") if meta else None) or None
            paper_year = (meta.get("year") if meta else None) or None
            paper_doi = (
                (meta.get("doi") if meta else None)
                or parsed.get("paper_doi")
                or None
            )
            paper_arxiv = (meta.get("arxiv_id") if meta else None) or None
            paper_cluster = (meta.get("cluster") if meta else None) or None

            knowledge_units = parsed.get("knowledge_units", [])

            if dry_run:
                meta_label = "matched" if meta else "no-meta"
                print(
                    f"  IMPORT: {paper_filename} ({meta_label}) "
                    f"-- {len(knowledge_units)} units"
                )
                stats["papers_created"] += 1
                stats["units_created"] += len(knowledge_units)
                continue

            # Create Paper record
            paper = Paper(
                title=paper_title,
                authors=paper_authors,
                year=paper_year,
                doi=paper_doi,
                arxiv_id=paper_arxiv,
                domain=domain,
                cluster=paper_cluster,
                filename=paper_filename,
            )
            db.add(paper)
            db.flush()  # get paper.id without committing yet
            stats["papers_created"] += 1

            # Create KnowledgeUnit records
            for ku_data in knowledge_units:
                content = (ku_data.get("content") or "").strip()
                if not content:
                    stats["units_skipped_no_content"] += 1
                    continue

                ku = KnowledgeUnit(
                    source_type=ku_data.get("source_type", "paper"),
                    title=ku_data.get("title", paper_title)[:255],
                    section=ku_data.get("section"),
                    knowledge_type=ku_data.get("knowledge_type", "unknown"),
                    topic_tags=ku_data.get("topic_tags", []),
                    question_intent_tags=ku_data.get("question_intent_tags", []),
                    content=content,
                    evidence_span=ku_data.get("evidence_span"),
                    dependencies=ku_data.get("dependencies"),
                    limitations=ku_data.get("limitations"),
                    confidence=ku_data.get("confidence", "medium"),
                    reusable_for_questions=ku_data.get("reusable_for_questions", []),
                    method_name=ku_data.get("method_name"),
                    field=ku_data.get("field"),
                    keywords=ku_data.get("keywords"),
                    problem_it_solves=ku_data.get("problem_it_solves"),
                    model_assumption=ku_data.get("model_assumption"),
                    input_format=ku_data.get("input_format"),
                    output_format=ku_data.get("output_format"),
                    typical_questions=ku_data.get("typical_questions"),
                    related_methods=ku_data.get("related_methods"),
                    paper_id=paper.id,
                )

                # Compute embedding if requested
                if not skip_embeddings and api_key:
                    emb_text = unit_to_embedding_text(ku_data)
                    if emb_text:
                        ku.embedding = compute_embedding(emb_text, api_key)
                        if ku.embedding:
                            stats["embeddings_computed"] += 1

                db.add(ku)
                stats["units_created"] += 1

            # Commit per paper (batch commit)
            db.commit()
            meta_label = "matched" if meta else "no-meta"
            print(
                f"  OK: {paper_filename} ({meta_label}) "
                f"-- paper_id={paper.id}, {len(knowledge_units)} units"
            )

    except Exception as e:
        if db:
            db.rollback()
        raise
    finally:
        if db:
            db.close()

    return stats


def print_summary(all_stats: list[dict]) -> None:
    """Print import summary."""
    total_files = sum(s["files_found"] for s in all_stats)
    total_papers = sum(s["papers_created"] for s in all_stats)
    total_skipped = sum(s["papers_skipped"] for s in all_stats)
    total_units = sum(s["units_created"] for s in all_stats)
    total_embeds = sum(s["embeddings_computed"] for s in all_stats)
    total_errors = sum(len(s["errors"]) for s in all_stats)

    print()
    print("=" * 50)
    print("Import Summary")
    print("=" * 50)
    print(f"  Parsed files found:  {total_files}")
    print(f"  Papers created:      {total_papers}")
    print(f"  Papers skipped:      {total_skipped}")
    print(f"  Knowledge units:     {total_units}")
    print(f"  Embeddings computed: {total_embeds}")
    if total_errors:
        print(f"  Errors:              {total_errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Import parsed paper knowledge units into the database"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Domain to import (e.g. missing_data), or 'all' for all domains",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without writing to DB",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding computation (much faster)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List domains with parsed file counts",
    )

    args = parser.parse_args()
    config = load_config()

    if args.list:
        list_domains(config)
        return

    if not args.domain:
        parser.error("--domain is required (or use --list)")

    if args.domain == "all":
        domains = sorted(config.keys())
    elif args.domain in config:
        domains = [args.domain]
    else:
        print(f"Error: Unknown domain '{args.domain}'")
        print(f"Available domains: {', '.join(sorted(config.keys()))}")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN (no DB writes) ===")
        print()

    all_stats = []
    for domain in domains:
        print(f"--- Domain: {domain} ---")
        stats = import_domain(
            domain,
            config,
            dry_run=args.dry_run,
            skip_embeddings=args.skip_embeddings,
        )
        all_stats.append(stats)

    print_summary(all_stats)


if __name__ == "__main__":
    main()
