#!/usr/bin/env python3
"""
Batch PDF Parsing Orchestrator

Generates prompt files for Claude subagents to parse papers into knowledge units.

Usage:
    python scripts/parse_papers.py --domain missing_data --batch-size 5
    python scripts/parse_papers.py --list
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Project root: two levels up from this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
CONFIG_PATH = PAPERS_DIR / "download_config.json"
METADATA_DIR = PAPERS_DIR / "metadata"
PARSED_DIR = PAPERS_DIR / "parsed"
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "scripts" / "parse_prompt.txt"


def load_config() -> dict:
    """Load download_config.json."""
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(domain: str) -> list[dict]:
    """Load metadata for a domain."""
    meta_path = METADATA_DIR / f"{domain}.json"
    if not meta_path.exists():
        print(f"Error: Metadata file not found: {meta_path}")
        print(f"  Run 'python scripts/fetch_metadata.py --domain {domain}' first.")
        sys.exit(1)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template() -> str:
    """Load the prompt template."""
    if not PROMPT_TEMPLATE_PATH.exists():
        print(f"Error: Prompt template not found: {PROMPT_TEMPLATE_PATH}")
        sys.exit(1)
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_existing_outputs(domain: str) -> set[str]:
    """Get set of filenames that already have output JSONs."""
    output_dir = PARSED_DIR / domain / "output"
    if not output_dir.exists():
        return set()
    existing = set()
    for f in output_dir.iterdir():
        if f.suffix == ".json":
            # Output filename is <pdf_stem>.json, so map back to <pdf_stem>.pdf
            existing.add(f.stem + ".pdf")
    return existing


def get_pdf_path(config: dict, domain: str, paper: dict) -> str:
    """Build the PDF path for a paper."""
    domain_config = config[domain]
    pdf_parent = domain_config["pdf_parent"]
    cluster = paper.get("cluster", "")
    filename = paper["filename"]
    return f"papers/pdf/{pdf_parent}/{cluster}/{filename}"


def format_paper_entry(idx: int, paper: dict, pdf_path: str) -> str:
    """Format a single paper entry for the prompt."""
    lines = [
        f"### Paper {idx}",
        f"- **Title:** {paper.get('title', 'Unknown')}",
        f"- **Authors:** {paper.get('authors', 'Unknown')}",
        f"- **Year:** {paper.get('year', 'Unknown')}",
        f"- **DOI:** {paper.get('doi', '')}",
        f"- **Domain:** {paper.get('domain', '')}",
        f"- **Cluster:** {paper.get('cluster', '')}",
        f"- **PDF Path:** {pdf_path}",
        f"- **Output Filename:** {Path(paper['filename']).stem}.json",
        "",
    ]
    return "\n".join(lines)


def list_domains(config: dict) -> None:
    """Print all available domains with paper counts."""
    print(f"{'Domain':<40} {'Papers':>8}  {'Clusters':>8}  Metadata")
    print("-" * 80)
    for domain, info in sorted(config.items()):
        meta_path = METADATA_DIR / f"{domain}.json"
        meta_status = "yes" if meta_path.exists() else "NO"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_count = len(json.load(f))
            meta_status = f"yes ({meta_count})"
        print(
            f"{domain:<40} {info['total_papers']:>8}  "
            f"{info['num_clusters']:>8}  {meta_status}"
        )


def generate_batches(
    domain: str, batch_size: int, config: dict, dry_run: bool = False
) -> None:
    """Generate batch prompt files for a domain."""
    metadata = load_metadata(domain)
    template = load_prompt_template()
    existing = get_existing_outputs(domain)

    # Filter out already-parsed papers
    pending = [p for p in metadata if p["filename"] not in existing]

    if not pending:
        print(f"All {len(metadata)} papers in '{domain}' already have output JSONs.")
        return

    print(f"Domain: {domain}")
    print(f"  Total papers: {len(metadata)}")
    print(f"  Already parsed: {len(existing)}")
    print(f"  Pending: {len(pending)}")
    print(f"  Batch size: {batch_size}")

    # Create output directories
    prompts_dir = PARSED_DIR / domain / "prompts"
    output_dir = PARSED_DIR / domain / "output"

    if not dry_run:
        prompts_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate batches
    num_batches = (len(pending) + batch_size - 1) // batch_size
    print(f"  Generating {num_batches} batch prompt(s)...")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(pending))
        batch_papers = pending[start:end]

        # Build papers list string
        papers_list_parts = []
        for i, paper in enumerate(batch_papers, 1):
            pdf_path = get_pdf_path(config, domain, paper)
            papers_list_parts.append(format_paper_entry(i, paper, pdf_path))

        papers_list_str = "\n".join(papers_list_parts)

        # Use forward slashes for the output dir path (cross-platform)
        output_dir_str = f"papers/parsed/{domain}/output"

        # Fill template
        prompt = template.replace("{papers_list}", papers_list_str)
        prompt = prompt.replace("{output_dir}", output_dir_str)
        prompt = prompt.replace("{domain}", domain)

        if dry_run:
            print(f"\n  Batch {batch_idx + 1}: {len(batch_papers)} papers")
            for p in batch_papers:
                print(f"    - {p['filename']}")
        else:
            prompt_path = prompts_dir / f"batch_{batch_idx + 1:03d}.md"
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"  Wrote: {prompt_path.relative_to(PROJECT_ROOT)} ({len(batch_papers)} papers)")

    if not dry_run:
        print(f"\nDone. Prompt files are in: {prompts_dir.relative_to(PROJECT_ROOT)}/")
        print(f"Output JSONs should be saved to: {output_dir.relative_to(PROJECT_ROOT)}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate batch prompt files for PDF parsing subagents"
    )
    parser.add_argument(
        "--domain", type=str, help="Domain to process (e.g., missing_data)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Papers per batch (default: 5)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available domains"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )

    args = parser.parse_args()
    config = load_config()

    if args.list:
        list_domains(config)
        return

    if not args.domain:
        parser.error("--domain is required (or use --list)")

    if args.domain not in config:
        print(f"Error: Unknown domain '{args.domain}'")
        print(f"Available domains: {', '.join(sorted(config.keys()))}")
        sys.exit(1)

    generate_batches(args.domain, args.batch_size, config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
