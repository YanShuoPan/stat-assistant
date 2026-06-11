#!/usr/bin/env python3
"""
Parse Progress Tracker

Scans output JSONs and reports parsing progress per domain.

Usage:
    python scripts/mark_parsed.py --domain missing_data
    python scripts/mark_parsed.py --domain all
    python scripts/mark_parsed.py --domain missing_data --verbose
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
CONFIG_PATH = PAPERS_DIR / "download_config.json"
METADATA_DIR = PAPERS_DIR / "metadata"
PARSED_DIR = PAPERS_DIR / "parsed"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(domain: str) -> list[dict] | None:
    """Load metadata for a domain. Returns None if file doesn't exist."""
    meta_path = METADATA_DIR / f"{domain}.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_output_files(domain: str) -> dict[str, Path]:
    """Get mapping of stem -> path for output JSONs."""
    output_dir = PARSED_DIR / domain / "output"
    if not output_dir.exists():
        return {}
    return {f.stem: f for f in output_dir.iterdir() if f.suffix == ".json"}


def validate_output(json_path: Path) -> tuple[bool, str]:
    """Check if an output JSON is valid and has knowledge units."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "knowledge_units" in data:
            n_units = len(data["knowledge_units"])
            return True, f"{n_units} units"
        elif isinstance(data, list):
            # Also accept raw array of knowledge units
            return True, f"{len(data)} units (raw array)"
        else:
            return False, "missing knowledge_units key"
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    except Exception as e:
        return False, f"error: {e}"


def report_domain(domain: str, verbose: bool = False) -> dict:
    """Report progress for a single domain. Returns stats dict."""
    metadata = load_metadata(domain)
    if metadata is None:
        print(f"  {domain}: no metadata file (run fetch_metadata.py first)")
        return {"domain": domain, "status": "no_metadata"}

    outputs = get_output_files(domain)

    total = len(metadata)
    parsed_filenames = set()
    valid_count = 0
    invalid_count = 0
    invalid_files = []

    for paper in metadata:
        stem = Path(paper["filename"]).stem
        if stem in outputs:
            parsed_filenames.add(paper["filename"])
            ok, detail = validate_output(outputs[stem])
            if ok:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_files.append((paper["filename"], detail))

    missing = [p for p in metadata if p["filename"] not in parsed_filenames]
    pct = (valid_count / total * 100) if total > 0 else 0

    print(f"  {domain}: {valid_count}/{total} parsed ({pct:.1f}%)")
    if invalid_count > 0:
        print(f"    Invalid outputs: {invalid_count}")

    if verbose:
        if invalid_files:
            print("    --- Invalid ---")
            for fname, detail in invalid_files:
                print(f"      {fname}: {detail}")
        if missing:
            print(f"    --- Missing ({len(missing)}) ---")
            for p in missing[:20]:
                print(f"      {p['filename']}")
            if len(missing) > 20:
                print(f"      ... and {len(missing) - 20} more")

    return {
        "domain": domain,
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "missing": len(missing),
        "pct": pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Track PDF parsing progress")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to check (or 'all' for all domains)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show individual paper status"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    args = parser.parse_args()
    config = load_config()

    if args.domain == "all":
        domains = sorted(config.keys())
    else:
        if args.domain not in config:
            print(f"Error: Unknown domain '{args.domain}'")
            print(f"Available: {', '.join(sorted(config.keys()))}")
            sys.exit(1)
        domains = [args.domain]

    print("Parsing Progress Report")
    print("=" * 60)

    all_stats = []
    for domain in domains:
        stats = report_domain(domain, verbose=args.verbose)
        all_stats.append(stats)

    # Summary for "all"
    if len(domains) > 1:
        total_papers = sum(s.get("total", 0) for s in all_stats if "total" in s)
        total_valid = sum(s.get("valid", 0) for s in all_stats if "valid" in s)
        total_invalid = sum(s.get("invalid", 0) for s in all_stats if "invalid" in s)
        pct = (total_valid / total_papers * 100) if total_papers > 0 else 0
        print("=" * 60)
        print(f"  TOTAL: {total_valid}/{total_papers} parsed ({pct:.1f}%)")
        if total_invalid > 0:
            print(f"  Invalid: {total_invalid}")

    if args.json:
        print("\n" + json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
