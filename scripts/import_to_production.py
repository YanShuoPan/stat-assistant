#!/usr/bin/env python3
"""
Import parsed paper JSON files to production API.

Usage:
    python scripts/import_to_production.py --list
    python scripts/import_to_production.py --domain all --dry-run
    python scripts/import_to_production.py --domain all
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
CONFIG_PATH = PAPERS_DIR / "download_config.json"
METADATA_DIR = PAPERS_DIR / "metadata"
PARSED_DIR = PAPERS_DIR / "parsed"

API_BASE = "https://stat-research-assistant-pfemb.ondigitalocean.app/api"


def login(username: str, password: str) -> str:
    """Login and return JWT token."""
    resp = requests.post(
        f"{API_BASE}/auth/login",
        json={"username": username, "password": password},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(domain: str) -> dict:
    meta_path = METADATA_DIR / f"{domain}.json"
    if not meta_path.exists():
        return {"doi": {}, "title": {}}
    with open(meta_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    by_doi, by_title = {}, {}
    for e in entries:
        doi = (e.get("doi") or "").strip().lower()
        if doi:
            by_doi[doi] = e
        title = (e.get("title") or "").strip().lower()
        if title:
            by_title[title] = e
    return {"doi": by_doi, "title": by_title}


def get_parsed_files(domain: str) -> list[Path]:
    output_dir = PARSED_DIR / domain / "output"
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*.json"))


def _to_str(v):
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    return v


def _to_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) if not isinstance(x, str) else x for x in v]
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [str(x) if not isinstance(x, str) else x for x in parsed]
            return [v]
        except Exception:
            return [v]
    return [str(v)]


def build_payload(parsed: dict, domain: str, filename: str, meta_index: dict) -> dict:
    """Convert parsed JSON to API KnowledgeUnitBulkCreate payload."""
    pm = parsed.get("paper_metadata", {})
    parsed_title = pm.get("title") or parsed.get("paper_title") or ""
    parsed_doi = (pm.get("doi") or parsed.get("paper_doi") or "").strip().lower()

    # Match with metadata
    meta = None
    if parsed_doi and parsed_doi in meta_index["doi"]:
        meta = meta_index["doi"][parsed_doi]
    elif parsed_title.strip().lower() in meta_index["title"]:
        meta = meta_index["title"][parsed_title.strip().lower()]

    paper_title = (meta.get("title") if meta else None) or parsed_title or filename
    raw_authors = (meta.get("authors") if meta else None) or pm.get("authors")
    paper_authors = ", ".join(raw_authors) if isinstance(raw_authors, list) else raw_authors
    paper_year = (meta.get("year") if meta else None) or pm.get("year")
    paper_doi = (meta.get("doi") if meta else None) or pm.get("doi") or None
    paper_arxiv = (meta.get("arxiv_id") if meta else None) or pm.get("arxiv_id")
    paper_cluster = (meta.get("cluster") if meta else None) or pm.get("cluster")

    paper = {
        "title": paper_title[:500],
        "authors": paper_authors,
        "year": paper_year,
        "doi": paper_doi,
        "arxiv_id": paper_arxiv,
        "domain": domain,
        "cluster": paper_cluster,
        "filename": filename,
    }

    units = []
    for ku in parsed.get("knowledge_units", []):
        content = (ku.get("content") or "").strip()
        if not content:
            continue

        conf = ku.get("confidence", "medium")
        if isinstance(conf, (int, float)):
            conf = "high" if conf >= 0.8 else ("medium" if conf >= 0.5 else "low")
        elif not isinstance(conf, str):
            conf = "medium"

        units.append({
            "source_type": ku.get("source_type", "paper"),
            "title": (ku.get("title") or paper_title)[:255],
            "section": ku.get("section"),
            "knowledge_type": ku.get("knowledge_type", "unknown"),
            "topic_tags": _to_list(ku.get("topic_tags")),
            "question_intent_tags": _to_list(ku.get("question_intent_tags")),
            "content": content,
            "evidence_span": ku.get("evidence_span"),
            "dependencies": _to_list(ku.get("dependencies")),
            "limitations": _to_str(ku.get("limitations")),
            "confidence": conf,
            "reusable_for_questions": _to_list(ku.get("reusable_for_questions")),
            "method_name": ku.get("method_name"),
            "field": ku.get("field"),
            "keywords": _to_list(ku.get("keywords")),
            "problem_it_solves": ku.get("problem_it_solves"),
            "model_assumption": ku.get("model_assumption"),
            "input_format": ku.get("input_format"),
            "output_format": ku.get("output_format"),
            "typical_questions": _to_list(ku.get("typical_questions")),
            "related_methods": _to_list(ku.get("related_methods")),
        })

    return {"units": units, "paper": paper}


def import_file(token: str, payload: dict, skip_embeddings: bool) -> dict:
    """POST one paper + its units to the API."""
    params = {}
    if skip_embeddings:
        params["skip_embeddings"] = "true"
        params["skip_postprocessing"] = "true"

    resp = requests.post(
        f"{API_BASE}/knowledge/upload",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def list_domains(config: dict):
    header = f"{'Domain':<45} {'Parsed':>8}"
    print(header)
    print("-" * 58)
    for domain in sorted(config.keys()):
        files = get_parsed_files(domain)
        print(f"{domain:<45} {len(files):>8}")


def main():
    parser = argparse.ArgumentParser(description="Import parsed papers to production API")
    parser.add_argument("--domain", type=str, help="Domain to import, or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported")
    parser.add_argument("--skip-embeddings", action="store_true", default=True,
                        help="Skip embedding computation (default: true)")
    parser.add_argument("--with-embeddings", action="store_true",
                        help="Compute embeddings during import")
    parser.add_argument("--list", action="store_true", help="List domains")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="admin123")
    args = parser.parse_args()

    config = load_config()

    if args.list:
        list_domains(config)
        return

    if not args.domain:
        parser.error("--domain is required (or use --list)")

    skip_emb = not args.with_embeddings

    if args.domain == "all":
        domains = sorted(config.keys())
    elif args.domain in config:
        domains = [args.domain]
    else:
        print(f"Unknown domain: {args.domain}")
        sys.exit(1)

    if not args.dry_run:
        print("Logging in...")
        token = login(args.username, args.password)
        print("Logged in.")
    else:
        token = None
        print("=== DRY RUN ===\n")

    total_papers = 0
    total_units = 0
    errors = []

    for domain in domains:
        files = get_parsed_files(domain)
        if not files:
            continue

        meta_index = load_metadata(domain)
        print(f"\n--- {domain} ({len(files)} files) ---")

        for pf in files:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
            except Exception as e:
                print(f"  SKIP (read error): {pf.name}: {e}")
                errors.append(f"{pf.name}: {e}")
                continue

            filename = pf.stem + ".pdf"
            payload = build_payload(parsed, domain, filename, meta_index)

            if not payload["units"]:
                print(f"  SKIP (no units): {filename}")
                continue

            if args.dry_run:
                print(f"  IMPORT: {filename} -- {len(payload['units'])} units")
                total_papers += 1
                total_units += len(payload["units"])
                continue

            try:
                result = import_file(token, payload, skip_emb)
                total_papers += 1
                total_units += len(result)
                print(f"  OK: {filename} -- {len(result)} units saved")
            except requests.HTTPError as e:
                msg = f"{filename}: {e.response.status_code} {e.response.text[:200]}"
                print(f"  ERROR: {msg}")
                errors.append(msg)
            except Exception as e:
                msg = f"{filename}: {e}"
                print(f"  ERROR: {msg}")
                errors.append(msg)

            # Small delay to avoid overwhelming the server
            time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"Papers imported: {total_papers}")
    print(f"Units imported:  {total_units}")
    if errors:
        print(f"Errors:          {len(errors)}")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
