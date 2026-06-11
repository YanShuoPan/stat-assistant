"""Fetch paper metadata from Semantic Scholar API for all domains.

Usage:
    python scripts/fetch_metadata.py                      # run all domains
    python scripts/fetch_metadata.py --domain survey      # substring match
    python scripts/fetch_metadata.py --domain hd          # matches high-dimensional
    python scripts/fetch_metadata.py --dry-run            # show what would be fetched
    python scripts/fetch_metadata.py --list               # show available domains

Resumes automatically. Saves per-domain JSON in papers/metadata/.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "papers"
CONFIG_FILE = BASE_DIR / "download_config.json"
METADATA_DIR = BASE_DIR / "metadata"
S2_API = "https://api.semanticscholar.org/graph/v1/paper"

API_DELAY = 5.0  # S2 free tier is strict on rate limits
MAX_RETRIES = 3


def normalize_doi(doi):
    """Strip URL prefix from DOI, returning bare DOI."""
    if doi.startswith("http"):
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return doi


def fetch_s2_metadata(doi):
    """Fetch metadata from Semantic Scholar by DOI.

    Returns dict with title, authors, year, arxiv_id or None on failure.
    """
    bare_doi = normalize_doi(doi)
    encoded = urllib.parse.quote(bare_doi, safe="")
    url = f"{S2_API}/DOI:{encoded}?fields=title,authors,year,externalIds"
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ModelBridge/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            authors_list = data.get("authors") or []
            author_names = ", ".join(
                a.get("name", "") for a in authors_list if a.get("name")
            )
            ext = data.get("externalIds", {})
            return {
                "title": data.get("title", ""),
                "authors": author_names or None,
                "year": data.get("year"),
                "arxiv_id": ext.get("ArXiv"),
            }
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = API_DELAY * (2 ** attempt) + 2
                print(f"    [429] Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
            elif e.code == 404:
                return None
            else:
                print(f"    [ERROR] HTTP {e.code} for DOI {bare_doi}")
                return None
        except Exception as e:
            print(f"    [ERROR] {e}")
            return None
    return None


def find_pdf_files(pdf_dir):
    """Return list of PDF filenames in a directory."""
    if not pdf_dir.exists():
        return []
    return [f.name for f in pdf_dir.iterdir() if f.suffix.lower() == ".pdf"]


def match_pdf_to_paper(title, pdf_files):
    """Try to match a paper title to a PDF filename.

    Uses a simple heuristic: normalize both and check for substring overlap.
    Returns the best matching filename or None.
    """
    if not pdf_files:
        return None

    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    norm_title = normalize(title)
    if len(norm_title) < 5:
        return None

    title_words = set(re.findall(r"[a-z]{4,}", title.lower()))
    best_score = 0
    best_file = None

    for fname in pdf_files:
        norm_fname = normalize(fname.replace(".pdf", ""))
        if norm_title[:30] in norm_fname or norm_fname[:30] in norm_title:
            return fname
        fname_words = set(re.findall(r"[a-z]{4,}", fname.lower()))
        overlap = len(title_words & fname_words)
        if overlap > best_score and overlap >= 3:
            best_score = overlap
            best_file = fname

    return best_file


def load_existing_metadata(path):
    """Load existing metadata JSON for resume support."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, Exception):
            return []
    return []


def save_metadata(path, records):
    """Save metadata records to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def match_domains(config, pattern):
    """Return list of domain keys matching a substring pattern."""
    pattern = pattern.lower().replace("-", "_").replace(" ", "_")
    matched = [k for k in config if pattern in k.lower()]
    if not matched:
        matched = [k for k in config if pattern in config[k]["pdf_parent"].lower()]
    return sorted(matched)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch paper metadata from Semantic Scholar API"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain substring filter (e.g. survey, bayesian)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched, no API calls",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available domains and exit"
    )
    args = parser.parse_args()

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    if args.list:
        print(f"Available domains ({len(config)}):")
        for k, v in sorted(config.items()):
            total = v["total_papers"]
            nc = v["num_clusters"]
            print(f"  {k:45s} {total:>5d} papers, {nc} clusters")
        return

    if args.domain:
        domain_keys = match_domains(config, args.domain)
        if not domain_keys:
            print("No domain matching. Use --list to see available domains.")
            return
        joined = ", ".join(domain_keys)
        print(f"Matched domains: {joined}")
    else:
        domain_keys = sorted(config.keys())

    grand_total = 0
    grand_fetched = 0
    grand_skipped = 0
    grand_no_doi = 0
    grand_failed = 0

    for domain_key in domain_keys:
        dc = config[domain_key]
        clusters_file = BASE_DIR / dc["clusters_file"]
        cluster_folders = dc["cluster_folders"]
        pdf_parent = dc["pdf_parent"]

        if not clusters_file.exists():
            print(f"[SKIP] {domain_key}: clusters file not found")
            continue

        raw = json.loads(clusters_file.read_text(encoding="utf-8"))
        clusters = raw.get("clusters", raw)

        metadata_path = METADATA_DIR / f"{domain_key}.json"
        existing = load_existing_metadata(metadata_path)
        existing_dois = {r["doi"] for r in existing if r.get("doi")}
        existing_titles = {r["title"] for r in existing if r.get("title")}

        sep = "#" * 60
        print(f"\n{sep}")
        total = dc["total_papers"]
        print(f"DOMAIN: {domain_key} ({total} papers)")
        print(f"  Existing metadata: {len(existing)} records")
        print(sep)

        records = list(existing)
        domain_fetched = 0
        domain_skipped = 0
        domain_no_doi = 0
        domain_failed = 0

        for ck in sorted(clusters.keys()):
            if ck not in clusters:
                continue
            idx = ck.split("_")[1]
            folder_name = cluster_folders.get(idx, f"cluster_{idx}")

            papers = clusters[ck]["papers"]
            pdf_dir = BASE_DIR / "pdf" / pdf_parent / folder_name
            pdf_files = find_pdf_files(pdf_dir)
            # Also search the flat 'all' directory if domain dir is empty
            all_dir = BASE_DIR / "pdf" / "all"
            if not pdf_files and all_dir.exists():
                pdf_files = find_pdf_files(all_dir)

            n_papers = len(papers)
            n_pdfs = len(pdf_files)
            print(
                f"\n  {ck} -> {folder_name}"
                f" ({n_papers} papers, {n_pdfs} PDFs on disk)"
            )

            for i, paper in enumerate(papers):
                title = paper["title"]
                doi = paper.get("doi", "")
                bare_doi = normalize_doi(doi) if doi else None
                grand_total += 1

                if bare_doi and bare_doi in existing_dois:
                    domain_skipped += 1
                    grand_skipped += 1
                    continue
                if title in existing_titles:
                    domain_skipped += 1
                    grand_skipped += 1
                    continue

                filename = match_pdf_to_paper(title, pdf_files)

                if not doi:
                    if args.dry_run:
                        safe_t = title.encode("ascii", "replace").decode()
                        print(
                            f"    [{i+1}/{n_papers}] NO_DOI: {safe_t[:65]}"
                        )
                    else:
                        records.append({
                            "title": title,
                            "authors": None,
                            "year": None,
                            "doi": None,
                            "arxiv_id": None,
                            "domain": domain_key,
                            "cluster": folder_name,
                            "filename": filename,
                        })
                    domain_no_doi += 1
                    grand_no_doi += 1
                    continue

                safe_t = title.encode("ascii", "replace").decode()
                if args.dry_run:
                    print(f"    [{i+1}/{n_papers}] FETCH: {safe_t[:65]}")
                    print(f"           DOI: {bare_doi}")
                    domain_fetched += 1
                    grand_fetched += 1
                    continue

                print(f"    [{i+1}/{n_papers}] {safe_t[:65]}")
                result = fetch_s2_metadata(doi)
                time.sleep(API_DELAY)

                if result is not None:
                    s2_title = result["title"] or title
                    authors = result["authors"]
                    year = result["year"]
                    arxiv_id = result["arxiv_id"]
                    records.append({
                        "title": s2_title,
                        "authors": authors,
                        "year": year,
                        "doi": bare_doi,
                        "arxiv_id": arxiv_id,
                        "domain": domain_key,
                        "cluster": folder_name,
                        "filename": filename,
                    })
                    domain_fetched += 1
                    grand_fetched += 1
                    ad = authors if authors else "Unknown"
                    if len(ad) > 50:
                        ad = ad[:50] + "..."
                    print(f"      -> {ad} ({year})".encode("ascii", "replace").decode())
                else:
                    records.append({
                        "title": title,
                        "authors": None,
                        "year": None,
                        "doi": bare_doi,
                        "arxiv_id": None,
                        "domain": domain_key,
                        "cluster": folder_name,
                        "filename": filename,
                    })
                    domain_failed += 1
                    grand_failed += 1
                    print("      -> FAILED (recorded with null metadata)")

                if domain_fetched % 10 == 0 and domain_fetched > 0:
                    save_metadata(metadata_path, records)

        if not args.dry_run:
            save_metadata(metadata_path, records)
            n_rec = len(records)
            print(f"\n  Saved {n_rec} records to {metadata_path.name}")

        print(
            f"  Domain summary: fetched={domain_fetched}"
            f" skipped={domain_skipped}"
            f" no_doi={domain_no_doi}"
            f" failed={domain_failed}"
        )

    sep = "=" * 60
    print(f"\n{sep}")
    print("Summary:")
    print(f"  Total papers:  {grand_total}")
    print(f"  Fetched:       {grand_fetched}")
    print(f"  Skipped:       {grand_skipped} (already in metadata)")
    print(f"  No DOI:        {grand_no_doi}")
    print(f"  API failed:    {grand_failed}")
    if args.dry_run:
        print("  (dry-run mode -- no API calls made)")
    print(sep)


if __name__ == "__main__":
    main()
