"""
PDF auto-rename tool
Usage: python tools/rename_papers.py papers/pdf/High-Dimensional/HD_Change_Point_Detection

Scans PDFs in folder, matches to paper list by DOI or filename,
renames to {FirstAuthor}_{Year}_{ShortTitle}.pdf
"""
import sys, io, os, re, csv
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")

CSVPATH = Path(__file__).parent.parent / "papers" / "top20_stats_journals_2021_2025_tagged.csv"

def load_papers():
    with open(CSVPATH, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def make_filename(paper):
    authors = paper.get("authors", "")
    first_author = authors.split(";")[0].strip().split()[-1] if authors else "Unknown"
    year = paper.get("year", "")
    title_words = re.sub(r"[^\w\s]", "", paper.get("title", "")).split()[:5]
    short_title = "_".join(title_words)
    return f"{first_author}_{year}_{short_title}.pdf"

def extract_doi_from_pdf(filepath):
    try:
        with open(filepath, "rb") as f:
            content = f.read(50000).decode("latin-1", errors="ignore")
        match = re.search(r"10\.\d{4,}/[^\s\"<>]+", content)
        if match:
            return match.group(0).rstrip(".")
    except Exception:
        pass
    return None

def match_by_doi(doi, papers):
    for p in papers:
        p_doi = p.get("doi", "").replace("https://doi.org/", "")
        if doi and p_doi and doi.lower() == p_doi.lower():
            return p
    return None

def match_by_filename(filename, papers):
    name = Path(filename).stem.lower()
    name_clean = re.sub(r"[_\-\.\(\)]", " ", name)
    words = set(name_clean.split())
    if len(words) < 2:
        return None
    best = None
    best_score = 0
    for p in papers:
        title = p.get("title", "").lower()
        title_words = set(re.sub(r"[^\w\s]", "", title).split())
        if not title_words:
            continue
        overlap = len(words & title_words)
        score = overlap / max(len(words), 1)
        if score > best_score and score > 0.4:
            best_score = score
            best = p
    return best

if len(sys.argv) < 2:
    print("Usage: python tools/rename_papers.py <pdf_folder>")
    sys.exit(1)

folder = Path(sys.argv[1])
if not folder.is_dir():
    print(f"Folder not found: {folder}")
    sys.exit(1)

papers = load_papers()
pdfs = list(folder.glob("*.pdf"))

if not pdfs:
    print(f"No PDFs in: {folder}")
    sys.exit(0)

print(f"Found {len(pdfs)} PDFs\n")

renames = []
unmatched = []

for pdf in pdfs:
    doi = extract_doi_from_pdf(pdf)
    matched = None
    if doi:
        matched = match_by_doi(doi, papers)
    if not matched:
        matched = match_by_filename(pdf.name, papers)
    if matched:
        new_name = make_filename(matched)
        renames.append((pdf, folder / new_name, matched["title"][:60]))
    else:
        unmatched.append(pdf)

if renames:
    print("=== Rename preview ===\n")
    for old, new, title in renames:
        print(f"  {old.name}")
        print(f"  -> {new.name}")
        print(f"     ({title})")
        print()
    choice = input(f"Rename {len(renames)} files? (y/n): ").strip().lower()
    if choice == "y":
        for old, new, _ in renames:
            if new.exists():
                print(f"  Skip (exists): {new.name}")
                continue
            old.rename(new)
            print(f"  OK: {new.name}")
        print(f"\nDone! Renamed {len(renames)} files")
    else:
        print("Cancelled")

if unmatched:
    print(f"\nUnmatched PDFs ({len(unmatched)}):")
    for pdf in unmatched:
        print(f"  {pdf.name}")
