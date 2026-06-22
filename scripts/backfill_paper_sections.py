#!/usr/bin/env python3
"""
Backfill paper_sections for existing papers that have no section data.

Reads original PDF files, splits into section chunks, generates LLM summaries,
and stores paper_sections + raw file bytes.

Usage:
    python scripts/backfill_paper_sections.py --dry-run
    python scripts/backfill_paper_sections.py
    python scripts/backfill_paper_sections.py --paper-id 1
"""

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))
sys.path.insert(0, str(PROJECT_ROOT / "packages"))

os.environ.setdefault("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'apps' / 'api' / 'dev.db'}")
os.environ.setdefault("JWT_SECRET_KEY", "unused")

from database import SessionLocal, Base, engine  # noqa: E402
from models import Paper, PaperSection  # noqa: E402
from config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known PDF locations for manually-imported papers.
# Maps paper.id to the PDF path relative to PROJECT_ROOT.
# Add entries here if you import more papers manually.
# ---------------------------------------------------------------------------
KNOWN_PDFS: dict[int, str] = {
    1: "method_DML/bouble_machine_learning.pdf",
    2: "method_OGA/2011_sinica.pdf",
    # Paper 3 (Stepwise Regression) has no separate PDF
    4: "method_bayesian/Bayesian_Optimization_Objective-Based_Experimental_Design.pdf",
    5: "method_deepkriging/Chen-DEEPKRIGING-2024.pdf",
}


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)


def _split_into_sections(text: str, max_chars: int = 12000) -> list[str]:
    """Split document text into sections at section-header boundaries.

    Replicates the logic from methods.py _split_into_chunks, but without
    the ``=== [...] ===`` prefix since we don't need it here.
    """
    import re

    if len(text) <= max_chars:
        return [text]

    section_pat = r"\n(?=\d+\.?\s+[A-Z]|#{1,3}\s)"
    parts = re.split(section_pat, text)

    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) > max_chars and current:
            chunks.append(current)
            current = part
        else:
            current += ("\n" if current else "") + part
    if current:
        chunks.append(current)

    return chunks


def _detect_section_type(chunk_text: str) -> str:
    """Detect section type from chunk text (mirrors methods.py logic)."""
    header = chunk_text[:300].lower()

    if "abstract" in header:
        return "abstract"
    if any(w in header for w in ("introduction", "motivation", "background")):
        return "introduction"
    if any(w in header for w in ("methodology", "method", "algorithm", "model",
                                  "framework", "proposed", "approach")):
        if "related" not in header:
            return "methodology"
    if any(w in header for w in ("theor", "convergence", "oracle", "consistency",
                                  "asymptotic", "properties")):
        return "theory"
    if any(w in header for w in ("experiment", "simulation", "numerical",
                                  "empirical result", "monte carlo")):
        return "experiment"
    if any(w in header for w in ("application", "real data", "case study",
                                  "data analysis", "empirical example",
                                  "effect of")):
        return "application"
    if any(w in header for w in ("conclusion", "discussion", "concluding",
                                  "summary", "remark")):
        return "discussion"
    if any(w in header for w in ("proof", "appendix", "lemma", "supplement")):
        return "proof"
    if any(w in header for w in ("related work", "literature", "review")):
        return "related_work"

    return "body"


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _generate_summaries(sections: list[dict], api_key: str) -> list[str]:
    """Generate one-sentence summaries for sections via LLM."""
    from openai import OpenAI

    if not sections:
        return []

    section_lines: list[str] = []
    for i, sec in enumerate(sections, 1):
        preview = sec["content"][:500]
        section_lines.append(
            f"[{i}] ({sec['section_type']}, {sec['char_count']} chars): {preview}..."
        )

    prompt = (
        "For each document section below, write ONE sentence summarizing its key technical content.\n"
        "Focus on what specific methods, theorems, algorithms, or results this section describes.\n"
        "Return ONLY a JSON array of strings, one summary per section, in the same order.\n\n"
        "Sections:\n" + "\n".join(section_lines)
    )

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        raw = _strip_markdown_fences(resp.choices[0].message.content or "[]")
        summaries = json.loads(raw)
        if isinstance(summaries, list) and len(summaries) == len(sections):
            return [str(s)[:500] for s in summaries]
    except Exception:
        logger.exception("Summary generation failed")

    # Fallback
    return [f"{sec['section_type']} section ({sec['char_count']} chars)" for sec in sections]


def backfill_paper(db, paper: Paper, pdf_path: Path, api_key: str, dry_run: bool) -> int:
    """Backfill sections for a single paper. Returns number of sections created."""

    # Read PDF
    pdf_bytes = pdf_path.read_bytes()
    text = _extract_text_from_pdf(pdf_bytes)
    logger.info("  Extracted %d chars from %s", len(text), pdf_path.name)

    # Split into section chunks
    chunks = _split_into_sections(text)
    logger.info("  Split into %d chunks", len(chunks))

    # Build section metadata
    raw_sections: list[dict] = []
    for idx, chunk in enumerate(chunks):
        section_type = _detect_section_type(chunk)
        raw_sections.append({
            "section_type": section_type,
            "section_index": idx,
            "content": chunk,
            "char_count": len(chunk),
        })

    if dry_run:
        for sec in raw_sections:
            logger.info("    [%d] %s (%d chars)", sec["section_index"],
                       sec["section_type"], sec["char_count"])
        return len(raw_sections)

    # Generate summaries
    summaries = _generate_summaries(raw_sections, api_key)

    # Save paper_sections
    for sec, summary in zip(raw_sections, summaries):
        ps = PaperSection(
            paper_id=paper.id,
            section_type=sec["section_type"],
            section_index=sec["section_index"],
            summary=summary,
            content=sec["content"],
            char_count=sec["char_count"],
        )
        db.add(ps)

    # Store raw file data on paper
    paper.file_data = pdf_bytes
    paper.file_content_type = "application/pdf"
    paper.file_size = len(pdf_bytes)

    db.commit()
    return len(raw_sections)


def main():
    parser = argparse.ArgumentParser(description="Backfill paper_sections for existing papers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--paper-id", type=int, help="Backfill only this paper ID")
    args = parser.parse_args()

    api_key = settings.OPENAI_API_KEY
    if not api_key and not args.dry_run:
        logger.error("OPENAI_API_KEY is required for summary generation")
        sys.exit(1)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Find papers that need backfilling (no sections yet)
        query = db.query(Paper)
        if args.paper_id:
            query = query.filter(Paper.id == args.paper_id)
        papers = query.all()

        papers_to_process = []
        for paper in papers:
            has_sections = db.query(PaperSection).filter(PaperSection.paper_id == paper.id).first() is not None
            if has_sections:
                logger.info("SKIP paper %d (%s) — already has sections", paper.id, paper.title[:60])
                continue

            # Find PDF file
            if paper.id in KNOWN_PDFS:
                pdf_path = PROJECT_ROOT / KNOWN_PDFS[paper.id]
            else:
                pdf_path = None

            if pdf_path and pdf_path.exists():
                papers_to_process.append((paper, pdf_path))
            else:
                logger.warning("SKIP paper %d (%s) — no PDF found", paper.id, paper.title[:60])

        if not papers_to_process:
            logger.info("No papers to backfill.")
            return

        total_sections = 0
        for paper, pdf_path in papers_to_process:
            logger.info("Processing paper %d: %s", paper.id, paper.title[:60])
            n = backfill_paper(db, paper, pdf_path, api_key, args.dry_run)
            total_sections += n
            logger.info("  → %d sections %s", n, "(dry-run)" if args.dry_run else "saved")

        logger.info("Done. %d papers, %d total sections.", len(papers_to_process), total_sections)

    finally:
        db.close()


if __name__ == "__main__":
    main()
