import io
import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from openai import OpenAI

from auth import get_current_user, require_role
from config import settings
from database import get_db
from models import KnowledgeUnit, MethodSkill, Paper, PaperSection, User
from schemas import (
    KnowledgeUnitBulkCreate,
    KnowledgeUnitParsed,
    KnowledgeUnitResponse,
    MethodSkillResponse,
    PaperListResponse,
    PaperResponse,
    PaperSectionParsed,
)
from chat.embeddings import compute_embedding, compute_embeddings_batch, unit_to_embedding_text
from chat.method_skills import generate_all_method_skills


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB per file

# ---------------------------------------------------------------------------
# System prompt for knowledge extraction
# ---------------------------------------------------------------------------

PARSE_PROMPT = """
Ensure to include specific convergence rates and conditions for methods when applicable.\
You are a knowledge extraction engine for a long-term research Q&A system.

Your job is NOT to produce a generic summary.
Your job is to transform papers, code, and technical documents into reusable knowledge that can answer real user questions later.

The stored knowledge must help answer questions such as:
- What problem does this method solve?
- When should this method be used?
- What are the assumptions, inputs, outputs, and limitations?
- How does the code work in practice?
- What do the parameters mean?
- What evidence supports the claims?
- How is this method different from related methods?
- What can fail, and under what conditions?

Do NOT store vague, generic, motivational, or high-level filler statements.
Do NOT store canned summary sentences unless they are truly useful for future Q&A.
Do NOT optimize for readability alone.
Optimize for future retrievability, factual precision, and usefulness in answering realistic technical questions.

For each input document, extract and store knowledge in structured form using the following rules:

1. Preserve concrete technical content:
- definitions
- assumptions
- formulas or algorithmic steps
- variable meanings
- parameter roles
- input/output behavior
- edge cases
- experimental findings
- limitations
- implementation details
- code dependencies
- failure modes

2. Break knowledge into small self-contained units.
Each unit should be understandable without needing the whole document.

3. Prefer Q&A-useful statements over narrative summary.
Bad example:
- "This paper proposes an interesting and effective method."
Good example:
- "The method improves feature recovery under nonlinear indicator-type signals by expanding candidate paths and then trimming redundant ones."

4. When processing code, extract:
- what the function/class/script does
- required inputs and expected data format
- important parameters and defaults
- returned outputs
- preprocessing assumptions
- model training logic
- decision logic
- common failure points
- any mismatch between code behavior and paper description

5. When processing papers, extract:
- research question
- method core idea
- mathematical assumptions
- identification conditions / theoretical guarantees
- comparison baselines
- experimental settings
- what conclusions are actually supported by results
- what is NOT supported

6. Explicitly capture uncertainty.
If the source is ambiguous, incomplete, or speculative, mark it clearly instead of inventing certainty.

7. The field "reusable_for_questions" must contain realistic future questions this knowledge can help answer.

8. Prefer granularity over merging.
If two passages discuss different aspects of the same concept (e.g., definition vs. limitation vs. convergence rate), keep them as SEPARATE units. Only merge if the content is truly identical. Aim for 8-20 units per document section.

9. Every stored unit must pass this test:
"Would this help answer a real technical user question later?"
If no, do not store it.

Return a JSON array of knowledge units. Each unit must have exactly these fields:

CRITICAL field rules:
- "source_type" must be one of: paper, code, docstring, note.
- "title" is the METHOD name (e.g., "OGA", "DML", "DeepKriging"), NOT the paper title.
- "section" is the specific section of the paper this knowledge comes from (e.g., "Section 2: Methodology", "Abstract"), NOT the paper title.
Double-check your field assignment before returning.

{
  "source_type": "paper | code | docstring | note",
  "title": "name of the method or document",
  "section": "which section or part this came from",
  "knowledge_type": "definition | assumption | algorithm | parameter | result | limitation | implementation | failure_mode | comparison",
  "topic_tags": ["tag1", "tag2"],
  "question_intent_tags": ["what_is_it", "when_to_use", "how_it_works", "why_failed", "compare_methods"],
  "content": "self-contained knowledge statement",
  "evidence_span": "original passage, formula, function name, or line reference",
  "dependencies": ["related symbols, functions, prior concepts"],
  "limitations": "what this statement does not imply",
  "confidence": "high | medium | low",
  "reusable_for_questions": [
    "Example question 1?",
    "Example question 2?"
  ],
  "method_name": "canonical name of the method (e.g., OGA, DML, Deep Kriging)",
  "field": "statistical sub-field (e.g., high-dimensional regression, causal inference, spatial statistics, Bayesian optimization)",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "problem_it_solves": "one sentence: what practical problem does this method address?",
  "model_assumption": "key assumptions the method requires (e.g., sparsity, linearity, independence)",
  "input_format": "what data/inputs the method expects (e.g., design matrix X, response Y, spatial coordinates)",
  "output_format": "what the method produces (e.g., selected variables, treatment effect estimate, predicted surface)",
  "typical_questions": [
    "A realistic question a researcher would ask about this method"
  ],
  "related_methods": ["method1", "method2"]
}

Rules for the new fields:
- method_name: ALWAYS use the PRIMARY method that this paper/code proposes (the author's contribution).
  For example, if the paper proposes DML and mentions Neyman orthogonality as a building block,
  ALL knowledge units should have method_name = "DML", not "Neyman Orthogonality".
  Sub-concepts, building blocks, and related methods go in "related_methods", not in "method_name".
  Use the most commonly recognized short name or abbreviation.
- field: Map to a recognized statistical sub-field, not a vague category.
- keywords: Include both formal terms and informal synonyms researchers might search for.
- problem_it_solves: Be specific. Not "solves regression" but "selects relevant variables when p >> n".
- model_assumption: List concrete mathematical or data assumptions, not vague statements.
- input_format / output_format: Be specific about data structures, dimensions, types.
- typical_questions: Write 1-3 questions a real researcher would ask. Use casual or applied language.
- related_methods: Name specific alternative or complementary methods.

10. Target quantity: Extract 8-25 knowledge units per document. If you produce fewer than 8 for a full paper, you are likely being too aggressive in merging.

11. When processing code files, generate AT LEAST one unit per function or class with knowledge_type "implementation". Include the function signature in evidence_span.

Return ONLY a valid JSON array, no markdown fences, no extra text.
"""


CODE_EXTS = {"py", "r", "rmd", "sas", "do", "jl"}
TEXT_EXTS = {"txt", "md", "csv"} | CODE_EXTS


def _get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _extract_text(filename: str, content: bytes) -> str:
    ext = _get_ext(filename)
    if ext == "pdf":
        try:
            from pypdf import PdfReader
            return "\n\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(content)).pages)
        except ImportError:
            raise HTTPException(status_code=400, detail="PDF parsing requires pypdf")
    return content.decode("utf-8", errors="replace")


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()




# ---------------------------------------------------------------------------
# Section-aware extraction focus
# ---------------------------------------------------------------------------

SECTION_FOCUS = {
    "abstract": (
        "[SECTION: Abstract] Focus on: method name, core contribution, "
        "problem statement, key result claims. Skip generic motivation."
    ),
    "introduction": (
        "[SECTION: Introduction] Focus on: what problem this method solves, "
        "advantages over existing approaches, motivation for the method. "
        "Skip general background and literature survey filler."
    ),
    "methodology": (
        "[SECTION: Methodology — this is the CORE section] Focus on: "
        "algorithm steps, mathematical formulation, parameters and their roles, "
        "input/output specification, key definitions. Extract with HIGH granularity."
    ),
    "theory": (
        "[SECTION: Theoretical Results] Focus on: convergence rates, consistency "
        "conditions, oracle properties, theoretical guarantees, required assumptions "
        "for each theorem. Include specific rates (e.g., O(n^{-1/2}))."
    ),
    "experiment": (
        "[SECTION: Experiments/Simulations] Focus on: experimental setup, "
        "data generation, baselines compared, key findings, when the method "
        "succeeds vs fails. Extract concrete numerical results."
    ),
    "application": (
        "[SECTION: Real Data Application] Focus on: data description, "
        "preprocessing steps, practical implementation details, real-world "
        "performance, practical recommendations."
    ),
    "discussion": (
        "[SECTION: Discussion/Conclusion] Focus on: limitations, open problems, "
        "practical recommendations, when NOT to use this method."
    ),
    "proof": (
        "[SECTION: Proofs/Appendix] Focus on: key proof techniques, "
        "critical lemmas, conditions that are actually used. "
        "Skip routine algebraic manipulations. Extract FEWER units here."
    ),
    "related_work": (
        "[SECTION: Related Work] Focus on: how this method differs from "
        "alternatives, specific comparisons. Skip generic descriptions of other methods."
    ),
}


def _detect_section_type(chunk_text: str) -> str:
    """Detect the section type from chunk text header/first lines."""
    # Look at first 300 chars for section indicators
    header = chunk_text[:300].lower()

    if "abstract" in header:
        return "abstract"
    if any(w in header for w in ("introduction", "motivation", "background")):
        return "introduction"
    if any(w in header for w in ("methodology", "method", "algorithm", "model",
                                  "framework", "proposed", "approach")):
        # Distinguish from "related work mentions of methods"
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

    return ""  # unknown section, no special focus



def _split_into_chunks(text: str, tag: str, filename: str, max_chars: int = 12000) -> list[str]:
    """Split a document into chunks at section boundaries.

    Code files are returned as a single chunk. Papers are split on section
    headers so each chunk stays semantically coherent.
    """
    prefix = "=== [" + tag + "] " + filename
    if tag == "CODE FILE" or len(text) <= max_chars:
        return [prefix + ' ===' + chr(10) + text]

    # Split on common section-header patterns
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

    labeled: list[str] = []
    for i, chunk in enumerate(chunks):
        header = prefix + " (part " + str(i + 1) + "/" + str(len(chunks)) + ") ===\n"
        labeled.append(header + chunk)
    return labeled


def _deduplicate_units(units: list[dict]) -> list[dict]:
    """Remove near-duplicate knowledge units by (method_name, knowledge_type, content prefix)."""
    seen: set[tuple[str, str, str]] = set()
    result: list[dict] = []
    for u in units:
        key = (
            (u.get("method_name") or "").lower(),
            (u.get("knowledge_type") or "").lower(),
            (u.get("content") or "")[:100].lower(),
        )
        if key not in seen:
            seen.add(key)
            result.append(u)
    return result

def _parse_one_chunk(client, chunk_text: str) -> list[dict]:
    """Send a single chunk to the LLM and return parsed knowledge units.

    Detects section type and prepends section-specific extraction guidance.
    """
    section_type = _detect_section_type(chunk_text)
    focus = SECTION_FOCUS.get(section_type, "")
    if focus:
        chunk_text = focus + chr(10) + chr(10) + chunk_text

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PARSE_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
        temperature=0.4,
        max_tokens=12000,
    )
    raw = _strip_markdown_fences(resp.choices[0].message.content or "[]")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    return parsed if isinstance(parsed, list) else []


def _strip_chunk_header(chunk_text: str) -> str:
    """Remove the ``=== [...] ===`` prefix that _split_into_chunks adds."""
    if chunk_text.startswith("==="):
        newline_pos = chunk_text.find("\n")
        if newline_pos != -1:
            return chunk_text[newline_pos + 1:]
    return chunk_text


def _generate_section_summaries(client, sections: list[dict]) -> list[str]:
    """Generate one-sentence summaries for each section via a single LLM call.

    *sections* is a list of dicts with keys ``section_type``, ``char_count``,
    and ``content``.  Returns a list of summary strings, one per section.
    """
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
        logger.exception("Section summary generation failed")

    # Fallback: return empty summaries so callers still get valid data
    return [f"{sec['section_type']} section ({sec['char_count']} chars)" for sec in sections]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/parse", response_model=KnowledgeUnitParsed)
async def parse_files(
    files: list[UploadFile] = File(...),
    current_user: User = Depends(require_role("admin", "researcher")),
):
    """Upload one or more files -> LLM extracts knowledge units per chunk."""
    chunks: list[str] = []
    total_text = 0
    for file in files:
        raw_content = await file.read()
        if len(raw_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' exceeds {MAX_UPLOAD_SIZE // (1024*1024)}MB limit",
            )
        fname = file.filename or "file.txt"
        text = _extract_text(fname, raw_content)
        if not text.strip():
            continue
        total_text += len(text)
        tag = "CODE FILE" if _get_ext(fname) in CODE_EXTS else "DOCUMENT"
        chunks.extend(_split_into_chunks(text, tag, fname))

    if total_text < 20:
        raise HTTPException(status_code=400, detail="Files contain too little text")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    all_units: list[dict] = []
    for chunk in chunks:
        all_units.extend(_parse_one_chunk(client, chunk))

    all_units = _deduplicate_units(all_units)

    # --- Build paper sections from document (non-code) chunks ---
    doc_chunks = [c for c in chunks if not c.startswith("=== [CODE FILE]")]
    raw_sections: list[dict] = []
    for idx, chunk in enumerate(doc_chunks):
        content = _strip_chunk_header(chunk)
        section_type = _detect_section_type(content)
        raw_sections.append({
            "section_type": section_type or "body",
            "section_index": idx,
            "content": content,
            "char_count": len(content),
        })

    # Generate summaries in one LLM call
    summaries = _generate_section_summaries(client, raw_sections) if raw_sections else []
    section_list: list[PaperSectionParsed] = []
    for sec, summary in zip(raw_sections, summaries):
        section_list.append(PaperSectionParsed(
            section_type=sec["section_type"],
            section_index=sec["section_index"],
            summary=summary,
            content=sec["content"],
            char_count=sec["char_count"],
        ))

    return KnowledgeUnitParsed(units=all_units, sections=section_list)


@router.post("/upload", response_model=list[KnowledgeUnitResponse], status_code=201)
def upload_knowledge(
    body: KnowledgeUnitBulkCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin", "researcher")),
    skip_embeddings: bool = False,
    skip_postprocessing: bool = False,
):
    """Save multiple knowledge units to the database with embeddings.

    If ``body.paper`` is provided, a Paper record is created first and
    every unit in the batch is linked to it via ``paper_id``.

    Query params:
    - skip_embeddings: skip embedding computation (faster bulk import)
    - skip_postprocessing: skip skill regeneration and taxonomy classification
    """
    saved = []
    try:
        # Create Paper record if paper metadata is provided
        paper_id: int | None = None
        if body.paper:
            paper = Paper(**body.paper.model_dump())
            db.add(paper)
            db.flush()
            paper_id = paper.id
            logger.info("Created Paper id=%d: %s", paper.id, paper.title)

        # Save paper sections
        if body.sections and paper_id is not None:
            for sec_data in body.sections:
                section = PaperSection(
                    paper_id=paper_id,
                    **sec_data.model_dump(),
                )
                db.add(section)
            db.flush()

        for unit_data in body.units:
            d = unit_data.model_dump()
            # Link to the newly created paper (overrides per-unit paper_id)
            if paper_id is not None:
                d["paper_id"] = paper_id
            embedding = None
            if not skip_embeddings:
                emb_text = unit_to_embedding_text(d)
                embedding = compute_embedding(emb_text, settings.OPENAI_API_KEY)
            unit = KnowledgeUnit(**d, uploaded_by=current_user.id, embedding=embedding)
            db.add(unit)
            db.flush()
            saved.append(unit)
        db.commit()
        for u in saved:
            db.refresh(u)
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to save knowledge units")
        raise HTTPException(status_code=500, detail="Failed to save knowledge units")

    if not skip_postprocessing:
        # Auto-regenerate method skills after new KUs are saved
        try:
            _regenerate_skills(db)
        except Exception:
            logger.exception("Skill regeneration failed after upload")

        # Auto-classify into taxonomy
        try:
            from chat.taxonomy import classify_units_to_taxonomy
            classify_units_to_taxonomy(db, saved, settings.OPENAI_API_KEY)
        except Exception:
            logger.exception("Taxonomy classification failed after upload")

    return saved


@router.post("/papers/{paper_id}/file", status_code=200)
async def upload_paper_file(
    paper_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin", "researcher")),
):
    """Upload the raw file binary to an existing Paper record."""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    paper.file_data = raw
    paper.file_content_type = file.content_type or "application/octet-stream"
    paper.file_size = len(raw)
    db.commit()
    return {"paper_id": paper_id, "file_size": len(raw)}


@router.post("/backfill-embeddings")
def backfill_embeddings(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
    batch_size: int = 100,
):
    """Generate embeddings for knowledge units that have NULL embeddings.

    Processes *batch_size* units per call. Returns stats so the caller
    can loop until ``remaining == 0``.
    """
    missing = (
        db.query(KnowledgeUnit)
        .filter(KnowledgeUnit.embedding.is_(None))
        .limit(batch_size)
        .all()
    )
    if not missing:
        total = db.query(KnowledgeUnit).count()
        return {"processed": 0, "remaining": 0, "total": total}

    # Build embedding texts
    texts: list[str] = []
    for ku in missing:
        d = {
            "title": ku.title,
            "knowledge_type": ku.knowledge_type,
            "content": ku.content,
            "topic_tags": ku.topic_tags or [],
            "question_intent_tags": ku.question_intent_tags or [],
            "reusable_for_questions": ku.reusable_for_questions or [],
            "method_name": ku.method_name,
            "field": ku.field,
            "keywords": ku.keywords or [],
            "problem_it_solves": ku.problem_it_solves,
            "related_methods": ku.related_methods or [],
        }
        texts.append(unit_to_embedding_text(d))

    embeddings = compute_embeddings_batch(texts, settings.OPENAI_API_KEY)

    updated = 0
    for ku, emb in zip(missing, embeddings):
        if emb:
            ku.embedding = emb
            updated += 1
    db.commit()

    remaining = (
        db.query(KnowledgeUnit)
        .filter(KnowledgeUnit.embedding.is_(None))
        .count()
    )
    total = db.query(KnowledgeUnit).count()
    return {
        "processed": updated,
        "remaining": remaining,
        "total": total,
    }


@router.get("", response_model=list[KnowledgeUnitResponse])
def list_knowledge(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(KnowledgeUnit).order_by(KnowledgeUnit.created_at.desc()).all()


# ---------------------------------------------------------------------------
# Method Skills
# ---------------------------------------------------------------------------

UNIT_FIELDS_FOR_SKILL = (
    "title", "source_type", "section", "knowledge_type",
    "content", "evidence_span", "limitations", "confidence",
    "method_name", "field", "problem_it_solves", "model_assumption",
    "input_format", "output_format",
)
LIST_FIELDS_FOR_SKILL = (
    "topic_tags", "question_intent_tags", "dependencies",
    "reusable_for_questions", "keywords", "typical_questions",
    "related_methods",
)


def _ku_to_dict(u) -> dict:
    """Convert a KnowledgeUnit ORM object to a plain dict."""
    d = {c: getattr(u, c) for c in UNIT_FIELDS_FOR_SKILL}
    d.update({c: getattr(u, c) or [] for c in LIST_FIELDS_FOR_SKILL})
    return d


def _regenerate_skills(db: Session) -> list:
    """Regenerate all method skill cards from current knowledge units.

    Groups KUs by method, calls LLM to summarize, upserts into method_skills.
    Returns list of saved MethodSkill ORM objects.
    """
    units = db.query(KnowledgeUnit).all()
    if not units:
        return []

    unit_dicts = [_ku_to_dict(u) for u in units]
    skills = generate_all_method_skills(unit_dicts, settings.OPENAI_API_KEY)

    saved = []
    for skill_data in skills:
        method_name = skill_data.get("method", "unknown")
        existing = db.query(MethodSkill).filter(MethodSkill.method == method_name).first()
        if existing:
            for key in ("field", "aliases", "purpose", "summary", "pipeline_steps",
                        "assumptions", "typical_questions", "related_methods"):
                if key in skill_data:
                    setattr(existing, key, skill_data[key])
            db.flush()
            saved.append(existing)
        else:
            ms = MethodSkill(
                method=method_name,
                field=skill_data.get("field", ""),
                aliases=skill_data.get("aliases", []),
                purpose=skill_data.get("purpose", ""),
                summary=skill_data.get("summary", ""),
                pipeline_steps=skill_data.get("pipeline_steps", []),
                assumptions=skill_data.get("assumptions", []),
                typical_questions=skill_data.get("typical_questions", []),
                related_methods=skill_data.get("related_methods", []),
            )
            db.add(ms)
            db.flush()
            saved.append(ms)

    db.commit()
    for s in saved:
        db.refresh(s)
    return saved


@router.post("/generate-skills", response_model=list[MethodSkillResponse])
def generate_skills(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Generate method skill cards from all knowledge units in the DB."""
    result = _regenerate_skills(db)
    if not result:
        raise HTTPException(status_code=400, detail="No knowledge units in database")
    return result


@router.get("/skills", response_model=list[MethodSkillResponse])
def list_skills(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all method skill cards."""
    return db.query(MethodSkill).order_by(MethodSkill.method).all()


# --- Single unit endpoints (must be AFTER static routes to avoid path conflict) ---

def _get_unit_or_404(unit_id: int, db: Session) -> KnowledgeUnit:
    unit = db.query(KnowledgeUnit).filter(KnowledgeUnit.id == unit_id).first()
    if not unit:
        raise HTTPException(status_code=404, detail="Knowledge unit not found")
    return unit


@router.get("/{unit_id}", response_model=KnowledgeUnitResponse)
def get_knowledge_unit(
    unit_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_unit_or_404(unit_id, db)


@router.delete("/{unit_id}", status_code=204)
def delete_knowledge_unit(
    unit_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    unit = _get_unit_or_404(unit_id, db)
    if current_user.role != "admin" and unit.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="You can only delete your own units")
    db.delete(unit)
    db.commit()

# ---------------------------------------------------------------------------
# Paper management
# ---------------------------------------------------------------------------

@router.get("/papers", response_model=list[PaperListResponse])
def list_papers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all uploaded papers with their KU counts."""
    from sqlalchemy import func
    rows = (
        db.query(Paper, func.count(KnowledgeUnit.id).label("ku_count"))
        .outerjoin(KnowledgeUnit, KnowledgeUnit.paper_id == Paper.id)
        .group_by(Paper.id)
        .order_by(Paper.created_at.desc())
        .all()
    )
    result = []
    for paper, ku_count in rows:
        d = PaperListResponse.model_validate(paper)
        d.ku_count = ku_count
        result.append(d)
    return result


@router.delete("/papers/{paper_id}", status_code=204)
def delete_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Delete a paper and all its associated KUs (admin only)."""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    db.delete(paper)
    db.commit()
