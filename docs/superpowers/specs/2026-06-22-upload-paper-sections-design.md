# Upload & Paper Section Retrieval Design

**Date:** 2026-06-22
**Status:** Draft

## Goal

Enable the chat system to reference original paper text (not just extracted knowledge units) when answering questions. This requires:

1. Storing paper section chunks and their summaries during upload
2. Storing the original file binary for future re-processing
3. Retrieving the most relevant sections during chat via LLM-based summary matching

## Scope

### In scope (this iteration)
- New `paper_sections` database table
- Paper table additions (`file_data`, `file_content_type`, `file_size`)
- Upload pipeline: generate section summaries + store sections + store raw file
- Chat pipeline: scoped section retrieval via LLM summary matching
- Alembic migration for schema changes

### Out of scope (future iterations)
- Frontend paper metadata form (title, authors, year, DOI)
- Upload progress feedback (SSE/polling)
- Bulk import web UI integration
- Taxonomy manual curation UI
- Cross-upload duplicate detection

## Data Model

### New table: `paper_sections`

```
paper_sections
├── id              PK, Integer, autoincrement
├── paper_id        FK → papers.id, NOT NULL, indexed
├── section_type    String(50), NOT NULL, indexed
│                   Values: abstract, introduction, methodology, theory,
│                           experiment, application, discussion, proof,
│                           related_work, unknown
├── section_index   Integer, NOT NULL
│                   Order within the paper (0, 1, 2...)
├── summary         String(500), NOT NULL
│                   One-sentence LLM-generated summary of this section
├── content         Text, NOT NULL
│                   Full extracted text of this section
├── char_count      Integer, NOT NULL
│                   Length of content (for token budget planning)
└── created_at      DateTime, default=utcnow
```

### Paper table additions

```
papers (existing table, add columns)
├── file_data           LargeBinary, nullable
│                       Raw PDF/file bytes
├── file_content_type   String(100), nullable
│                       MIME type (e.g. application/pdf, text/plain)
└── file_size           Integer, nullable
                        File size in bytes
```

All new Paper columns are nullable for backward compatibility with existing records.

## Upload Pipeline Changes

### Current flow

```
Files → POST /knowledge/parse → extract text → chunk → LLM extract KUs → return KUs
     → POST /knowledge/upload → save KUs with embeddings → done
```

### New flow

```
Files → POST /knowledge/parse
        ├── extract text (unchanged)
        ├── split into section chunks (reuse existing _split_into_chunks + _detect_section_type)
        ├── LLM extract KUs from each chunk (unchanged)
        ├── LLM generate one-sentence summary per section chunk (NEW)
        └── return { units: [...], sections: [...] }

     → POST /knowledge/upload
        ├── create Paper record with file_data binary (MODIFIED)
        ├── save paper_sections rows (NEW)
        ├── save KUs with embeddings (unchanged)
        └── post-processing: skills + taxonomy (unchanged)
```

### Section summary generation

During the parse step, after splitting into chunks and detecting section types, generate a summary for each section. This can be batched into a single LLM call:

**Prompt:**
```
For each section below, write ONE sentence summarizing its key content.
Focus on what specific information this section provides (methods, theorems, results, etc.).
Return a JSON array of strings, one summary per section, in the same order.

Sections:
[1] (methodology, 3200 chars): <first 500 chars of section>...
[2] (theory, 2800 chars): <first 500 chars of section>...
...
```

- Model: gpt-4o-mini
- Send first ~500 chars of each section as preview (enough for LLM to understand the content)
- One LLM call for all sections of a paper (efficient)

### Schema changes for parse response

`KnowledgeUnitParsed` response adds `sections` field:

```python
class PaperSectionParsed(BaseModel):
    section_type: str
    section_index: int
    summary: str
    content: str
    char_count: int

class KnowledgeUnitParsed(BaseModel):
    units: list[KnowledgeUnitBase] = []
    sections: list[PaperSectionParsed] = []  # NEW
```

### Schema changes for upload request

`KnowledgeUnitBulkCreate` adds `sections` field:

```python
class PaperSectionCreate(BaseModel):
    section_type: str
    section_index: int
    summary: str
    content: str
    char_count: int

class KnowledgeUnitBulkCreate(BaseModel):
    units: list[KnowledgeUnitCreate]
    paper: PaperCreate | None = None
    sections: list[PaperSectionCreate] = []  # NEW
```

### File binary storage

The parse endpoint currently reads file bytes but doesn't persist them. Changes:

1. Parse endpoint: raw file bytes are too large for JSON response, but the frontend already has the File objects in memory
2. Option A: upload endpoint accepts multipart form data with both JSON payload and file(s) — complex
3. Option B: separate file upload endpoint after paper creation — simpler

**Decision: Two-request approach (simpler)**
- `/knowledge/parse` — unchanged, accepts files, returns parsed data
- `/knowledge/upload` — unchanged, accepts JSON body with units + paper + sections
- `/papers/{paper_id}/file` — NEW endpoint, accepts file upload after paper is created
  - Frontend calls this immediately after upload succeeds
  - Stores file_data, file_content_type, file_size on the Paper record

This avoids mixing multipart + JSON in one request and keeps the existing flow intact.

### Multi-file uploads

When multiple files are uploaded together (e.g. paper.pdf + code.py):
- Sections are only generated from document files (PDF, TXT, MD), not from code files
- Code files are chunked for KU extraction as before but do not produce paper_sections
- All sections from all document files in one upload batch are linked to the same Paper record
- Typical use case: one paper PDF per upload. Multi-paper batches should be separate uploads.

## Chat Retrieval Changes

### Current flow (simplified)

```
User question
→ Step 1:   Classify question, extract search queries
→ Step 1.5: Select relevant methods
→ Step 2:   Hybrid search KUs (pgvector + tsvector + RRF)
→ Step 2.5: LLM rerank top-20 KUs → keep top-5
→ Step 3:   Build knowledge context from matched KUs
→ Step 4:   Generate response with system prompt + knowledge context
```

### New flow (adds Step 2.75)

```
User question
→ Step 1:   Classify question, extract search queries (unchanged)
→ Step 1.5: Select relevant methods (unchanged)
→ Step 2:   Hybrid search KUs (unchanged)
→ Step 2.5: LLM rerank top-20 KUs → keep top-5 (unchanged)
→ Step 2.75: Scoped section retrieval (NEW)
             ├── Collect paper_ids from matched KUs
             ├── Query paper_sections for those papers (summary + section_type only)
             ├── LLM selects 2-3 most relevant sections based on summaries
             ├── Fetch full content of selected sections
             └── Truncate each to ~3000 chars
→ Step 3:   Build knowledge context from KUs + paper sections (MODIFIED)
→ Step 4:   Generate response (unchanged, but has more context)
```

### Step 2.75 detail: LLM section selection

**Input:** user question + list of section summaries from matched papers

**Prompt:**
```
You are selecting which paper sections to include as reference material for answering a research question.

## User question
{question}

## Available sections
[1] Paper: "{paper_title}" — Section: {section_type} — {summary}
[2] Paper: "{paper_title}" — Section: {section_type} — {summary}
...

## Task
Select the 2-3 sections most likely to contain information needed to answer the question.
Return ONLY a JSON array of section numbers, e.g. [1, 3, 5].
If no section is relevant, return [].
```

- Model: gpt-4o-mini
- Temperature: 0
- Max tokens: 50
- Very cheap call: input is ~200-500 tokens (summaries are one sentence each)

**Skip condition:** If matched KUs have no `paper_id` (i.e. no linked paper), skip Step 2.75 entirely.

### Context format

Paper sections are appended after the existing knowledge units section:

```
## Knowledge Base - Matched Units
[existing KU context unchanged]

## Original Paper Sections
The following are relevant sections from matched papers, provided as additional reference.

### Paper: "OGA+HDIC+Trim" — Methodology
[section content, truncated to ~3000 chars]

### Paper: "OGA+HDIC+Trim" — Theoretical Results
[section content, truncated to ~3000 chars]
```

### Token budget

- Each section: max ~3000 chars (~750 tokens)
- Max 3 sections: ~2250 tokens for paper context
- Existing KU context: ~1000-2000 tokens
- Total knowledge context: ~3000-4000 tokens
- Well within gpt-4o-mini's 128K context window

## Files to Modify

### Backend (`apps/api/`)

| File | Change |
|------|--------|
| `models.py` | Add `PaperSection` model, add `file_data`/`file_content_type`/`file_size` to `Paper` |
| `schemas.py` | Add `PaperSectionParsed`, `PaperSectionCreate`, update `KnowledgeUnitParsed` and `KnowledgeUnitBulkCreate` |
| `routers/methods.py` | Modify parse to generate section summaries, modify upload to save sections, add `/papers/{id}/file` endpoint |

### Chat (`packages/chat/`)

| File | Change |
|------|--------|
| `service.py` | Add Step 2.75 (`_select_paper_sections`), modify `_build_knowledge_context` or add `_build_section_context`, update `_prepare_generation_context` |

### Frontend (`apps/web/src/`)

| File | Change |
|------|--------|
| `app/upload/page.tsx` | Send sections in upload request body, call file upload endpoint after save |

### Database migration

| File | Change |
|------|--------|
| New migration | Add `paper_sections` table, add columns to `papers` table |

## Testing Strategy

1. **Unit test: section summary generation** — Mock LLM, verify sections are parsed and summaries returned
2. **Unit test: upload with sections** — Verify paper_sections rows are created alongside KUs
3. **Unit test: section selection** — Mock LLM section selector, verify correct sections are picked
4. **Integration test: end-to-end** — Upload a paper with sections → ask a question → verify sections appear in LLM context (via debug output)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Section summary generation adds latency to parse | Batch all sections into one LLM call (~500 tokens input) |
| Section selection adds latency to chat | Very small LLM call (~200 tokens), and only triggered when papers are matched |
| Large PDFs produce too many sections | Cap at 15 sections per paper; merge very short sections |
| file_data bloats the database | PostgreSQL handles BYTEA well for typical paper sizes (1-10MB); can migrate to S3 later if needed |
| Backward compatibility | All new columns/tables are additive; existing papers without sections work fine (Step 2.75 simply skips) |
