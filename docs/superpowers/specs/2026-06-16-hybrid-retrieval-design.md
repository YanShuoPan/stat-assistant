# Hybrid Retrieval & pgvector Migration Design

**Date:** 2026-06-16
**Status:** Draft
**Goal:** Improve retrieval quality by migrating to pgvector + PostgreSQL full-text search, implementing hybrid retrieval with RRF, and enhancing query expansion.

## Context

Current retrieval pipeline loads ALL knowledge unit embeddings into memory as Python lists, computes cosine similarity in a loop, and returns top-K results. With 1000-10000 KUs this is increasingly impractical. More importantly, the system relies solely on vector similarity, missing exact keyword matches for statistical method names (LASSO, OGA, Ridge, etc.) where keyword search outperforms embedding.

## Changes Overview

1. **pgvector migration** - embedding columns from JSON to `vector(1536)` with HNSW index
2. **PostgreSQL full-text search** - `tsvector` column on `knowledge_units` with GIN index
3. **Hybrid retrieval** - dual-path vector + keyword search, merged via RRF
4. **Query expansion** - leverage MethodSkill aliases/related_methods for keyword search
5. **Method selection soft boost** - replace hard filter with score boost
6. **Alembic** - schema migration management

## 1. Database Schema Changes

### 1.1 pgvector Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1.2 KnowledgeUnit Table Changes

**Before:**
```python
embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
```

**After:**
```python
from pgvector.sqlalchemy import Vector

embedding = mapped_column(Vector(1536), nullable=True)
search_vector = mapped_column(TSVector, nullable=True)
```

HNSW index for approximate nearest neighbor:
```sql
CREATE INDEX idx_ku_embedding_hnsw
  ON knowledge_units USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

GIN index for full-text search:
```sql
CREATE INDEX idx_ku_search_vector ON knowledge_units USING gin(search_vector);
```

### 1.3 search_vector Composition

The `search_vector` tsvector is built from:

| Field | Weight | Rationale |
|-------|--------|-----------|
| `method_name` | A (highest) | Exact method name matches are most valuable |
| `field` | A | Field matching (e.g. "causal inference") is high signal |
| `content` | B | Main content body |
| `problem_it_solves` | B | Problem descriptions help match user intent |
| `keywords` (JSON array, joined) | A | Domain-specific keywords |
| `title` | B | Unit title |

PostgreSQL tsvector weights: A > B > C > D. We use A for precise terms and B for general content.

Generation function (used in migration and on insert/update):
```python
def build_search_vector(unit) -> str:
    """Build the raw text for tsvector generation, with weight markers."""
    # Will use setweight() in SQL:
    # setweight(to_tsvector('english', method_name || field || keywords), 'A') ||
    # setweight(to_tsvector('english', content || problem || title), 'B')
```

### 1.4 MethodNode (method_taxonomy) Table Changes

Same embedding column change: JSON -> `Vector(1536)`. No tsvector needed (taxonomy nodes are not searched via full-text).

### 1.5 Alembic Setup

- `alembic.ini` at project root
- `alembic/` directory with `env.py` pointing to `apps/api/database.py` engine
- Migration 001: initial schema baseline (auto-generated from current models)
- Migration 002: pgvector extension, vector columns, tsvector column, indexes, data migration

### 1.6 Data Migration Strategy

Migration 002 does:
1. `CREATE EXTENSION IF NOT EXISTS vector`
2. Add new `embedding_vec` column as `Vector(1536)`
3. Add `search_vector` column as `TSVector`
4. Bulk-convert existing JSON embeddings: `UPDATE knowledge_units SET embedding_vec = embedding::vector WHERE embedding IS NOT NULL`
5. Drop old `embedding` JSON column, rename `embedding_vec` to `embedding`
6. Same for `method_taxonomy.embedding`
7. Build `search_vector` for all existing rows
8. Create HNSW and GIN indexes

### 1.7 SQLite / Test Strategy

pgvector and tsvector are PostgreSQL-only. Testing approach:

- **Unit tests** (conftest.py): Continue using SQLite in-memory for auth, CRUD, session tests
- **Retrieval tests**: Mock `hybrid_search()` at the function level, or mark as `@pytest.mark.requires_pg` and skip in CI without PostgreSQL
- **Development**: Standardize on PostgreSQL via existing docker-compose; remove SQLite as a dev path for retrieval features
- `models.py` will use conditional column types: `Vector(1536)` when PostgreSQL, `JSON` fallback for SQLite (test only)

## 2. Hybrid Retrieval Pipeline

### 2.1 New Function: `hybrid_search()`

Located in `packages/chat/embeddings.py`:

```python
def hybrid_search(
    db: Session,
    query_text: str,
    query_embedding: list[float],
    top_k: int = 50,
    boost_ids: set[int] | None = None,
    method_boost_ids: set[int] | None = None,
) -> list[tuple[float, dict]]:
```

### 2.2 Vector Search Path

```sql
SELECT id, embedding <=> :query_embedding AS distance
FROM knowledge_units
WHERE embedding IS NOT NULL
ORDER BY embedding <=> :query_embedding
LIMIT :top_k;
```

Returns up to `top_k` results ranked by cosine distance (lower = more similar).

### 2.3 Full-Text Search Path

```sql
SELECT id, ts_rank_cd(search_vector, query, 32) AS rank
FROM knowledge_units, plainto_tsquery('english', :query_text) AS query
WHERE search_vector @@ query
ORDER BY rank DESC
LIMIT :top_k;
```

The `32` normalization flag divides rank by itself+1 to normalize between 0 and 1.

For expanded queries with OR terms:
```sql
to_tsquery('english', 'LASSO | OGA | ridge & regression')
```

### 2.4 RRF Merge

Reciprocal Rank Fusion combines both ranked lists without requiring score normalization:

```python
RRF_K = 60  # standard constant

def rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)

# For each KU appearing in either list:
# final_score = rrf(vector_rank) + rrf(text_rank)
# If KU only appears in one list, the other term is 0
```

### 2.5 Boost Application

After RRF merge:
1. **Taxonomy boost**: If `ku.id in boost_ids`, multiply score by 1.3
2. **Method boost**: If `ku.id in method_boost_ids`, multiply score by 1.5

Both boosts can stack (max multiplier: 1.3 * 1.5 = 1.95).

### 2.6 Return Format

Returns `list[tuple[float, dict]]` — same interface as current `_multi_query_search()`, so downstream code (`_select_strategy`, `_build_knowledge_context`) needs minimal changes.

## 3. Query Expansion

### 3.1 Alias-Based Expansion

New function in `packages/chat/service.py`:

```python
def _expand_queries_with_skills(
    search_queries: list[str],
    method_skills: list[dict],
) -> tuple[list[str], list[str]]:
    """Expand search queries using MethodSkill aliases and related methods.

    Returns:
        (vector_queries, keyword_terms)
        - vector_queries: original search queries (used for embedding search)
        - keyword_terms: additional method names/aliases (used for full-text search only)
    """
```

Logic:
1. For each search query, fuzzy-match against method_skills (name, aliases, field)
2. Collect matched skills' `aliases` and `related_methods`
3. Return these as extra keyword terms for full-text search
4. Do NOT create extra embedding queries (avoids OpenAI API calls)

### 3.2 Full-Text Query Construction

Combine original query + expanded terms:

```python
# Original: "variable selection for high dimensional data"
# Expanded terms: ["OGA", "LASSO", "SCAD", "forward regression"]
# Full-text query: "variable & selection & high & dimensional | OGA | LASSO | SCAD"
```

The original query uses AND (all terms must match), expanded terms use OR (any match is a bonus).

### 3.3 Integration Point

In `_prepare_generation_context()`, before the retrieval step:

```python
# Current flow:
# queries -> _multi_query_search(queries, ...)

# New flow:
# queries -> _expand_queries_with_skills(queries, method_skills)
#         -> returns (vector_queries, keyword_terms)
#         -> hybrid_search(db, vector_queries, keyword_terms, ...)
```

## 4. Method Selection: Hard Filter -> Soft Boost

### 4.1 Current Behavior

`_filter_units_by_methods()` removes all KUs not matching selected methods. If the LLM picks the wrong method, relevant KUs are eliminated.

### 4.2 New Behavior

Replace `_filter_units_by_methods()` with `_compute_method_boost_ids()`:

```python
def _compute_method_boost_ids(
    units: list[dict],
    selected_methods: list[str],
    method_skills: list[dict],
) -> set[int]:
    """Return IDs of KUs matching selected methods (for boosting, not filtering)."""
```

Same matching logic (name match, alias match, field match), but returns a set of IDs instead of a filtered list. These IDs get a 1.5x score boost in `hybrid_search()`.

### 4.3 Fallback Guarantee

Since we no longer filter, all KUs are always candidates. The search pool is always the full set. Method selection only influences ranking, not exclusion.

## 5. Changes to service.py Flow

### 5.1 Current Flow (in `_prepare_generation_context`)

```
1. rewrite query (if follow-up)
2. classify question -> search_queries
3. select methods -> filter KUs (hard filter)
4. taxonomy boost IDs
5. _multi_query_search(queries, filtered_pool, boost_ids)
   - for each query: compute_embedding -> cosine sim against all units in memory
6. _select_strategy -> pick units -> build context -> LLM answer
```

### 5.2 New Flow

```
1. rewrite query (if follow-up) — unchanged
2. classify question -> search_queries — unchanged
3. select methods -> compute method_boost_ids (soft boost, no filter)
4. expand queries with skills -> (vector_queries, keyword_terms)
5. taxonomy boost IDs — unchanged (but use pgvector for taxonomy node matching too)
6. hybrid_search(db, vector_queries, keyword_terms, boost_ids, method_boost_ids)
   - vector path: pgvector ANN for each query
   - keyword path: tsvector search with expanded terms
   - RRF merge + taxonomy boost + method boost
7. _select_strategy -> pick units -> build context -> LLM answer — unchanged
```

### 5.3 Signature Changes

`generate_response()` and `generate_response_stream()` need a `db: Session` parameter, since `hybrid_search()` queries the database directly instead of receiving all KUs in memory.

The chat router (`apps/api/routers/chat.py`) currently loads all KU dicts and passes them as `method_context`. This changes to:
- Still load `skill_dicts` (small, needed for method selection and query expansion)
- Still load `taxonomy_nodes` (small, needed for taxonomy boosting)
- **Stop loading all KU dicts** — `hybrid_search()` queries the DB directly
- Pass `db` session to `generate_response()` instead

`_load_unit_dicts()` in chat.py is no longer called for the main retrieval path. It may still be useful for debug output, but can be removed or made lazy.

### 5.4 Taxonomy Node Matching

`_taxonomy_locate()` currently loads all taxonomy node embeddings and computes cosine similarity in Python. Since taxonomy nodes are few (typically < 100), this can stay as-is for now. If needed later, taxonomy embeddings can also move to pgvector queries.

## 6. File Change Summary

### New Files

| File | Purpose |
|------|---------|
| `alembic.ini` | Alembic configuration |
| `alembic/env.py` | Alembic environment setup |
| `alembic/versions/001_initial_schema.py` | Baseline schema |
| `alembic/versions/002_pgvector_and_tsvector.py` | pgvector + tsvector migration |

### Modified Files

| File | Changes |
|------|---------|
| `apps/api/models.py` | `embedding`: JSON -> Vector(1536); add `search_vector` TSVector column; conditional types for SQLite test compat |
| `apps/api/database.py` | No functional change (Alembic uses same engine) |
| `apps/api/main.py` | Remove `Base.metadata.create_all()` from lifespan (Alembic manages schema) |
| `apps/api/routers/chat.py` | Remove `_load_unit_dicts()` call; pass `db` to service; keep skill/taxonomy loading |
| `apps/api/routers/methods.py` | On upload: build and save `search_vector` for new KUs |
| `packages/chat/embeddings.py` | Add `hybrid_search()`; deprecate `retrieve_top_k()` and `_score_methods()` |
| `packages/chat/service.py` | Accept `db` param; use `hybrid_search()`; add `_expand_queries_with_skills()`; replace `_filter_units_by_methods()` with `_compute_method_boost_ids()` |
| `packages/chat/taxonomy.py` | Write Vector type embeddings instead of JSON lists |
| `scripts/import_parsed.py` | Build `search_vector` during import |
| `tests/conftest.py` | Keep SQLite for non-retrieval tests; add mock/skip for hybrid_search tests |
| `requirements.txt` | Add `pgvector`, `alembic` |

### Unchanged Files

| File | Reason |
|------|--------|
| `packages/chat/router.py` | Question classification unchanged |
| `packages/chat/method_skills.py` | Skill generation unchanged |
| `packages/chat/skill_loader.py` | Skill loading unchanged |
| `apps/api/schemas.py` | API schemas unaffected |
| `apps/api/routers/auth.py` | Auth unaffected |
| `apps/api/routers/taxonomy.py` | Taxonomy API reads unchanged |
| `apps/api/config.py` | No new config needed |
| `packages/chat/skills/*.yaml` | Skill definitions unchanged |

## 7. Migration & Rollback Plan

### Forward Migration

1. Ensure PostgreSQL has pgvector extension installed (`apt install postgresql-16-pgvector` or equivalent)
2. Run `alembic upgrade head`
3. Migration 002 converts all existing data in-place
4. Verify: `SELECT count(*) FROM knowledge_units WHERE embedding IS NOT NULL` should match pre-migration count

### Rollback

`alembic downgrade -1` reverses migration 002:
1. Convert vector columns back to JSON
2. Drop tsvector column and indexes
3. Drop pgvector extension

### Data Safety

- Migration 002 keeps old JSON data until vector conversion is verified
- Downgrade path preserves data by converting back to JSON
- Recommend: take a pg_dump before migrating

## 8. Performance Expectations

| Metric | Current | After Migration |
|--------|---------|-----------------|
| Query latency (retrieval) | ~100-500ms (in-memory, scales with KU count) | ~10-50ms (HNSW index, constant time) |
| Memory usage | All KU embeddings loaded per request | Minimal (DB handles storage) |
| Embedding storage | JSON text (~12KB per KU) | Binary vector (~6KB per KU) |
| Keyword matching | None | Sub-millisecond (GIN index) |
| Scalability ceiling | ~10K KUs before noticeable lag | ~1M+ KUs with HNSW |

## 9. Module Boundaries for Parallel Implementation

These work streams are independent and can be implemented in parallel:

| Work Stream | Files | Dependencies |
|-------------|-------|-------------|
| **A: Alembic + Models** | `alembic.ini`, `alembic/`, `models.py`, `main.py`, `requirements.txt` | None |
| **B: hybrid_search** | `embeddings.py` | A (needs vector column type) |
| **C: Service pipeline** | `service.py`, `chat.py` | B (needs hybrid_search function) |
| **D: Query expansion** | `service.py` (new function) | None (pure logic, can be written and tested independently) |
| **E: Ingestion updates** | `methods.py`, `import_parsed.py`, `taxonomy.py` | A (needs search_vector column) |
| **F: Tests** | `conftest.py`, new test files | A + B + C |

Parallel groups:
- **Phase 1** (parallel): A + D
- **Phase 2** (parallel, after A): B + E
- **Phase 3** (after B): C
- **Phase 4** (after all): F
