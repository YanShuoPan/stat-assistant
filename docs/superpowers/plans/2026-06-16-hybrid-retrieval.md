# Hybrid Retrieval & pgvector Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace in-memory brute-force cosine similarity with pgvector ANN search + PostgreSQL full-text search, merged via RRF, to improve retrieval quality and scalability.

**Architecture:** Dual-path retrieval — pgvector HNSW for semantic search, tsvector/GIN for keyword search — merged with Reciprocal Rank Fusion. Method selection changes from hard filter to soft boost. Query expansion leverages MethodSkill aliases for full-text search.

**Tech Stack:** PostgreSQL 16 + pgvector extension, SQLAlchemy + pgvector-python, Alembic migrations, existing OpenAI embeddings (text-embedding-3-small, 1536 dimensions).

**Design spec:** `docs/superpowers/specs/2026-06-16-hybrid-retrieval-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|----------------|
| `alembic.ini` | Alembic configuration pointing to apps/api |
| `alembic/env.py` | Alembic environment — imports engine from database.py |
| `alembic/script.py.mako` | Migration template |
| `alembic/versions/001_baseline.py` | Baseline schema snapshot |
| `alembic/versions/002_pgvector_tsvector.py` | pgvector + tsvector migration |

### Modified Files
| File | Changes |
|------|---------|
| `docker-compose.yml` | Switch to pgvector-enabled image |
| `apps/api/requirements.txt` | Add `pgvector`, `alembic` |
| `apps/api/models.py` | Conditional Vector/JSON columns, add search_vector |
| `apps/api/main.py` | Remove `create_all`, add migration check |
| `packages/chat/embeddings.py` | Add `hybrid_search()`, deprecate old functions |
| `packages/chat/service.py` | Accept `db`, use hybrid_search, query expansion, soft boost |
| `apps/api/routers/chat.py` | Pass `db` to service, remove `_load_unit_dicts()` call |
| `tests/conftest.py` | Add hybrid_search mock for SQLite tests |

---

## Task 1: Infrastructure Setup

**Files:**
- Modify: `docker-compose.yml`
- Modify: `apps/api/requirements.txt`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`

- [ ] **Step 1: Update docker-compose.yml for pgvector**

Replace the postgres image with the pgvector-enabled one:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: stat_assistant_db
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: stat_assistant
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

- [ ] **Step 2: Add dependencies to requirements.txt**

Append to `apps/api/requirements.txt`:

```
pgvector>=0.3.0
alembic>=1.13.0
```

- [ ] **Step 3: Install dependencies**

Run: `pip install pgvector alembic`

- [ ] **Step 4: Create alembic.ini at project root**

```ini
[alembic]
script_location = alembic
prepend_sys_path = apps/api:packages

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

Note: `sqlalchemy.url` is NOT set here — we read it from config at runtime in env.py.

- [ ] **Step 5: Create alembic/env.py**

```python
import sys
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make apps/api and packages importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from config import settings
from database import Base
import models  # noqa: F401 — register all models with Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        {"sqlalchemy.url": settings.DATABASE_URL},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 6: Create alembic/script.py.mako**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 7: Create alembic/versions/ directory**

Run: `mkdir -p alembic/versions`

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml apps/api/requirements.txt alembic.ini alembic/
git commit -m "feat: add pgvector docker image, alembic setup, and new dependencies"
```

---

## Task 2: Models & Database Migrations

**Files:**
- Modify: `apps/api/models.py`
- Modify: `apps/api/main.py`
- Create: `alembic/versions/001_baseline.py`
- Create: `alembic/versions/002_pgvector_tsvector.py`

- [ ] **Step 1: Update models.py with conditional column types**

Replace the imports and add conditional type logic at the top of `apps/api/models.py`. The embedding columns use `Vector(1536)` on PostgreSQL and `JSON` on SQLite (tests). A `search_vector` tsvector column is added for full-text search.

Full replacement of `apps/api/models.py`:

```python
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from database import Base
from config import settings

_IS_PG = settings.DATABASE_URL.startswith("postgresql")

if _IS_PG:
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.dialects.postgresql import TSVECTOR

    _EmbeddingType = Vector(1536)
    _SearchVectorType = TSVECTOR
else:
    _EmbeddingType = JSON
    _SearchVectorType = Text


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="viewer")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    domain: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    cluster: Mapped[str | None] = mapped_column(String(100), nullable=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class KnowledgeUnit(Base):
    __tablename__ = "knowledge_units"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    knowledge_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    topic_tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    question_intent_tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_span: Mapped[str | None] = mapped_column(Text, nullable=True)
    dependencies: Mapped[list | None] = mapped_column(JSON, nullable=True)
    limitations: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")
    reusable_for_questions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    method_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    field: Mapped[str | None] = mapped_column(String(255), nullable=True)
    keywords: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    problem_it_solves: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_assumption: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_format: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_format: Mapped[str | None] = mapped_column(Text, nullable=True)
    typical_questions: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    related_methods: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    embedding = mapped_column(_EmbeddingType, nullable=True)
    search_vector = mapped_column(_SearchVectorType, nullable=True)
    paper_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("papers.id"), nullable=True)
    uploaded_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class MethodSkill(Base):
    __tablename__ = "method_skills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    method: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    field: Mapped[str] = mapped_column(String(255), nullable=False)
    aliases: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_steps: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    assumptions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    typical_questions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    related_methods: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    method_node_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("method_taxonomy.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class MethodNode(Base):
    __tablename__ = "method_taxonomy"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    node_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("method_taxonomy.id"), nullable=True)
    aliases: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    auto_generated: Mapped[bool] = mapped_column(default=True)
    embedding = mapped_column(_EmbeddingType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class KnowledgeUnitNode(Base):
    __tablename__ = "knowledge_unit_nodes"

    knowledge_unit_id: Mapped[int] = mapped_column(Integer, ForeignKey("knowledge_units.id"), primary_key=True)
    method_node_id: Mapped[int] = mapped_column(Integer, ForeignKey("method_taxonomy.id"), primary_key=True)
```

- [ ] **Step 2: Update main.py — remove create_all**

In `apps/api/main.py`, replace the lifespan function. Remove `Base.metadata.create_all()` since Alembic manages schema:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema managed by Alembic — run: alembic upgrade head
    yield
```

Remove the `from database import Base, engine` import line (no longer needed). Keep `from database import get_db` if it was used elsewhere, but it's not imported in main.py currently — so just remove the Base/engine import.

Updated imports in `apps/api/main.py`:

```python
import logging
import os
import sys
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.auth import router as auth_router
from routers.chat import router as chat_router
from routers.methods import router as methods_router
from routers.taxonomy import router as taxonomy_router

import models  # noqa: F401
from config import settings
```

- [ ] **Step 3: Create migration 001 — baseline**

Create `alembic/versions/001_baseline.py`. This marks the existing schema as the baseline (empty migration since tables already exist):

```python
"""baseline schema

Revision ID: 001
Revises: None
Create Date: 2026-06-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Baseline — existing tables already in DB.
    # This migration exists so Alembic can track subsequent changes.
    pass


def downgrade() -> None:
    pass
```

After creating, stamp the DB so Alembic knows migration 001 is "already applied":

Run: `alembic stamp 001`

- [ ] **Step 4: Create migration 002 — pgvector + tsvector**

Create `alembic/versions/002_pgvector_tsvector.py`:

```python
"""add pgvector embeddings and tsvector search

Revision ID: 002
Revises: 001
Create Date: 2026-06-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Add new vector columns alongside existing JSON columns
    op.add_column("knowledge_units", sa.Column("embedding_vec", sa.Text(), nullable=True))
    op.add_column("method_taxonomy", sa.Column("embedding_vec", sa.Text(), nullable=True))

    # 3. Convert existing JSON embeddings to vector format
    #    JSON text like [0.1, 0.2, ...] is compatible with pgvector text input
    op.execute("""
        UPDATE knowledge_units
        SET embedding_vec = embedding::text
        WHERE embedding IS NOT NULL AND embedding::text != 'null'
    """)
    op.execute("""
        UPDATE method_taxonomy
        SET embedding_vec = embedding::text
        WHERE embedding IS NOT NULL AND embedding::text != 'null'
    """)

    # 4. Drop old JSON columns, add proper vector columns
    op.drop_column("knowledge_units", "embedding")
    op.drop_column("method_taxonomy", "embedding")

    op.execute("ALTER TABLE knowledge_units ADD COLUMN embedding vector(1536)")
    op.execute("ALTER TABLE method_taxonomy ADD COLUMN embedding vector(1536)")

    # 5. Copy data from text temp columns to vector columns
    op.execute("""
        UPDATE knowledge_units
        SET embedding = embedding_vec::vector(1536)
        WHERE embedding_vec IS NOT NULL
    """)
    op.execute("""
        UPDATE method_taxonomy
        SET embedding = embedding_vec::vector(1536)
        WHERE embedding_vec IS NOT NULL
    """)

    # 6. Drop temp columns
    op.drop_column("knowledge_units", "embedding_vec")
    op.drop_column("method_taxonomy", "embedding_vec")

    # 7. Create HNSW index for knowledge_units
    op.execute("""
        CREATE INDEX idx_ku_embedding_hnsw
        ON knowledge_units USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # 8. Add tsvector column + GIN index
    op.execute(
        "ALTER TABLE knowledge_units ADD COLUMN search_vector tsvector"
    )
    op.execute(
        "CREATE INDEX idx_ku_search_vector ON knowledge_units USING gin(search_vector)"
    )

    # 9. Create trigger to auto-update search_vector on INSERT/UPDATE
    op.execute("""
        CREATE OR REPLACE FUNCTION ku_search_vector_update() RETURNS trigger AS $$
        DECLARE
            kw_text text := '';
        BEGIN
            IF NEW.keywords IS NOT NULL AND NEW.keywords::text != 'null' THEN
                SELECT coalesce(string_agg(elem, ' '), '')
                INTO kw_text
                FROM json_array_elements_text(NEW.keywords::json) AS elem;
            END IF;

            NEW.search_vector :=
                setweight(to_tsvector('english',
                    coalesce(NEW.method_name, '') || ' ' ||
                    coalesce(NEW.field, '') || ' ' ||
                    coalesce(kw_text, '')
                ), 'A') ||
                setweight(to_tsvector('english',
                    coalesce(NEW.content, '') || ' ' ||
                    coalesce(NEW.title, '') || ' ' ||
                    coalesce(NEW.problem_it_solves, '')
                ), 'B');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE TRIGGER ku_search_vector_trigger
        BEFORE INSERT OR UPDATE ON knowledge_units
        FOR EACH ROW EXECUTE FUNCTION ku_search_vector_update()
    """)

    # 10. Backfill search_vector for all existing rows
    #     Touch each row to fire the trigger
    op.execute("""
        UPDATE knowledge_units SET title = title
    """)


def downgrade() -> None:
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS ku_search_vector_trigger ON knowledge_units")
    op.execute("DROP FUNCTION IF EXISTS ku_search_vector_update()")

    # Drop tsvector column and index
    op.execute("DROP INDEX IF EXISTS idx_ku_search_vector")
    op.execute("ALTER TABLE knowledge_units DROP COLUMN IF EXISTS search_vector")

    # Drop HNSW index
    op.execute("DROP INDEX IF EXISTS idx_ku_embedding_hnsw")

    # Convert vector columns back to JSON
    op.add_column("knowledge_units", sa.Column("embedding_json", sa.Text(), nullable=True))
    op.add_column("method_taxonomy", sa.Column("embedding_json", sa.Text(), nullable=True))

    op.execute("""
        UPDATE knowledge_units
        SET embedding_json = embedding::text
        WHERE embedding IS NOT NULL
    """)
    op.execute("""
        UPDATE method_taxonomy
        SET embedding_json = embedding::text
        WHERE embedding IS NOT NULL
    """)

    op.drop_column("knowledge_units", "embedding")
    op.drop_column("method_taxonomy", "embedding")

    op.add_column("knowledge_units", sa.Column("embedding", sa.JSON(), nullable=True))
    op.add_column("method_taxonomy", sa.Column("embedding", sa.JSON(), nullable=True))

    op.execute("""
        UPDATE knowledge_units
        SET embedding = embedding_json::json
        WHERE embedding_json IS NOT NULL
    """)
    op.execute("""
        UPDATE method_taxonomy
        SET embedding = embedding_json::json
        WHERE embedding_json IS NOT NULL
    """)

    op.drop_column("knowledge_units", "embedding_json")
    op.drop_column("method_taxonomy", "embedding_json")

    op.execute("DROP EXTENSION IF EXISTS vector")
```

- [ ] **Step 5: Rebuild docker container and run migration**

Run: `docker-compose down && docker-compose up -d`

Wait for PostgreSQL to be ready, then:

Run: `alembic upgrade head`

Expected: Migration applies successfully. Verify with:

Run: `alembic current`

Expected output includes: `002 (head)`

- [ ] **Step 6: Verify migration**

Run: `python -c "import sys; sys.path.insert(0,'apps/api'); sys.path.insert(0,'packages'); from database import SessionLocal; db=SessionLocal(); r=db.execute(__import__('sqlalchemy').text('SELECT count(*) FROM knowledge_units WHERE embedding IS NOT NULL')); print('KUs with embeddings:', r.scalar()); db.close()"`

Expected: Count matches the number of KUs that had embeddings before migration.

- [ ] **Step 7: Commit**

```bash
git add apps/api/models.py apps/api/main.py alembic/versions/
git commit -m "feat: add pgvector + tsvector migration, update models for vector types"
```

---

## Task 3: hybrid_search Function

**Files:**
- Modify: `packages/chat/embeddings.py`

**Depends on:** Task 2 (vector column type must exist)

- [ ] **Step 1: Add hybrid_search and helpers to embeddings.py**

Add the following code to `packages/chat/embeddings.py`, after the existing functions (keep all existing functions — they'll be used until service.py is updated in Task 5):

```python
# ---------------------------------------------------------------------------
# Hybrid search (pgvector + tsvector)
# ---------------------------------------------------------------------------

RRF_K = 60  # Reciprocal Rank Fusion constant


def _rrf_score(rank: int) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-indexed)."""
    return 1.0 / (RRF_K + rank)


def hybrid_search(
    db,
    vector_queries: list[str],
    keyword_terms: list[str],
    api_key: str,
    top_k: int = 50,
    boost_ids: set[int] | None = None,
    method_boost_ids: set[int] | None = None,
) -> list[tuple[float, dict]]:
    """Run dual-path retrieval: pgvector ANN + tsvector full-text, merged via RRF.

    Args:
        db: SQLAlchemy Session.
        vector_queries: Search queries for embedding-based retrieval.
        keyword_terms: Terms for full-text search (method names, aliases, keywords).
        api_key: OpenAI API key for computing query embeddings.
        top_k: Max results per search path.
        boost_ids: KU IDs to boost (taxonomy match) — score *= 1.3.
        method_boost_ids: KU IDs to boost (method match) — score *= 1.5.

    Returns:
        Sorted list of (score, ku_dict) tuples, highest score first.
    """
    from sqlalchemy import text as sa_text

    # --- Vector search: one query per search string, merge best rank per KU ---
    vector_ranked: dict[int, int] = {}  # ku_id -> best rank (1-indexed)
    for query_str in vector_queries:
        if not query_str.strip():
            continue
        emb = compute_embedding(query_str, api_key)
        if not emb:
            continue
        emb_str = "[" + ",".join(str(x) for x in emb) + "]"
        rows = db.execute(
            sa_text("""
                SELECT id, embedding <=> :emb AS distance
                FROM knowledge_units
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :emb
                LIMIT :k
            """),
            {"emb": emb_str, "k": top_k},
        ).fetchall()
        for rank, row in enumerate(rows, 1):
            ku_id = row[0]
            if ku_id not in vector_ranked or rank < vector_ranked[ku_id]:
                vector_ranked[ku_id] = rank

    # --- Full-text search: combine all keyword terms with OR ---
    text_ranked: dict[int, int] = {}  # ku_id -> rank (1-indexed)
    all_terms = list(keyword_terms)
    # Also add the original vector queries as keyword search terms
    for q in vector_queries:
        if q.strip():
            all_terms.append(q.strip())

    if all_terms:
        # Combine all terms with "or" for websearch_to_tsquery (parameterized, no injection risk)
        search_text = " or ".join(t.strip() for t in all_terms if t.strip())
        if search_text:
            rows = db.execute(
                sa_text("""
                    SELECT id, ts_rank_cd(search_vector, websearch_to_tsquery('english', :query), 32) AS rank
                    FROM knowledge_units
                    WHERE search_vector @@ websearch_to_tsquery('english', :query)
                    ORDER BY rank DESC
                    LIMIT :k
                """),
                {"query": search_text, "k": top_k},
            ).fetchall()
            for rank, row in enumerate(rows, 1):
                text_ranked[row[0]] = rank

    # --- RRF merge ---
    all_ids = set(vector_ranked.keys()) | set(text_ranked.keys())
    if not all_ids:
        return []

    rrf_scores: dict[int, float] = {}
    for ku_id in all_ids:
        score = 0.0
        if ku_id in vector_ranked:
            score += _rrf_score(vector_ranked[ku_id])
        if ku_id in text_ranked:
            score += _rrf_score(text_ranked[ku_id])
        # Apply boosts
        if boost_ids and ku_id in boost_ids:
            score *= 1.3
        if method_boost_ids and ku_id in method_boost_ids:
            score *= 1.5
        rrf_scores[ku_id] = round(score, 6)

    # --- Load full KU data for top results ---
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    top_ids = sorted_ids[:top_k]

    from models import KnowledgeUnit, Paper
    kus = db.query(KnowledgeUnit).filter(KnowledgeUnit.id.in_(top_ids)).all()
    ku_map = {ku.id: ku for ku in kus}

    # Load paper metadata
    paper_ids = {ku.paper_id for ku in kus if ku.paper_id}
    papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all() if paper_ids else []
    paper_map = {p.id: p for p in papers}

    # Build result dicts (matching _load_unit_dicts format in chat.py)
    _UNIT_FIELDS = (
        "title", "source_type", "section", "knowledge_type",
        "content", "evidence_span", "limitations", "confidence",
        "method_name", "field", "problem_it_solves", "model_assumption",
        "input_format", "output_format",
    )
    _LIST_FIELDS = (
        "topic_tags", "question_intent_tags", "dependencies",
        "reusable_for_questions", "keywords", "typical_questions",
        "related_methods",
    )

    results: list[tuple[float, dict]] = []
    for ku_id in top_ids:
        ku = ku_map.get(ku_id)
        if not ku:
            continue
        d = {"id": ku.id}
        d.update({c: getattr(ku, c) for c in _UNIT_FIELDS})
        d.update({c: getattr(ku, c) or [] for c in _LIST_FIELDS})
        if ku.paper_id and ku.paper_id in paper_map:
            paper = paper_map[ku.paper_id]
            d["_domain"] = paper.domain
            d["_paper_title"] = paper.title
            d["_paper_authors"] = paper.authors
            d["_paper_year"] = paper.year
            d["_paper_doi"] = paper.doi
        score = rrf_scores[ku_id]
        d["_similarity"] = score
        results.append((score, d))

    results.sort(key=lambda x: x[0], reverse=True)
    return results
```

- [ ] **Step 2: Verify no import errors**

Run: `python -c "import sys; sys.path.insert(0,'apps/api'); sys.path.insert(0,'packages'); from chat.embeddings import hybrid_search; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add packages/chat/embeddings.py
git commit -m "feat: add hybrid_search with pgvector ANN + tsvector full-text + RRF merge"
```

---

## Task 4: Query Expansion & Method Boost Functions

**Files:**
- Modify: `packages/chat/service.py`

**Depends on:** None (pure logic, can be done in parallel with Tasks 1-3)

- [ ] **Step 1: Add _expand_queries_with_skills to service.py**

Add this function after the `_filter_units_by_methods` function (around line 515) in `packages/chat/service.py`:

```python
def _expand_queries_with_skills(
    search_queries: list[str],
    method_skills: list[dict],
) -> tuple[list[str], list[str]]:
    """Expand search queries using MethodSkill aliases and related methods.

    Returns:
        (vector_queries, keyword_terms)
        - vector_queries: original search queries (for embedding search)
        - keyword_terms: extra method names/aliases (for full-text search only)
    """
    vector_queries = [q for q in search_queries if q.strip()]
    keyword_terms: list[str] = []
    seen_terms: set[str] = set()

    for query in vector_queries:
        query_lower = query.lower()
        for skill in method_skills:
            method = skill.get("method", "")
            aliases = skill.get("aliases", [])
            related = skill.get("related_methods", [])
            field = skill.get("field", "")

            # Check if any skill name/alias/field appears in the query
            all_names = [method] + aliases + [field]
            matched = any(
                name.lower() in query_lower or query_lower in name.lower()
                for name in all_names if name
            )
            if not matched:
                continue

            # Collect aliases and related methods as keyword terms
            for term in [method] + aliases + related:
                term = term.strip()
                if term and term.lower() not in seen_terms:
                    seen_terms.add(term.lower())
                    keyword_terms.append(term)

    return vector_queries, keyword_terms
```

- [ ] **Step 2: Add _compute_method_boost_ids to service.py**

Add this function right after `_expand_queries_with_skills`:

```python
def _compute_method_boost_ids(
    db,
    selected_methods: list[str],
    method_skills: list[dict],
) -> set[int]:
    """Return KU IDs matching selected methods, for score boosting (not filtering).

    Uses the same matching logic as the old _filter_units_by_methods but queries
    the DB directly and returns IDs instead of filtering a list.
    """
    if not selected_methods:
        return set()

    from models import KnowledgeUnit

    # Build allowed names and fields
    allowed_names: set[str] = set()
    allowed_fields: set[str] = set()
    for ms in method_skills:
        if ms.get("method") in selected_methods:
            allowed_names.add(ms["method"].lower())
            for alias in ms.get("aliases", []):
                allowed_names.add(alias.lower())
            field = (ms.get("field") or "").lower().strip()
            if field:
                allowed_fields.add(field)
    for m in selected_methods:
        allowed_names.add(m.lower())

    # Query all KU ids with method_name or field
    kus = db.query(KnowledgeUnit.id, KnowledgeUnit.method_name, KnowledgeUnit.field).all()
    boost_ids: set[int] = set()
    for ku_id, method_name, field in kus:
        name = (method_name or "").lower()
        f = (field or "").lower().strip()
        if any(a in name or name in a for a in allowed_names if a):
            boost_ids.add(ku_id)
        elif f and f in allowed_fields:
            boost_ids.add(ku_id)

    return boost_ids
```

- [ ] **Step 3: Verify no syntax errors**

Run: `python -c "import sys; sys.path.insert(0,'apps/api'); sys.path.insert(0,'packages'); from chat.service import _expand_queries_with_skills, _compute_method_boost_ids; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add packages/chat/service.py
git commit -m "feat: add query expansion with skill aliases and method soft-boost function"
```

---

## Task 5: Service Pipeline & Chat Router Refactor

**Files:**
- Modify: `packages/chat/service.py`
- Modify: `apps/api/routers/chat.py`

**Depends on:** Tasks 3 + 4 (needs hybrid_search, expansion, and boost functions)

- [ ] **Step 1: Update service.py imports**

At the top of `packages/chat/service.py`, add the `hybrid_search` import. Change line 18:

Old:
```python
from .embeddings import compute_embedding, cosine_similarity, _score_methods
```

New:
```python
from .embeddings import compute_embedding, cosine_similarity, _score_methods, hybrid_search
```

- [ ] **Step 2: Update generate_response signature**

Replace the `generate_response` function signature and body in `packages/chat/service.py` (around line 600). Add `db` parameter, remove `method_context`:

```python
def generate_response(
    message: str,
    api_key: str,
    db=None,
    history: list[dict[str, str]] | None = None,
    method_context: list[dict] | None = None,
    method_skills: list[dict] | None = None,
    taxonomy_nodes: list[dict] | None = None,
    dify_api_key: str | None = None,
    dify_base_url: str = "https://api.dify.ai/v1",
) -> tuple[str, str, list[dict]]:
    """Classify question -> retrieve knowledge -> select strategy -> generate response."""
    result = _prepare_generation_context(
        message, api_key, db=db, history=history,
        method_context=method_context, method_skills=method_skills,
        taxonomy_nodes=taxonomy_nodes,
    )

    # Clarify mode
    if result[1] == "clarify":
        _, _, clarify_text, debug_lines, matched_units = result
        return clarify_text, chr(10).join(debug_lines), matched_units

    msgs, strategy, knowledge_section, debug_lines, matched_units = result

    # Try Dify first
    if dify_api_key:
        try:
            history_text = json.dumps(history or [], ensure_ascii=False)
            answer = _call_dify_workflow(
                question=message,
                knowledge_context=knowledge_section,
                history=history_text,
                strategy=strategy,
                dify_api_key=dify_api_key,
                dify_base_url=dify_base_url,
            )
            debug_lines.append("LLM backend: **Dify Workflow**")
            return answer, chr(10).join(debug_lines), matched_units
        except Exception as e:
            logger.warning(f"[Chat] Dify call failed: {e}, falling back to OpenAI")
            debug_lines.append(f"LLM backend: **Dify FAILED ({e}), fell back to OpenAI**")

    # Fallback: direct OpenAI
    gen_params = {
        "direct_answer": {"temperature": 0.3, "max_tokens": 600},
        "comparison":    {"temperature": 0.5, "max_tokens": 800},
        "llm_only":      {"temperature": 0.7, "max_tokens": 400},
    }[strategy]
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, **gen_params)
    answer = resp.choices[0].message.content or "No response generated."
    debug_lines.append("LLM backend: **Direct OpenAI**")

    logger.info("[Chat] Done.")
    return answer, chr(10).join(debug_lines), matched_units
```

- [ ] **Step 3: Update generate_response_stream signature**

Same change for the streaming version (around line 792):

```python
def generate_response_stream(
    message: str,
    api_key: str,
    db=None,
    history: list[dict[str, str]] | None = None,
    method_context: list[dict] | None = None,
    method_skills: list[dict] | None = None,
    taxonomy_nodes: list[dict] | None = None,
    dify_api_key: str | None = None,
    dify_base_url: str = "https://api.dify.ai/v1",
):
```

And update the call to `_prepare_generation_context` inside it:

```python
    result = _prepare_generation_context(
        message, api_key, db=db, history=history,
        method_context=method_context, method_skills=method_skills,
        taxonomy_nodes=taxonomy_nodes,
    )
```

(The rest of the streaming function body stays the same.)

- [ ] **Step 4: Refactor _prepare_generation_context**

This is the core change. Replace the `_prepare_generation_context` function (around line 656) with the new hybrid search flow:

```python
def _prepare_generation_context(
    message: str,
    api_key: str,
    db=None,
    history: list[dict[str, str]] | None = None,
    method_context: list[dict] | None = None,
    method_skills: list[dict] | None = None,
    taxonomy_nodes: list[dict] | None = None,
) -> tuple:
    """Run the classification/retrieval pipeline, return everything needed for generation.

    Returns either:
    - (None, "clarify", clarify_text, debug_lines, []) for clarification mode
    - (msgs, strategy, knowledge_section, debug_lines, matched_units) for normal generation
    """
    effective_message = message
    if history:
        effective_message = _rewrite_query(message, history, api_key)
        if effective_message != message:
            logger.info(f"[Chat] Step 0: Rewrote query: '{message}' -> '{effective_message}'")

    logger.info("[Chat] Step 1: Classifying question...")
    route = classify_question(effective_message, _skills, api_key, history)

    queries = getattr(route, "search_queries", []) or [route.search_query]
    queries = [q for q in queries if q.strip()]

    debug_lines = [
        f"Skill: **{route.skill}**",
        f"Search queries: *{queries or '(none)'}*",
        f"Confidence: **{route.confidence}**",
    ]
    if effective_message != message:
        debug_lines.append(f"Rewritten query: *{effective_message}*")

    CONFIDENCE_THRESHOLD = 0.5
    if route.confidence < CONFIDENCE_THRESHOLD and not history:
        logger.info(f"[Chat] Low confidence ({route.confidence}), generating clarification...")
        client = OpenAI(api_key=api_key)
        clarify_msgs: list[dict[str, str]] = [{"role": "system", "content": CLARIFY_PROMPT}]
        clarify_msgs.append({"role": "user", "content": message})
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=clarify_msgs, temperature=0.5, max_tokens=200
        )
        clarify_text = resp.choices[0].message.content or "Could you provide more details?"
        debug_lines.append(f"Confidence below {CONFIDENCE_THRESHOLD}, clarifying")
        return None, "clarify", clarify_text, debug_lines, []

    strategy = "llm_only"
    system_prompt = LLM_ONLY_PROMPT
    knowledge_section = ""
    matched_units: list[dict] = []

    # Step 1.5: Method selection (soft boost)
    logger.info("[Chat] Step 1.5: Selecting methods...")
    selected_methods: list[str] = []
    q_analysis: dict = {}
    method_boost_ids: set[int] = set()
    if method_skills:
        selected_methods, q_analysis = _select_methods(effective_message, method_skills, api_key, history)
        if selected_methods:
            debug_lines.append(f"Question analysis: {q_analysis}")
            debug_lines.append(f"Selected methods: **{selected_methods}**")
            # Soft boost instead of hard filter
            if db is not None:
                method_boost_ids = _compute_method_boost_ids(db, selected_methods, method_skills)
                debug_lines.append(f"Method boost: {len(method_boost_ids)} KUs")
        else:
            debug_lines.append(f"Question analysis: {q_analysis}")
            debug_lines.append("Method selection: no specific method matched, searching all KUs")

    # Append method purpose to queries (same as before)
    if selected_methods and method_skills:
        for ms in method_skills:
            if ms.get("method") in selected_methods:
                purpose = ms.get("purpose", "")
                method_name = ms.get("method", "")
                field = ms.get("field", "")
                if purpose:
                    queries.append(f"{method_name} {field} {purpose}")

    # Step 1.6: Query expansion with skill aliases
    vector_queries = queries
    keyword_terms: list[str] = []
    if method_skills:
        vector_queries, keyword_terms = _expand_queries_with_skills(queries, method_skills)
        if keyword_terms:
            debug_lines.append(f"Expanded keywords: {keyword_terms[:10]}")

    # Step 1.75: Taxonomy-based boosting
    boost_ids: set[int] | None = None
    if vector_queries and taxonomy_nodes:
        logger.info("[Chat] Step 1.75: Taxonomy locating...")
        first_query_emb = compute_embedding(vector_queries[0], api_key)
        if first_query_emb:
            boost_ids = _taxonomy_locate(first_query_emb, taxonomy_nodes)
            debug_lines.append(f"Taxonomy boost: {len(boost_ids)} units from matched branches")
        else:
            debug_lines.append("Taxonomy boost: skipped (embedding failed)")

    # Step 2: Retrieval
    logger.info("[Chat] Step 2: Retrieving knowledge units...")
    use_hybrid = db is not None and vector_queries
    if use_hybrid:
        # New hybrid search path (pgvector + tsvector + RRF)
        scored = hybrid_search(
            db=db,
            vector_queries=vector_queries,
            keyword_terms=keyword_terms,
            api_key=api_key,
            top_k=50,
            boost_ids=boost_ids,
            method_boost_ids=method_boost_ids if method_boost_ids else None,
        )
        debug_lines.append(f"Hybrid search: {len(scored)} results")
    elif method_context and queries:
        # Legacy in-memory path (fallback for tests or when db is not passed)
        scored = _multi_query_search(
            queries, method_context, api_key,
            boost_ids=boost_ids,
        )
    else:
        scored = []

    if scored:
        debug_lines.append("")
        debug_lines.append("**Knowledge unit scores (top results):**")
        for score, u in scored[:15]:
            title = u.get("title", "unknown")
            ktype = u.get("knowledge_type", "")
            tags = u.get("topic_tags", [])
            tag_str = f" -- tags: {', '.join(tags[:5])}" if tags else ""
            debug_lines.append(f"- {title} [{ktype}]: {score}{tag_str}")

        strategy, selected_units, system_prompt = _select_strategy(
            scored, pre_filtered=bool(selected_methods)
        )
        knowledge_section = _build_knowledge_context(selected_units)
        matched_units = selected_units

    # Sibling method recommendations
    if matched_units and taxonomy_nodes:
        siblings = _get_sibling_methods(matched_units, taxonomy_nodes)
        if siblings:
            sibling_names = [f"{s['name']} ({s['node_type']})" for s in siblings]
            debug_lines.append(f"Related methods (taxonomy siblings): {', '.join(sibling_names)}")

    debug_lines.insert(1, f"Strategy: **{strategy}**")

    full_system = system_prompt
    if history:
        full_system += FOLLOW_UP_ADDENDUM
    if knowledge_section:
        full_system += CITATION_INSTRUCTION
        full_system += chr(10)*2 + "## Knowledge Base - Matched Units" + chr(10)*2 + knowledge_section

    msgs: list[dict[str, str]] = [{"role": "system", "content": full_system}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": message})

    return msgs, strategy, knowledge_section, debug_lines, matched_units
```

- [ ] **Step 5: Update chat.py router — pass db, remove KU bulk load**

In `apps/api/routers/chat.py`, update the `/chat` endpoint (around line 118). Change the generate_response call to pass `db` and remove `_load_unit_dicts()`:

Replace the body of the `chat()` function (lines 125-165):

```python
@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    x_session_id: str | None = Header(None),
):
    session_id = x_session_id or str(uuid.uuid4())

    # Save user message
    db.add(Message(session_id=session_id, user_id=current_user.id, role="user", content=body.message))
    db.commit()

    # Load conversation history for this session (excluding the message we just saved)
    past = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    history = [
        {"role": m.role, "content": m.content}
        for m in past[:-1]
    ][-MAX_HISTORY:]

    # Load skill cards and taxonomy nodes (small, kept in memory)
    skill_dicts = _load_skill_dicts(db)
    taxonomy = _load_taxonomy_nodes(db)

    # Generate response — db passed directly for hybrid search
    response_text, debug_text, matched_units = generate_response(
        body.message,
        api_key=settings.OPENAI_API_KEY,
        db=db,
        history=history,
        method_skills=skill_dicts,
        taxonomy_nodes=taxonomy or None,
        dify_api_key=settings.DIFY_API_KEY or None,
        dify_base_url=settings.DIFY_BASE_URL,
    )

    # Save assistant message
    db.add(Message(session_id=session_id, user_id=current_user.id, role="assistant", content=response_text))
    db.commit()

    references = _build_references(matched_units)
    return ChatResponse(response=response_text, debug=debug_text, session_id=session_id, references=references)
```

- [ ] **Step 6: Update chat_stream endpoint similarly**

Replace the streaming endpoint body (around line 280):

```python
@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    x_session_id: str | None = Header(None),
):
    """SSE streaming version of /chat. Returns Server-Sent Events."""
    session_id = x_session_id or str(uuid.uuid4())

    # Save user message
    db.add(Message(session_id=session_id, user_id=current_user.id, role="user", content=body.message))
    db.commit()

    # Load conversation history
    past = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    history = [
        {"role": m.role, "content": m.content}
        for m in past[:-1]
    ][-MAX_HISTORY:]

    # Load skill cards and taxonomy nodes
    skill_dicts = _load_skill_dicts(db)
    taxonomy = _load_taxonomy_nodes(db)

    def event_generator():
        full_answer = ""
        try:
            for event_type, data in generate_response_stream(
                body.message,
                api_key=settings.OPENAI_API_KEY,
                db=db,
                history=history,
                method_skills=skill_dicts,
                taxonomy_nodes=taxonomy or None,
                dify_api_key=settings.DIFY_API_KEY or None,
                dify_base_url=settings.DIFY_BASE_URL,
            ):
                if event_type == "token":
                    yield 'event: token' + chr(10) + 'data: ' + _json.dumps({'text': data}, ensure_ascii=False) + chr(10) + chr(10)
                elif event_type == "debug":
                    yield 'event: debug' + chr(10) + 'data: ' + _json.dumps({'debug': data}, ensure_ascii=False) + chr(10) + chr(10)
                elif event_type == "references":
                    refs = _build_references(data)
                    yield 'event: references' + chr(10) + 'data: ' + _json.dumps({'references': refs}, ensure_ascii=False) + chr(10) + chr(10)
                elif event_type == "done":
                    full_answer = data
                    db.add(Message(session_id=session_id, user_id=current_user.id, role="assistant", content=full_answer))
                    db.commit()
                    yield 'event: done' + chr(10) + 'data: ' + _json.dumps({'session_id': session_id}, ensure_ascii=False) + chr(10) + chr(10)
                elif event_type == "error":
                    yield 'event: error' + chr(10) + 'data: ' + _json.dumps({'error': data}, ensure_ascii=False) + chr(10) + chr(10)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[Chat] Stream generator error: {tb}")
            yield 'event: error' + chr(10) + 'data: ' + _json.dumps({'error': 'An internal error occurred'}, ensure_ascii=False) + chr(10) + chr(10)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

- [ ] **Step 7: Clean up unused imports in chat.py**

The `_load_unit_dicts` function and `KnowledgeUnit` import are no longer needed for the chat endpoints. However, `_load_unit_dicts` is still defined in the file — leave it for now (other code might reference it). Just remove `KnowledgeUnit` from the import if no other code in chat.py uses it.

Check: `KnowledgeUnit` is imported on line 14 but is only used in `_load_unit_dicts`. Since we're keeping `_load_unit_dicts` in the file (just not calling it from chat/stream), leave the import.

- [ ] **Step 8: Verify the server starts**

Run: `cd apps/api && python -c "from main import app; print('App loaded OK')"`

Expected: `App loaded OK` (no import errors)

- [ ] **Step 9: Commit**

```bash
git add packages/chat/service.py apps/api/routers/chat.py
git commit -m "feat: integrate hybrid search into service pipeline, pass db to retrieval"
```

---

## Task 6: Test Infrastructure & Verification

**Files:**
- Modify: `tests/conftest.py`

**Depends on:** All previous tasks

- [ ] **Step 1: Update conftest.py to handle pgvector absence in SQLite**

The test suite uses SQLite which doesn't support pgvector or tsvector. The models.py conditional types already handle this (uses JSON when DATABASE_URL is sqlite). We need to ensure the service layer falls back to the legacy in-memory path when `db` is not a PostgreSQL session.

The existing service.py already handles this: when `db is None` or the hybrid path isn't available, it falls back to `_multi_query_search` with `method_context`. But tests need to pass `method_context` (the old way) since SQLite doesn't support hybrid search.

Update `tests/conftest.py` to add a helper for creating test KUs:

```python
"""Shared test fixtures — SQLite in-memory DB, FastAPI TestClient, auth helpers."""

import sys
import os

# Make apps/api and packages importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

# Must set env vars BEFORE importing app modules
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["OPENAI_API_KEY"] = "sk-test-fake-key"
os.environ["JWT_SECRET_KEY"] = "test-secret"

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from database import Base, get_db
from main import app

engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    return TestClient(app)


def create_user(client: TestClient, username: str, password: str, role: str = "viewer", admin_token: str | None = None) -> dict:
    """Helper to register a user and return the user data."""
    headers = {}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"
    res = client.post(
        "/auth/register",
        json={"username": username, "password": password, "role": role},
        headers=headers,
    )
    assert res.status_code == 201, res.json()
    return res.json()


def login_user(client: TestClient, username: str, password: str) -> str:
    """Helper to login and return the access token."""
    res = client.post("/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200
    return res.json()["access_token"]


def auth_header(token: str) -> dict:
    """Return auth header dict."""
    return {"Authorization": f"Bearer {token}"}
```

This is actually unchanged from the original — the key point is that `models.py` now conditionally uses JSON for SQLite, so `Base.metadata.create_all()` will create tables with JSON columns for embedding (not Vector), and the search_vector column will be Text (not TSVECTOR). Tests that don't touch retrieval will work as before.

- [ ] **Step 2: Run existing tests to verify nothing broke**

Run: `pytest tests/ -v`

Expected: All existing tests pass. If any tests import or call `generate_response` without the `db` param, they still work because `db` defaults to `None` and the service falls back to the legacy `method_context` path.

- [ ] **Step 3: Manual smoke test with running server**

Start the server and test the chat endpoint:

Run: `uvicorn apps.api.main:app --reload`

Then test with curl or the frontend. The chat endpoint now passes `db` to the service, which uses `hybrid_search` for retrieval.

- [ ] **Step 4: Commit test changes (if any)**

```bash
git add tests/
git commit -m "test: verify existing tests pass with pgvector model changes"
```

---

## Parallelization Summary

```
Task 1 (Infrastructure) ─────┐
                              ├──> Task 3 (hybrid_search) ──┐
Task 2 (Models+Migrations) ──┘                              ├──> Task 5 (Service+Router) ──> Task 6 (Tests)
                                                             │
Task 4 (Query Expansion) ───────────────────────────────────┘
```

- **Parallel group 1:** Task 1 + Task 4 (no dependencies between them)
- **Parallel group 2:** Task 2 + Task 3 (Task 2 needs Task 1; Task 3 needs Task 2 models)
- **Sequential:** Task 5 needs Tasks 3+4; Task 6 needs Task 5

In practice, Tasks 1→2→3 are sequential (infrastructure chain), while Task 4 runs in parallel with all of them. Task 5 merges everything. Task 6 verifies.
