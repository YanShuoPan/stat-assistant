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
    conn = op.get_bind()

    # --- pgvector (optional: only if extension is available) ---
    has_pgvector = False
    try:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        has_pgvector = True
    except Exception:
        # Dev Database may not support extensions; skip vector conversion
        conn.execute(sa.text("ROLLBACK"))
        conn.execute(sa.text("BEGIN"))

    if has_pgvector:
        # Check if embedding column is already vector type (from a previous partial run)
        result = conn.execute(sa.text("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name = 'knowledge_units' AND column_name = 'embedding'
        """))
        row = result.fetchone()
        embedding_is_vector = row and row[0] == 'USER-DEFINED' if row else False

        if not embedding_is_vector:
            # 1. Add temp text columns alongside existing JSON columns
            op.execute("ALTER TABLE knowledge_units ADD COLUMN IF NOT EXISTS embedding_vec text")
            op.execute("ALTER TABLE method_taxonomy ADD COLUMN IF NOT EXISTS embedding_vec text")

            # 2. Convert existing JSON embeddings to text for vector cast
            #    Filter out null, 'null', and empty arrays '[]' (0-dimension vectors are invalid)
            op.execute("""
                UPDATE knowledge_units
                SET embedding_vec = embedding::text
                WHERE embedding IS NOT NULL
                  AND embedding::text != 'null'
                  AND embedding::text != '[]'
                  AND json_array_length(embedding::json) = 1536
            """)
            op.execute("""
                UPDATE method_taxonomy
                SET embedding_vec = embedding::text
                WHERE embedding IS NOT NULL
                  AND embedding::text != 'null'
                  AND embedding::text != '[]'
                  AND json_array_length(embedding::json) = 1536
            """)

            # 3. Drop old JSON columns, add proper vector columns
            op.execute("ALTER TABLE knowledge_units DROP COLUMN IF EXISTS embedding")
            op.execute("ALTER TABLE method_taxonomy DROP COLUMN IF EXISTS embedding")

            op.execute("ALTER TABLE knowledge_units ADD COLUMN embedding vector(1536)")
            op.execute("ALTER TABLE method_taxonomy ADD COLUMN embedding vector(1536)")

            # 4. Copy data from text temp columns to vector columns
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

            # 5. Drop temp columns
            op.execute("ALTER TABLE knowledge_units DROP COLUMN IF EXISTS embedding_vec")
            op.execute("ALTER TABLE method_taxonomy DROP COLUMN IF EXISTS embedding_vec")

        # 6. Create HNSW index for knowledge_units (idempotent)
        op.execute("DROP INDEX IF EXISTS idx_ku_embedding_hnsw")
        op.execute("""
            CREATE INDEX idx_ku_embedding_hnsw
            ON knowledge_units USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

    # --- tsvector (always available in PostgreSQL) ---

    # 7. Replace search_vector column with proper tsvector type + GIN index
    #    Drop existing column (Text from 001 or from create_all) if present
    op.execute(
        "ALTER TABLE knowledge_units DROP COLUMN IF EXISTS search_vector"
    )
    op.execute(
        "ALTER TABLE knowledge_units ADD COLUMN search_vector tsvector"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ku_search_vector ON knowledge_units USING gin(search_vector)"
    )

    # 8. Create trigger to auto-update search_vector on INSERT/UPDATE
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
    op.execute("DROP TRIGGER IF EXISTS ku_search_vector_trigger ON knowledge_units")
    op.execute("""
        CREATE TRIGGER ku_search_vector_trigger
        BEFORE INSERT OR UPDATE ON knowledge_units
        FOR EACH ROW EXECUTE FUNCTION ku_search_vector_update()
    """)

    # 9. Backfill search_vector for all existing rows
    op.execute("""
        UPDATE knowledge_units SET title = title
    """)


def downgrade() -> None:
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS ku_search_vector_trigger ON knowledge_units")
    op.execute("DROP FUNCTION IF EXISTS ku_search_vector_update()")

    # Drop tsvector column and index, restore as Text
    op.execute("DROP INDEX IF EXISTS idx_ku_search_vector")
    op.execute("ALTER TABLE knowledge_units DROP COLUMN IF EXISTS search_vector")
    op.add_column("knowledge_units", sa.Column("search_vector", sa.Text(), nullable=True))

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
