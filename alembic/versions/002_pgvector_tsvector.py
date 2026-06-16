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
