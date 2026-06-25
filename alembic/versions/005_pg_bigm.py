"""enable pg_bigm extension and add bigram indexes for CJK search

Revision ID: 005
Revises: 004
Create Date: 2026-06-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Check if pg_bigm extension is available before trying to create it
    available = conn.execute(sa.text(
        "SELECT 1 FROM pg_available_extensions WHERE name = 'pg_bigm'"
    )).fetchone()

    if not available:
        import logging
        logging.getLogger(__name__).warning(
            "pg_bigm extension not available on this server, skipping bigram indexes"
        )
        return

    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pg_bigm"))

    # Check existing indexes before creating (idempotent)
    existing = {
        r[0] for r in conn.execute(
            sa.text("SELECT indexname FROM pg_indexes WHERE tablename='knowledge_units'")
        ).fetchall()
    }
    if "ix_knowledge_units_title_bigm" not in existing:
        conn.execute(sa.text(
            "CREATE INDEX ix_knowledge_units_title_bigm "
            "ON knowledge_units USING gin (title gin_bigm_ops)"
        ))
    if "ix_knowledge_units_content_bigm" not in existing:
        conn.execute(sa.text(
            "CREATE INDEX ix_knowledge_units_content_bigm "
            "ON knowledge_units USING gin (content gin_bigm_ops)"
        ))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("DROP INDEX IF EXISTS ix_knowledge_units_content_bigm"))
    conn.execute(sa.text("DROP INDEX IF EXISTS ix_knowledge_units_title_bigm"))
    # Do not drop pg_bigm extension - it may be used by other tables
