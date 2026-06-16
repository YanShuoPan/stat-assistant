"""add columns missing from old create_all tables

Revision ID: 003
Revises: 002
Create Date: 2026-06-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    cols = {c["name"] for c in sa.inspect(conn).get_columns(table)}
    return column in cols


def upgrade() -> None:
    # knowledge_units columns that may be missing from the original create_all()
    if not _column_exists("knowledge_units", "paper_id"):
        op.add_column("knowledge_units",
                       sa.Column("paper_id", sa.Integer(), nullable=True))
        op.execute(
            "ALTER TABLE knowledge_units "
            "ADD CONSTRAINT fk_ku_paper FOREIGN KEY (paper_id) REFERENCES papers(id)"
        )

    if not _column_exists("knowledge_units", "uploaded_by"):
        op.add_column("knowledge_units",
                       sa.Column("uploaded_by", sa.Integer(), nullable=True))
        op.execute(
            "ALTER TABLE knowledge_units "
            "ADD CONSTRAINT fk_ku_user FOREIGN KEY (uploaded_by) REFERENCES users(id)"
        )

    if not _column_exists("knowledge_units", "updated_at"):
        op.add_column("knowledge_units",
                       sa.Column("updated_at", sa.DateTime(), nullable=True))

    # method_skills columns that may be missing
    if not _column_exists("method_skills", "method_node_id"):
        op.add_column("method_skills",
                       sa.Column("method_node_id", sa.Integer(), nullable=True))
        op.execute(
            "ALTER TABLE method_skills "
            "ADD CONSTRAINT fk_ms_node FOREIGN KEY (method_node_id) REFERENCES method_taxonomy(id)"
        )

    if not _column_exists("method_skills", "updated_at"):
        op.add_column("method_skills",
                       sa.Column("updated_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.execute("ALTER TABLE method_skills DROP CONSTRAINT IF EXISTS fk_ms_node")
    op.execute("ALTER TABLE knowledge_units DROP CONSTRAINT IF EXISTS fk_ku_user")
    op.execute("ALTER TABLE knowledge_units DROP CONSTRAINT IF EXISTS fk_ku_paper")

    for table, col in [
        ("method_skills", "method_node_id"),
        ("method_skills", "updated_at"),
        ("knowledge_units", "updated_at"),
        ("knowledge_units", "uploaded_by"),
        ("knowledge_units", "paper_id"),
    ]:
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {col}")
