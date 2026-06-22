"""add paper_sections table and paper file columns

Revision ID: 004
Revises: 003
Create Date: 2026-06-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    conn = op.get_bind()
    return sa.inspect(conn).has_table(table)


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    cols = {c["name"] for c in sa.inspect(conn).get_columns(table)}
    return column in cols


def upgrade() -> None:
    # --- New table: paper_sections ---
    if not _table_exists("paper_sections"):
        op.create_table(
            "paper_sections",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("paper_id", sa.Integer(), sa.ForeignKey("papers.id"), nullable=False),
            sa.Column("section_type", sa.String(50), nullable=False),
            sa.Column("section_index", sa.Integer(), nullable=False),
            sa.Column("summary", sa.String(500), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("char_count", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_paper_sections_paper_id", "paper_sections", ["paper_id"])
        op.create_index("ix_paper_sections_section_type", "paper_sections", ["section_type"])

    # --- New columns on papers table ---
    if not _column_exists("papers", "file_data"):
        op.add_column("papers", sa.Column("file_data", sa.LargeBinary(), nullable=True))

    if not _column_exists("papers", "file_content_type"):
        op.add_column("papers", sa.Column("file_content_type", sa.String(100), nullable=True))

    if not _column_exists("papers", "file_size"):
        op.add_column("papers", sa.Column("file_size", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_table("paper_sections")

    for col in ("file_data", "file_content_type", "file_size"):
        op.execute(f"ALTER TABLE papers DROP COLUMN IF EXISTS {col}")
