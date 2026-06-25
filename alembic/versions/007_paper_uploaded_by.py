"""add uploaded_by column to papers table

Revision ID: 007
Revises: 006
Create Date: 2026-06-25
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("uploaded_by", sa.Integer(), nullable=True))
    op.create_index("ix_papers_uploaded_by", "papers", ["uploaded_by"])
    op.create_foreign_key(
        "papers_uploaded_by_fkey",
        "papers", "users",
        ["uploaded_by"], ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("papers_uploaded_by_fkey", "papers", type_="foreignkey")
    op.drop_index("ix_papers_uploaded_by", "papers")
    op.drop_column("papers", "uploaded_by")
