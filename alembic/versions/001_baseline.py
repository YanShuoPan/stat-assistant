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
