"""pg_bigm CJK search indexes (skipped - replaced by LLM query translation)

Revision ID: 005
Revises: 004
Create Date: 2026-06-24
"""
from typing import Sequence, Union

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Originally added pg_bigm bigram indexes for CJK search.
    # Now replaced by LLM-based query translation to English,
    # so these indexes are no longer needed.
    pass


def downgrade() -> None:
    pass
