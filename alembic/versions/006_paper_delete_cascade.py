"""add ON DELETE CASCADE to knowledge_units and paper_sections paper_id FKs

Revision ID: 006
Revises: 005
Create Date: 2026-06-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _find_fk_on_column(table: str, column: str):
    """Find FK constraint names by the actual column they reference."""
    conn = op.get_bind()
    return [
        r[0] for r in conn.execute(sa.text(
            "SELECT con.conname FROM pg_constraint con "
            "JOIN pg_attribute att ON att.attnum = ANY(con.conkey) "
            "AND att.attrelid = con.conrelid "
            "WHERE con.contype = 'f' "
            f"AND con.conrelid = '{table}'::regclass "
            f"AND att.attname = '{column}'"
        )).fetchall()
    ]


def upgrade() -> None:
    # knowledge_units: drop existing FK on paper_id, recreate with CASCADE
    for fk_name in _find_fk_on_column('knowledge_units', 'paper_id'):
        op.drop_constraint(fk_name, 'knowledge_units', type_='foreignkey')
    op.create_foreign_key(
        'knowledge_units_paper_id_fkey',
        'knowledge_units', 'papers',
        ['paper_id'], ['id'],
        ondelete='CASCADE',
    )

    # paper_sections: drop existing FK on paper_id, recreate with CASCADE
    for fk_name in _find_fk_on_column('paper_sections', 'paper_id'):
        op.drop_constraint(fk_name, 'paper_sections', type_='foreignkey')
    op.create_foreign_key(
        'paper_sections_paper_id_fkey',
        'paper_sections', 'papers',
        ['paper_id'], ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    # Remove CASCADE (revert to plain FK, no action on delete)
    op.drop_constraint('knowledge_units_paper_id_fkey', 'knowledge_units', type_='foreignkey')
    op.create_foreign_key(
        'knowledge_units_paper_id_fkey',
        'knowledge_units', 'papers',
        ['paper_id'], ['id'],
    )

    op.drop_constraint('paper_sections_paper_id_fkey', 'paper_sections', type_='foreignkey')
    op.create_foreign_key(
        'paper_sections_paper_id_fkey',
        'paper_sections', 'papers',
        ['paper_id'], ['id'],
    )
