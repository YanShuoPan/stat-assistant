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


def upgrade() -> None:
    conn = op.get_bind()

    # knowledge_units: drop existing FK, recreate with CASCADE
    existing_ku = [
        r[0] for r in conn.execute(sa.text(
            "SELECT conname FROM pg_constraint "
            "WHERE contype='f' AND conrelid='knowledge_units'::regclass"
        )).fetchall()
    ]
    ku_fk = next(
        (n for n in existing_ku if 'paper_id' in n),
        'knowledge_units_paper_id_fkey'
    )
    op.drop_constraint(ku_fk, 'knowledge_units', type_='foreignkey')
    op.create_foreign_key(
        'knowledge_units_paper_id_fkey',
        'knowledge_units', 'papers',
        ['paper_id'], ['id'],
        ondelete='CASCADE',
    )

    # paper_sections: drop existing FK, recreate with CASCADE
    existing_ps = [
        r[0] for r in conn.execute(sa.text(
            "SELECT conname FROM pg_constraint "
            "WHERE contype='f' AND conrelid='paper_sections'::regclass"
        )).fetchall()
    ]
    ps_fk = next(
        (n for n in existing_ps if 'paper_id' in n),
        'paper_sections_paper_id_fkey'
    )
    op.drop_constraint(ps_fk, 'paper_sections', type_='foreignkey')
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
