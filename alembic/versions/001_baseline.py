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


def _table_exists(name: str) -> bool:
    conn = op.get_bind()
    return sa.inspect(conn).has_table(name)


def upgrade() -> None:
    if not _table_exists("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("username", sa.String(100), unique=True, nullable=False, index=True),
            sa.Column("hashed_password", sa.String(255), nullable=False),
            sa.Column("role", sa.String(20), nullable=False, server_default="viewer"),
            sa.Column("created_at", sa.DateTime),
        )

    if not _table_exists("papers"):
        op.create_table(
            "papers",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("title", sa.String(500), nullable=False),
            sa.Column("authors", sa.Text, nullable=True),
            sa.Column("year", sa.Integer, nullable=True),
            sa.Column("doi", sa.String(255), nullable=True, index=True),
            sa.Column("arxiv_id", sa.String(50), nullable=True, index=True),
            sa.Column("domain", sa.String(100), nullable=False, index=True),
            sa.Column("cluster", sa.String(100), nullable=True),
            sa.Column("filename", sa.String(500), nullable=False),
            sa.Column("created_at", sa.DateTime),
        )

    if not _table_exists("method_taxonomy"):
        op.create_table(
            "method_taxonomy",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(255), nullable=False, index=True),
            sa.Column("node_type", sa.String(50), nullable=False, index=True),
            sa.Column("parent_id", sa.Integer, sa.ForeignKey("method_taxonomy.id"), nullable=True),
            sa.Column("aliases", sa.JSON, nullable=False, server_default="[]"),
            sa.Column("description", sa.Text, nullable=True),
            sa.Column("auto_generated", sa.Boolean, server_default="true"),
            sa.Column("embedding", sa.JSON, nullable=True),
            sa.Column("created_at", sa.DateTime),
            sa.Column("updated_at", sa.DateTime),
        )

    if not _table_exists("messages"):
        op.create_table(
            "messages",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("session_id", sa.String(36), nullable=False, index=True),
            sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("role", sa.String(20), nullable=False),
            sa.Column("content", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime, index=True),
        )

    if not _table_exists("knowledge_units"):
        op.create_table(
            "knowledge_units",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("source_type", sa.String(50), nullable=False, index=True),
            sa.Column("title", sa.String(255), nullable=False, index=True),
            sa.Column("section", sa.String(255), nullable=True),
            sa.Column("knowledge_type", sa.String(50), nullable=False, index=True),
            sa.Column("topic_tags", sa.JSON, nullable=False, server_default="[]"),
            sa.Column("question_intent_tags", sa.JSON, nullable=False, server_default="[]"),
            sa.Column("content", sa.Text, nullable=False),
            sa.Column("evidence_span", sa.Text, nullable=True),
            sa.Column("dependencies", sa.JSON, nullable=True),
            sa.Column("limitations", sa.Text, nullable=True),
            sa.Column("confidence", sa.String(20), nullable=False, server_default="medium"),
            sa.Column("reusable_for_questions", sa.JSON, nullable=False, server_default="[]"),
            sa.Column("method_name", sa.String(255), nullable=True, index=True),
            sa.Column("field", sa.String(255), nullable=True),
            sa.Column("keywords", sa.JSON, nullable=True),
            sa.Column("problem_it_solves", sa.Text, nullable=True),
            sa.Column("model_assumption", sa.Text, nullable=True),
            sa.Column("input_format", sa.Text, nullable=True),
            sa.Column("output_format", sa.Text, nullable=True),
            sa.Column("typical_questions", sa.JSON, nullable=True),
            sa.Column("related_methods", sa.JSON, nullable=True),
            sa.Column("embedding", sa.JSON, nullable=True),
            sa.Column("search_vector", sa.Text, nullable=True),
            sa.Column("paper_id", sa.Integer, sa.ForeignKey("papers.id"), nullable=True),
            sa.Column("uploaded_by", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime),
            sa.Column("updated_at", sa.DateTime),
        )

    if not _table_exists("method_skills"):
        op.create_table(
            "method_skills",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("method", sa.String(255), unique=True, nullable=False, index=True),
            sa.Column("field", sa.String(255), nullable=False),
            sa.Column("aliases", sa.JSON, nullable=False, server_default="[]"),
            sa.Column("purpose", sa.Text, nullable=False),
            sa.Column("summary", sa.Text, nullable=False),
            sa.Column("pipeline_steps", sa.JSON, nullable=False, server_default="[]"),
            sa.Column("assumptions", sa.JSON, nullable=False, server_default="[]"),
            sa.Column("typical_questions", sa.JSON, nullable=False, server_default="[]"),
            sa.Column("related_methods", sa.JSON, nullable=False, server_default="[]"),
            sa.Column("method_node_id", sa.Integer, sa.ForeignKey("method_taxonomy.id"), nullable=True),
            sa.Column("created_at", sa.DateTime),
            sa.Column("updated_at", sa.DateTime),
        )

    if not _table_exists("knowledge_unit_nodes"):
        op.create_table(
            "knowledge_unit_nodes",
            sa.Column("knowledge_unit_id", sa.Integer, sa.ForeignKey("knowledge_units.id"), primary_key=True),
            sa.Column("method_node_id", sa.Integer, sa.ForeignKey("method_taxonomy.id"), primary_key=True),
        )


def downgrade() -> None:
    op.drop_table("knowledge_unit_nodes")
    op.drop_table("method_skills")
    op.drop_table("knowledge_units")
    op.drop_table("messages")
    op.drop_table("method_taxonomy")
    op.drop_table("papers")
    op.drop_table("users")
