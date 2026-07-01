from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from database import Base
from config import settings

_IS_PG = settings.DATABASE_URL.startswith("postgresql")

if _IS_PG:
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.dialects.postgresql import TSVECTOR

    _EmbeddingType = Vector(1536)
    _SearchVectorType = TSVECTOR
else:
    _EmbeddingType = JSON
    _SearchVectorType = Text


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="viewer")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    domain: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    cluster: Mapped[str | None] = mapped_column(String(100), nullable=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    file_content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PaperSection(Base):
    __tablename__ = "paper_sections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("papers.id"), nullable=False, index=True)
    section_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    section_index: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class KnowledgeUnit(Base):
    __tablename__ = "knowledge_units"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    knowledge_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    topic_tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    question_intent_tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_span: Mapped[str | None] = mapped_column(Text, nullable=True)
    dependencies: Mapped[list | None] = mapped_column(JSON, nullable=True)
    limitations: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")
    reusable_for_questions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    method_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    field: Mapped[str | None] = mapped_column(String(255), nullable=True)
    keywords: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    problem_it_solves: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_assumption: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_format: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_format: Mapped[str | None] = mapped_column(Text, nullable=True)
    typical_questions: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    related_methods: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    embedding = mapped_column(_EmbeddingType, nullable=True)
    search_vector = mapped_column(_SearchVectorType, nullable=True)
    paper_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("papers.id"), nullable=True)
    uploaded_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class MethodSkill(Base):
    __tablename__ = "method_skills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    method: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    field: Mapped[str] = mapped_column(String(255), nullable=False)
    aliases: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_steps: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    assumptions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    typical_questions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    related_methods: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    method_node_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("method_taxonomy.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class MethodNode(Base):
    __tablename__ = "method_taxonomy"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    node_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("method_taxonomy.id"), nullable=True)
    aliases: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    auto_generated: Mapped[bool] = mapped_column(default=True)
    embedding = mapped_column(_EmbeddingType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class KnowledgeUnitNode(Base):
    __tablename__ = "knowledge_unit_nodes"

    knowledge_unit_id: Mapped[int] = mapped_column(Integer, ForeignKey("knowledge_units.id"), primary_key=True)
    method_node_id: Mapped[int] = mapped_column(Integer, ForeignKey("method_taxonomy.id"), primary_key=True)
