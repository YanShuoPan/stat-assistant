from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


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
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


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
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
