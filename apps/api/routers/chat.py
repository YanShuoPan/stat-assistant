import uuid

from fastapi import APIRouter, Depends, Header
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import get_current_user
from config import settings
from database import get_db
from models import Message, KnowledgeUnit, MethodSkill, User
from schemas import ChatRequest, ChatResponse, MessageResponse, SessionSummary
from chat.service import generate_response

router = APIRouter(prefix="", tags=["chat"])


MAX_HISTORY = 20  # max prior messages sent to LLM


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    x_session_id: str | None = Header(None),
):
    session_id = x_session_id or str(uuid.uuid4())

    # Save user message
    db.add(Message(session_id=session_id, role="user", content=body.message))
    db.commit()

    # Load conversation history for this session (excluding the message we just saved)
    past = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    history = [
        {"role": m.role, "content": m.content}
        for m in past[:-1]
    ][-MAX_HISTORY:]

    # Load knowledge units as dicts for the skill system
    UNIT_FIELDS = (
        "title", "source_type", "section", "knowledge_type",
        "content", "evidence_span", "limitations", "confidence", "embedding",
        "method_name", "field", "problem_it_solves", "model_assumption",
        "input_format", "output_format",
    )
    LIST_FIELDS = (
        "topic_tags", "question_intent_tags", "dependencies",
        "reusable_for_questions", "keywords", "typical_questions",
        "related_methods",
    )
    unit_dicts = []
    for u in db.query(KnowledgeUnit).all():
        d = {c: getattr(u, c) for c in UNIT_FIELDS}
        d.update({c: getattr(u, c) or [] for c in LIST_FIELDS})
        unit_dicts.append(d)

    # Load method skill cards for pre-filtering
    SKILL_FIELDS = ("method", "field", "aliases", "purpose", "summary",
                    "pipeline_steps", "assumptions", "typical_questions",
                    "related_methods")
    skill_dicts = []
    for ms in db.query(MethodSkill).all():
        skill_dicts.append({c: getattr(ms, c) for c in SKILL_FIELDS})

    # Generate response via skill-routed LLM
    response_text, debug_text = generate_response(
        body.message,
        api_key=settings.OPENAI_API_KEY,
        history=history,
        method_context=unit_dicts,
        method_skills=skill_dicts,
        dify_api_key=settings.DIFY_API_KEY or None,
        dify_base_url=settings.DIFY_BASE_URL,
    )

    # Save assistant message (clean, without debug)
    db.add(Message(session_id=session_id, role="assistant", content=response_text))
    db.commit()

    return ChatResponse(response=response_text, debug=debug_text, session_id=session_id)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all conversation sessions for the current user, newest first."""
    # Get sessions that have at least one message from this user
    # For now, list all sessions (multi-user filtering can be added later)
    rows = (
        db.query(
            Message.session_id,
            func.min(Message.content).label("first_content"),
            func.max(Message.created_at).label("last_active"),
            func.count(Message.id).label("message_count"),
        )
        .group_by(Message.session_id)
        .order_by(func.max(Message.created_at).desc())
        .all()
    )
    result = []
    for row in rows:
        # Use first user message as title
        first_user_msg = (
            db.query(Message.content)
            .filter(Message.session_id == row.session_id, Message.role == "user")
            .order_by(Message.created_at.asc())
            .first()
        )
        title = (first_user_msg[0][:80] + "...") if first_user_msg and len(first_user_msg[0]) > 80 else (first_user_msg[0] if first_user_msg else "New conversation")
        result.append(SessionSummary(
            session_id=row.session_id,
            title=title,
            last_active=row.last_active,
            message_count=row.message_count,
        ))
    return result


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all messages for a specific session."""
    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return messages


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete all messages in a session."""
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.commit()
