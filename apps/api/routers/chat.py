import random
import uuid
import logging

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import get_current_user
from config import settings
from database import get_db
from models import Message, KnowledgeUnit, MethodSkill, MethodNode, KnowledgeUnitNode, User, Paper
from schemas import ChatRequest, ChatResponse, MessageResponse, SessionSummary
from chat.service import generate_response

router = APIRouter(prefix="", tags=["chat"])

# Field tuples for converting ORM objects to dicts
_UNIT_FIELDS = (
    "title", "source_type", "section", "knowledge_type",
    "content", "evidence_span", "limitations", "confidence", "embedding",
    "method_name", "field", "problem_it_solves", "model_assumption",
    "input_format", "output_format",
)
_LIST_FIELDS = (
    "topic_tags", "question_intent_tags", "dependencies",
    "reusable_for_questions", "keywords", "typical_questions",
    "related_methods",
)
_SKILL_FIELDS = (
    "method", "field", "aliases", "purpose", "summary",
    "pipeline_steps", "assumptions", "typical_questions",
    "related_methods",
)


def _load_unit_dicts(db: Session) -> list[dict]:
    """Load all knowledge units as dicts for the chat service."""
    # Pre-load papers for units that have paper_id to avoid N+1 queries
    paper_cache: dict[int, Paper] = {}
    result = []
    for u in db.query(KnowledgeUnit).all():
        d = {"id": u.id}
        d.update({c: getattr(u, c) for c in _UNIT_FIELDS})
        d.update({c: getattr(u, c) or [] for c in _LIST_FIELDS})
        # Load paper metadata if linked
        if u.paper_id:
            d["paper_id"] = u.paper_id
            if u.paper_id not in paper_cache:
                paper = db.query(Paper).filter(Paper.id == u.paper_id).first()
                if paper:
                    paper_cache[u.paper_id] = paper
            paper = paper_cache.get(u.paper_id)
            if paper:
                d["_domain"] = paper.domain
                d["_paper_title"] = paper.title
                d["_paper_authors"] = paper.authors
                d["_paper_year"] = paper.year
                d["_paper_doi"] = paper.doi
        result.append(d)
    return result


def _load_skill_dicts(db: Session) -> list[dict]:
    """Load all method skill cards as dicts."""
    return [{c: getattr(ms, c) for c in _SKILL_FIELDS} for ms in db.query(MethodSkill).all()]


def _load_taxonomy_nodes(db: Session) -> list[dict]:
    """Load all taxonomy nodes with their linked KU IDs for retrieval boosting."""
    nodes = db.query(MethodNode).all()
    if not nodes:
        return []

    # Build children lookup
    children_map: dict[int, list[int]] = {}
    for n in nodes:
        if n.parent_id is not None:
            children_map.setdefault(n.parent_id, []).append(n.id)

    # Load KU links
    ku_links: dict[int, list[int]] = {}
    for row in db.query(KnowledgeUnitNode).all():
        ku_links.setdefault(row.method_node_id, []).append(row.knowledge_unit_id)

    return [
        {
            "id": n.id,
            "name": n.name,
            "node_type": n.node_type,
            "parent_id": n.parent_id,
            "embedding": n.embedding,
            "children_ids": children_map.get(n.id, []),
            "knowledge_unit_ids": ku_links.get(n.id, []),
        }
        for n in nodes
    ]


def _build_references(matched_units: list[dict]) -> list[dict]:
    """Build reference list from matched knowledge units."""
    refs = []
    for i, u in enumerate(matched_units, 1):
        refs.append({
            "index": i,
            "method_name": u.get("method_name") or u.get("title", "Unknown"),
            "paper_title": u.get("_paper_title"),
            "authors": u.get("_paper_authors"),
            "year": u.get("_paper_year"),
            "doi": u.get("_paper_doi"),
        })
    return refs


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
    db.add(Message(session_id=session_id, user_id=current_user.id, role="user", content=body.message))
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

    # Load skill cards and taxonomy nodes
    skill_dicts = _load_skill_dicts(db)
    taxonomy = _load_taxonomy_nodes(db)

    # Generate response via skill-routed LLM (Dify disabled, use OpenAI directly)
    response_text, debug_text, matched_units = generate_response(
        body.message,
        api_key=settings.OPENAI_API_KEY,
        db=db,
        history=history,
        method_skills=skill_dicts,
        taxonomy_nodes=taxonomy or None,
    )

    # Save assistant message (clean, without debug)
    db.add(Message(session_id=session_id, user_id=current_user.id, role="assistant", content=response_text))
    db.commit()

    references = _build_references(matched_units)
    return ChatResponse(response=response_text, debug=debug_text, session_id=session_id, references=references)

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List conversation sessions for the current user, newest first."""
    rows = (
        db.query(
            Message.session_id,
            func.max(Message.created_at).label("last_active"),
            func.count(Message.id).label("message_count"),
        )
        .filter(Message.user_id == current_user.id)
        .group_by(Message.session_id)
        .order_by(func.max(Message.created_at).desc())
        .all()
    )
    result = []
    for row in rows:
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
    """Get all messages for a specific session owned by current user."""
    owns = db.query(Message).filter(
        Message.session_id == session_id,
        Message.user_id == current_user.id,
    ).first()
    if not owns:
        raise HTTPException(status_code=404, detail="Session not found")
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
    """Delete all messages in a session owned by current user."""
    owns = db.query(Message).filter(
        Message.session_id == session_id,
        Message.user_id == current_user.id,
    ).first()
    if not owns:
        raise HTTPException(status_code=404, detail="Session not found")
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.commit()

@router.get("/suggested-questions")
def get_suggested_questions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return 5 random suggested questions from method skills."""
    skills = db.query(MethodSkill).all()
    if not skills:
        return {"questions": [
            "What is OGA and how does it work?",
            "How do I handle high-dimensional data?",
            "What are the differences between LASSO and ridge regression?",
        ]}

    all_questions = []
    for skill in skills:
        typical = skill.typical_questions or []
        for q in typical:
            all_questions.append({"question": q, "method": skill.method})

    if not all_questions:
        return {"questions": ["What statistical methods are available?"]}

    selected = random.sample(all_questions, min(5, len(all_questions)))
    return {"questions": selected}


# ---------------------------------------------------------------------------
# Streaming chat endpoint (SSE)
# ---------------------------------------------------------------------------

from fastapi.responses import StreamingResponse
from chat.service import generate_response_stream
import json as _json


@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    x_session_id: str | None = Header(None),
):
    """SSE streaming version of /chat. Returns Server-Sent Events."""
    session_id = x_session_id or str(uuid.uuid4())

    # Save user message
    db.add(Message(session_id=session_id, user_id=current_user.id, role="user", content=body.message))
    db.commit()

    # Load conversation history
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

    # Load skill cards and taxonomy nodes
    skill_dicts = _load_skill_dicts(db)
    taxonomy = _load_taxonomy_nodes(db)

    def event_generator():
        full_answer = ""
        try:
            for event_type, data in generate_response_stream(
                body.message,
                api_key=settings.OPENAI_API_KEY,
                db=db,
                history=history,
                method_skills=skill_dicts,
                taxonomy_nodes=taxonomy or None,
            ):
                if event_type == "token":
                    yield 'event: token' + chr(10) + 'data: ' + _json.dumps({'text': data}, ensure_ascii=False) + chr(10) + chr(10)
                elif event_type == "debug":
                    yield 'event: debug' + chr(10) + 'data: ' + _json.dumps({'debug': data}, ensure_ascii=False) + chr(10) + chr(10)
                elif event_type == "references":
                    refs = _build_references(data)
                    yield 'event: references' + chr(10) + 'data: ' + _json.dumps({'references': refs}, ensure_ascii=False) + chr(10) + chr(10)
                elif event_type == "done":
                    full_answer = data
                    db.add(Message(session_id=session_id, user_id=current_user.id, role="assistant", content=full_answer))
                    db.commit()
                    yield 'event: done' + chr(10) + 'data: ' + _json.dumps({'session_id': session_id}, ensure_ascii=False) + chr(10) + chr(10)
                elif event_type == "error":
                    yield 'event: error' + chr(10) + 'data: ' + _json.dumps({'error': data}, ensure_ascii=False) + chr(10) + chr(10)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[Chat] Stream generator error: {tb}")
            yield 'event: error' + chr(10) + 'data: ' + _json.dumps({'error': 'An internal error occurred'}, ensure_ascii=False) + chr(10) + chr(10)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
