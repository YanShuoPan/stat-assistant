from datetime import datetime

from pydantic import BaseModel, Field


# --- Auth ---

class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class UserCreate(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=4)
    role: str = "viewer"


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- Chat ---

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ReferenceItem(BaseModel):
    index: int
    method_name: str
    paper_title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None


class ChatResponse(BaseModel):
    response: str
    debug: str | None = None
    session_id: str | None = None
    references: list[ReferenceItem] = []

class MessageResponse(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionSummary(BaseModel):
    session_id: str
    title: str
    last_active: datetime
    message_count: int



# --- Papers ---

class PaperBase(BaseModel):
    title: str
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    domain: str
    cluster: str | None = None
    filename: str


class PaperResponse(PaperBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Knowledge Units ---

class KnowledgeUnitBase(BaseModel):
    source_type: str
    title: str
    section: str | None = None
    knowledge_type: str
    topic_tags: list[str] = []
    question_intent_tags: list[str] = []
    content: str
    evidence_span: str | None = None
    dependencies: list[str] = []
    limitations: str | None = None
    confidence: str = "medium"
    reusable_for_questions: list[str] = []
    method_name: str | None = None
    field: str | None = None
    keywords: list[str] = []
    problem_it_solves: str | None = None
    model_assumption: str | None = None
    input_format: str | None = None
    output_format: str | None = None
    typical_questions: list[str] = []
    related_methods: list[str] = []
    paper_id: int | None = None


class KnowledgeUnitCreate(KnowledgeUnitBase):
    pass


class KnowledgeUnitParsed(BaseModel):
    """LLM returns an array of these from uploaded files."""
    units: list[KnowledgeUnitBase] = []


class KnowledgeUnitResponse(KnowledgeUnitBase):
    id: int
    uploaded_by: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KnowledgeUnitBulkCreate(BaseModel):
    """Batch save multiple knowledge units at once."""
    units: list[KnowledgeUnitCreate]


# --- Method Skills ---

class MethodSkillBase(BaseModel):
    method: str
    field: str
    aliases: list[str] = []
    purpose: str
    summary: str
    pipeline_steps: list[str] = []
    assumptions: list[str] = []
    typical_questions: list[str] = []
    related_methods: list[str] = []


class MethodSkillResponse(MethodSkillBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
