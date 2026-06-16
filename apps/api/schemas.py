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


class PaperCreate(PaperBase):
    """Used when creating a new paper via API."""
    pass


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
    """Batch save multiple knowledge units at once.

    If ``paper`` is provided, a Paper record is created first and all
    units are linked to it (individual ``paper_id`` values on units are
    ignored in that case).
    """
    units: list[KnowledgeUnitCreate]
    paper: PaperCreate | None = None


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


# --- Taxonomy ---

class MethodNodeBase(BaseModel):
    name: str
    node_type: str  # "problem_category" | "method_family" | "method" | "variant"
    parent_id: int | None = None
    aliases: list[str] = []
    description: str | None = None


class MethodNodeCreate(MethodNodeBase):
    pass


class MethodNodeUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    parent_id: int | None = None
    aliases: list[str] | None = None


class MethodNodeSummary(BaseModel):
    id: int
    name: str
    node_type: str
    description: str | None = None
    auto_generated: bool = True
    children_count: int = 0
    unit_count: int = 0
    children: list["MethodNodeSummary"] = []

    model_config = {"from_attributes": True}


class MethodNodeDetail(BaseModel):
    id: int
    name: str
    node_type: str
    description: str | None = None
    aliases: list[str] = []
    auto_generated: bool = True
    parent: MethodNodeSummary | None = None
    children: list[MethodNodeSummary] = []
    siblings: list[MethodNodeSummary] = []
    units_by_type: dict[str, int] = {}
    units: list[KnowledgeUnitResponse] = []
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MethodNodeResponse(MethodNodeBase):
    id: int
    auto_generated: bool = True
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TaxonomyTreeResponse(BaseModel):
    nodes: list[MethodNodeSummary]


class MergeNodesRequest(BaseModel):
    source_id: int
    target_id: int
