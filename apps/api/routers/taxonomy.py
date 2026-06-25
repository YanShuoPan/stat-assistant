import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import get_current_user, require_role
from database import get_db
from models import MethodNode, KnowledgeUnitNode, KnowledgeUnit, User
from schemas import (
    TaxonomyTreeResponse,
    MethodNodeSummary,
    MethodNodeDetail,
    MethodNodeResponse,
    MethodNodeUpdate,
    MergeNodesRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/taxonomy", tags=["taxonomy"])


def _build_tree(db: Session) -> list[dict]:
    """Build nested tree from flat MethodNode rows."""
    nodes = db.query(MethodNode).order_by(MethodNode.name).all()

    # Count units per node
    counts = (
        db.query(
            KnowledgeUnitNode.method_node_id,
            func.count(KnowledgeUnitNode.knowledge_unit_id),
        )
        .group_by(KnowledgeUnitNode.method_node_id)
        .all()
    )
    unit_counts: dict[int, int] = {node_id: cnt for node_id, cnt in counts}

    # Build lookup
    node_map: dict[int, dict] = {}
    for n in nodes:
        node_map[n.id] = {
            "id": n.id,
            "name": n.name,
            "node_type": n.node_type,
            "description": n.description,
            "auto_generated": n.auto_generated,
            "children_count": 0,
            "unit_count": unit_counts.get(n.id, 0),
            "children": [],
            "parent_id": n.parent_id,
        }

    # Build tree
    roots: list[dict] = []
    for node_id, node in node_map.items():
        parent_id = node.pop("parent_id")
        if parent_id and parent_id in node_map:
            node_map[parent_id]["children"].append(node)
            node_map[parent_id]["children_count"] += 1
        else:
            roots.append(node)

    # Propagate unit counts upward
    def _propagate_counts(node: dict) -> int:
        total = node["unit_count"]
        for child in node["children"]:
            total += _propagate_counts(child)
        node["unit_count"] = total
        return total

    for root in roots:
        _propagate_counts(root)

    return roots


@router.get("", response_model=TaxonomyTreeResponse)
def get_taxonomy_tree(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the full taxonomy tree."""
    tree = _build_tree(db)
    return TaxonomyTreeResponse(nodes=tree)


@router.get("/{node_id}", response_model=MethodNodeDetail)
def get_taxonomy_node(
    node_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get detailed info for a single taxonomy node."""
    node = db.query(MethodNode).filter(MethodNode.id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Taxonomy node not found")

    # Get children
    children = (
        db.query(MethodNode)
        .filter(MethodNode.parent_id == node_id)
        .order_by(MethodNode.name)
        .all()
    )

    # Get siblings (same parent, excluding self)
    siblings: list = []
    if node.parent_id:
        siblings = (
            db.query(MethodNode)
            .filter(MethodNode.parent_id == node.parent_id, MethodNode.id != node_id)
            .order_by(MethodNode.name)
            .all()
        )

    # Get parent
    parent = None
    if node.parent_id:
        parent = db.query(MethodNode).filter(MethodNode.id == node.parent_id).first()

    # Get related knowledge units
    unit_ids = [
        row[0]
        for row in db.query(KnowledgeUnitNode.knowledge_unit_id)
        .filter(KnowledgeUnitNode.method_node_id == node_id)
        .all()
    ]
    units: list = []
    units_by_type: dict[str, int] = {}
    if unit_ids:
        units = db.query(KnowledgeUnit).filter(KnowledgeUnit.id.in_(unit_ids)).all()
        for u in units:
            units_by_type[u.knowledge_type] = units_by_type.get(u.knowledge_type, 0) + 1

    # Count children's units
    child_unit_counts: dict[int, int] = {}
    if children:
        child_ids = [c.id for c in children]
        for cid, cnt in (
            db.query(
                KnowledgeUnitNode.method_node_id,
                func.count(KnowledgeUnitNode.knowledge_unit_id),
            )
            .filter(KnowledgeUnitNode.method_node_id.in_(child_ids))
            .group_by(KnowledgeUnitNode.method_node_id)
            .all()
        ):
            child_unit_counts[cid] = cnt

    def _to_summary(n, unit_count: int = 0) -> MethodNodeSummary:
        return MethodNodeSummary(
            id=n.id,
            name=n.name,
            node_type=n.node_type,
            description=n.description,
            auto_generated=n.auto_generated,
            children_count=db.query(MethodNode).filter(MethodNode.parent_id == n.id).count(),
            unit_count=unit_count,
        )

    return MethodNodeDetail(
        id=node.id,
        name=node.name,
        node_type=node.node_type,
        description=node.description,
        aliases=node.aliases or [],
        auto_generated=node.auto_generated,
        parent=_to_summary(parent) if parent else None,
        children=[_to_summary(c, child_unit_counts.get(c.id, 0)) for c in children],
        siblings=[_to_summary(s) for s in siblings],
        units_by_type=units_by_type,
        units=units,
        created_at=node.created_at,
        updated_at=node.updated_at,
    )


@router.put("/{node_id}", response_model=MethodNodeResponse)
def update_taxonomy_node(
    node_id: int,
    body: MethodNodeUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Update a taxonomy node (admin only)."""
    node = db.query(MethodNode).filter(MethodNode.id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Taxonomy node not found")

    if body.name is not None:
        node.name = body.name
    if body.description is not None:
        node.description = body.description
    if body.parent_id is not None:
        # Prevent circular reference
        if body.parent_id == node_id:
            raise HTTPException(status_code=400, detail="Node cannot be its own parent")
        node.parent_id = body.parent_id
    if body.aliases is not None:
        node.aliases = body.aliases

    node.auto_generated = False  # Mark as human-reviewed
    db.commit()
    db.refresh(node)
    return node


@router.post("/merge", response_model=MethodNodeResponse)
def merge_nodes(
    body: MergeNodesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Merge source node into target node (admin only).

    - Moves all KU links from source to target
    - Moves all children of source to target
    - Adds source name to target aliases
    - Deletes source node
    """
    if body.source_id == body.target_id:
        raise HTTPException(status_code=400, detail="Cannot merge a node into itself")

    source = db.query(MethodNode).filter(MethodNode.id == body.source_id).first()
    target = db.query(MethodNode).filter(MethodNode.id == body.target_id).first()
    if not source or not target:
        raise HTTPException(status_code=404, detail="Source or target node not found")

    # Move KU links — delete duplicates that already point to target first
    existing_ku_ids = {
        row[0]
        for row in db.query(KnowledgeUnitNode.knowledge_unit_id)
        .filter(KnowledgeUnitNode.method_node_id == target.id)
        .all()
    }
    source_links = (
        db.query(KnowledgeUnitNode)
        .filter(KnowledgeUnitNode.method_node_id == source.id)
        .all()
    )
    for link in source_links:
        if link.knowledge_unit_id in existing_ku_ids:
            # Would create a duplicate PK — just delete the source link
            db.delete(link)
        else:
            link.method_node_id = target.id

    # Move children
    db.query(MethodNode).filter(MethodNode.parent_id == source.id).update(
        {"parent_id": target.id}
    )

    # Add source name and aliases to target
    new_aliases = list(target.aliases or [])
    if source.name not in new_aliases and source.name != target.name:
        new_aliases.append(source.name)
    for alias in source.aliases or []:
        if alias not in new_aliases and alias != target.name:
            new_aliases.append(alias)
    target.aliases = new_aliases

    # Delete source
    db.delete(source)
    db.commit()
    db.refresh(target)
    from routers.chat import invalidate_chat_cache
    invalidate_chat_cache()
    return target


@router.post("/classify", response_model=dict)
def classify_existing_units(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin", "researcher")),
):
    """Trigger classification of all existing KUs into taxonomy.

    This is for initial population or re-classification.
    """
    from chat.taxonomy import classify_units_to_taxonomy
    from config import settings

    units = db.query(KnowledgeUnit).all()
    if not units:
        raise HTTPException(status_code=400, detail="No knowledge units in database")

    result = classify_units_to_taxonomy(db, units, settings.OPENAI_API_KEY)
    from routers.chat import invalidate_chat_cache
    invalidate_chat_cache()
    return {"classified": result["classified"], "new_nodes": result["new_nodes"]}
