# Method Taxonomy Design Spec

**Date:** 2026-06-14
**Status:** Approved

## Overview

Restructure knowledge storage around a **method taxonomy** — a three-level hierarchy:

```
Problem Category (e.g. Variable Selection, Causal Inference)
  └─ Method Family (e.g. Greedy Methods, Penalization)
       └─ Method / Variant (e.g. OGA, OGA+HDIC+Trim)
```

**Goals:**
1. Retrieval: taxonomy-aware search with boost scoring for matched branches
2. Browsing: frontend tree navigation on `/methods` page
3. Fully automatic: LLM classifies methods into taxonomy during paper parsing

---

## Data Model

### New Table: `method_taxonomy`

```python
class MethodNode(Base):
    __tablename__ = "method_taxonomy"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    node_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
        # "problem_category" | "method_family" | "method" | "variant"
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("method_taxonomy.id"), nullable=True)
    aliases: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    auto_generated: Mapped[bool] = mapped_column(default=True)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
```

### New Table: `knowledge_unit_nodes` (junction)

```python
class KnowledgeUnitNode(Base):
    __tablename__ = "knowledge_unit_nodes"

    knowledge_unit_id: Mapped[int] = mapped_column(Integer, ForeignKey("knowledge_units.id"), primary_key=True)
    method_node_id: Mapped[int] = mapped_column(Integer, ForeignKey("method_taxonomy.id"), primary_key=True)
```

### Modifications to Existing Tables

- **MethodSkill**: add `method_node_id: int | None` FK to `method_taxonomy`
- **KnowledgeUnit**: keep existing `method_name` field (backward compat), add relationship via junction

---

## Auto-Classification Logic

### When It Runs

1. After `/knowledge/upload` saves units (alongside existing skill regeneration)
2. After `scripts/parse_papers_gpt.py` batch processing

### Two-Phase Process

**Phase 1: LLM Classification (per upload batch)**

Input to LLM:
- Paper title + abstract (if available)
- List of distinct `method_name` values from the uploaded units
- Current taxonomy tree (JSON dump, max 200 nodes; if larger, send top 2 levels + relevant branches)

LLM returns per method:
```json
{
  "method_name": "OGA+HDIC",
  "problem_category": "Variable Selection",
  "method_family": "Greedy Methods",
  "parent_method": "OGA",
  "node_type": "variant",
  "aliases": ["OGA with HDIC criterion"],
  "description": "OGA variable selection with HDIC stopping rule",
  "is_new_node": true
}
```

**Phase 2: Match & Insert (programmatic)**

1. **Normalize** — lowercase + strip the returned names
2. **Exact match** — compare against existing `MethodNode.name` and `aliases`
3. **Embedding similarity** — if no exact match, compute embedding of the suggested name+description, compare against all existing node embeddings. Threshold: > 0.90 = same node
4. **Decision**:
   - Match found → create `KnowledgeUnitNode` links
   - No match + `is_new_node=true` → create new `MethodNode(auto_generated=True)`, create missing parent nodes if needed
5. **Same-parent dedup** — if a new sibling has > 0.85 similarity to existing sibling under same parent, merge into existing (add as alias)

### LLM Prompt Design Principles

- Provide full existing tree as context
- Instruct: "Prefer mapping to existing nodes. Only suggest new nodes for genuinely new methods."
- Return structured JSON array
- Model: `gpt-4o-mini`, temperature=0.2

---

## Retrieval Integration

### Modified Pipeline

```
User Question
  → [1] Router (existing)
  → [2] Taxonomy Locator (NEW — embedding-based, no LLM call)
  → [3] Boosted search (modified)
  → [4] Response generation + sibling recommendations (modified)
```

### Taxonomy Locator

- Compute question embedding
- Compare against all `MethodNode.embedding` vectors
- Take top-3 matching nodes
- Expand upward (get ancestors) and downward (get descendants)
- Collect `knowledge_unit_id` set from matched subtree via junction table

### Boost Strategy (soft filter)

```python
for unit in all_units:
    score = cosine_similarity(query_emb, unit.embedding)
    if unit.id in taxonomy_matched_ids:
        score *= 1.3  # 30% boost
    scores.append((score, unit))
```

- Non-matched units still appear (prevents taxonomy errors from hiding results)
- Unclassified units are never penalized

### Sibling Recommendations

After generating the answer, if the response mentions a specific method:
1. Find that method's `MethodNode`
2. Query siblings (same `parent_id`, different `id`)
3. Query cousins (same grandparent → different family)
4. Append to response as "Related methods" section

No extra LLM call needed — pure tree query.

---

## API Endpoints

### New Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/taxonomy` | any | Full tree (nested JSON) |
| GET | `/taxonomy/{node_id}` | any | Node detail + children + KU summary |
| PUT | `/taxonomy/{node_id}` | admin | Edit node (name, description, parent_id) |
| POST | `/taxonomy/merge` | admin | Merge two nodes (dedup) |
| POST | `/taxonomy/classify` | admin/researcher | Manually trigger classification for existing KUs |

### Tree Response Format

```json
{
  "nodes": [
    {
      "id": 1,
      "name": "Variable Selection",
      "node_type": "problem_category",
      "description": "...",
      "auto_generated": true,
      "children_count": 3,
      "unit_count": 45,
      "children": [
        {
          "id": 2,
          "name": "Greedy Methods",
          "node_type": "method_family",
          "children_count": 2,
          "unit_count": 28,
          "children": [...]
        }
      ]
    }
  ]
}
```

### Node Detail Response

```json
{
  "id": 3,
  "name": "OGA",
  "node_type": "method",
  "description": "...",
  "aliases": ["Orthogonal Greedy Algorithm"],
  "auto_generated": true,
  "parent": { "id": 2, "name": "Greedy Methods" },
  "children": [...],
  "siblings": [{ "id": 5, "name": "Forward Stepwise" }],
  "units_by_type": {
    "definition": 5,
    "algorithm": 8,
    "assumption": 4,
    "result": 6
  },
  "units": [...]
}
```

---

## Frontend: `/methods` Page (Taxonomy Browser)

### Layout

```
┌─────────────────────────────────────────────────┐
│  Method Taxonomy                    [Search...]  │
├─────────────────┬───────────────────────────────┤
│  Tree Nav       │  Detail Panel                 │
│                 │                               │
│  ▼ Variable     │  OGA                          │
│    Selection    │  Field: high-dimensional reg. │
│    ▼ Greedy     │  Aliases: Orthogonal Greedy.. │
│      • OGA ←   │                               │
│      • Forward  │  Description: ...             │
│    ▶ Penaliz.. │                               │
│                 │  Knowledge Units (28):        │
│  ▶ Causal      │  [grouped by knowledge_type]  │
│    Inference    │                               │
│                 │  Siblings: Forward Stepwise   │
│  ▶ Missing     │  [Ask about this method →]    │
│    Data         │                               │
└─────────────────┴───────────────────────────────┘
```

### Interactions

- Click node → show detail in right panel
- Expand/collapse → lazy-load children from `/taxonomy/{id}`
- Search box → client-side fuzzy filter on node names + aliases
- "Ask about this method" → navigate to `/chat` with pre-filled context

### Replace existing `/methods` page

The current `/methods` page shows a flat list of KnowledgeUnits grouped by title. The new taxonomy browser replaces it entirely, providing hierarchical navigation instead.

---

## Migration Strategy

1. Create new tables (`method_taxonomy`, `knowledge_unit_nodes`)
2. Add `method_node_id` column to `method_skills`
3. Run one-time classification of all existing KnowledgeUnits to populate taxonomy
4. Keep `KnowledgeUnit.method_name` field (read-only backward compat; new code uses junction table)

---

## File Changes Summary

| File | Change |
|------|--------|
| `apps/api/models.py` | Add `MethodNode`, `KnowledgeUnitNode` models; add `method_node_id` to `MethodSkill` |
| `apps/api/schemas.py` | Add taxonomy request/response schemas |
| `apps/api/routers/taxonomy.py` | New router: CRUD + tree + merge + classify endpoints |
| `apps/api/routers/methods.py` | Hook classification into `/knowledge/upload` |
| `packages/chat/taxonomy.py` | New module: classification logic (LLM call + match + insert) |
| `packages/chat/service.py` | Add taxonomy locator + boost logic + sibling recommendations |
| `packages/chat/embeddings.py` | Add node embedding helper |
| `apps/web/src/app/methods/page.tsx` | Replace with taxonomy browser |
| `apps/web/src/app/lib/api.ts` | Add taxonomy API helpers |
| `tests/test_taxonomy.py` | New test file |
