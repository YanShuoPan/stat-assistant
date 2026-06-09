# Project Overview

This project is an LLM platform for statistics researchers.

The system allows:
1. researchers to upload their methods, papers, notes, and code
2. the platform to convert them into retrieval-ready knowledge
3. end users to ask questions about those uploaded methods
4. the LLM to answer using stored knowledge rather than generic model memory

The current priority is:
- make the upload → summarize/store → retrieve → answer pipeline work reliably
- keep the interface simple
- avoid unnecessary architecture expansion
- prefer practical fixes over broad refactoring

# Tech Stack

- Backend: FastAPI (Python) — `apps/api/`
- Frontend: Next.js 15 + React 19 + TypeScript — `apps/web/`
- Shared logic: `packages/chat/` (service, embeddings, skill routing)
- Database: PostgreSQL (prod), SQLite (dev/test)
- LLM: OpenAI API — gpt-4o-mini (parsing/routing), text-embedding-3-small (embeddings)
- Auth: JWT + bcrypt, role-based (admin/researcher/viewer)

# Development

- Install backend deps: `pip install -r apps/api/requirements.txt`
- Install frontend deps: `cd apps/web && npm install`
- Run backend: `uvicorn apps.api.main:app --reload`
- Run frontend: `cd apps/web && npm run dev`
- Run tests: `pytest tests/`
- Environment variables: copy `.env.example` to `.env` and fill in values
- PostgreSQL via `docker-compose up -d`

# Conventions

- API routes go in `apps/api/routers/`, one file per feature
- Pydantic schemas in `apps/api/schemas.py`, ORM models in `apps/api/models.py`
- Chat skills are YAML files in `packages/chat/skills/`
- Frontend pages use Next.js app router at `apps/web/src/app/[page]/page.tsx`
- API client helper in `apps/web/src/app/lib/api.ts`, backend at localhost:8000
- No linter/formatter configured yet — keep code style consistent with existing files
- Tests use SQLite in-memory + FastAPI TestClient; OPENAI_API_KEY mocked in fixtures

# Modification Rules

When making changes:
- first understand the existing module flow
- preserve current file structure unless explicitly asked to change it
- do not introduce new dependencies without discussion
- do not replace a working pipeline with a new architecture
- prefer targeted edits over full rewrites
- explain how each change fits into the current system

If a change affects multiple files, clearly state:
- which files should be modified
- why each modification is needed
- whether the change is required or optional

# Task Planning and Execution

When receiving a task, follow this workflow:

1. **Analyze** — Read relevant files to understand the current state before proposing changes
2. **Plan** — Break the task into concrete, independent subtasks using TodoWrite
3. **Dispatch** — Assign independent subtasks to subagents in parallel whenever possible
4. **Review** — Verify results after subagents complete, then mark tasks done

## When to Use Subagents

Actively look for parallelization opportunities. Common patterns in this project:

| Scenario | Parallel agents |
|----------|----------------|
| Feature touching backend + frontend | Agent 1: `apps/api/` changes, Agent 2: `apps/web/` changes |
| Adding a new router endpoint | Agent 1: router + schema, Agent 2: frontend page, Agent 3: test file |
| Bug investigation | Agent 1: explore backend logs/code, Agent 2: explore frontend code |
| Code review / understanding | Agent 1: read backend files, Agent 2: read frontend files, Agent 3: read packages/chat/ |
| Refactoring across modules | One agent per module (api, web, packages/chat) |

## Rules for Subagent Dispatch

- **Default to parallel**: If two subtasks don't depend on each other's output, run them as parallel subagents
- **Be specific**: Give each subagent the exact file paths, function names, and expected output
- **Include context**: Each subagent starts fresh — include relevant code snippets or file paths in the prompt
- **Use Explore agents** for research and investigation tasks
- **Use general-purpose agents** for tasks that require writing code
- **Use worktree isolation** (`isolation: "worktree"`) when agents need to make conflicting file edits
- **Run in background** when you have other work to do while waiting

## Project-Specific Boundaries for Subagents

Each subagent should stay within its module boundary:

- **Backend agent**: `apps/api/` — routers, models, schemas, auth, database, config
- **Frontend agent**: `apps/web/src/` — pages, components, lib/api.ts
- **Chat package agent**: `packages/chat/` — service, embeddings, router, skill_loader, method_skills
- **Test agent**: `tests/` — test files, conftest.py

## Planning Format

Use TodoWrite to track all subtasks. Example:

```
[ ] 1. Backend: add new endpoint in apps/api/routers/methods.py
[ ] 2. Schema: add request/response models in apps/api/schemas.py
[ ] 3. Frontend: add page in apps/web/src/app/newpage/page.tsx
[ ] 4. Tests: add test cases in tests/test_methods.py
```

Mark each task done immediately after completion — do not batch.
