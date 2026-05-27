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
