# DigitalOcean App Platform Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the FastAPI + Next.js stat-research-assistant to DigitalOcean App Platform with auto-deploy from GitHub.

**Architecture:** Three App Platform components — FastAPI Web Service (Dockerfile), Next.js Static Site (static export), Dev Database (PostgreSQL). GitHub push triggers auto-deploy.

**Tech Stack:** FastAPI, Next.js 15, PostgreSQL, Docker, DigitalOcean App Platform

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `apps/api/Dockerfile` | Create | Docker build for FastAPI backend |
| `apps/api/main.py` | Modify (lines 35-41) | Make CORS origins configurable via env var |
| `apps/api/config.py` | Modify | Add CORS_ORIGINS setting |
| `apps/web/next.config.ts` | Modify | Add `output: 'export'` for static site |
| `apps/web/src/app/lib/api.ts` | Modify (line 1) | Use env var for API URL |
| `.do/app.yaml` | Create | App Platform deployment spec |
| `.dockerignore` | Create | Exclude unnecessary files from Docker build |
| `.gitignore` | Modify (if needed) | Ensure .env is excluded |

---

### Task 1: Create Dockerfile for FastAPI backend

**Files:**
- Create: `apps/api/Dockerfile`
- Create: `apps/api/.dockerignore`

- [ ] **Step 1: Create `apps/api/Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY apps/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api/ /app/apps/api/
COPY packages/ /app/packages/

WORKDIR /app/apps/api

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Note: Dockerfile is placed at `apps/api/Dockerfile` but uses repo root as build context (configured in app.yaml). This is because `main.py` adds `packages/` to sys.path.

- [ ] **Step 2: Create `.dockerignore` at repo root**

```
.git
.env
__pycache__
*.pyc
node_modules
apps/web
.do
docs
tests
*.md
```

- [ ] **Step 3: Test Docker build locally (optional)**

```bash
cd "d:/OneDrive - Morale AI/model_bridge2"
docker build -f apps/api/Dockerfile -t stat-api .
```

Expected: successful build

- [ ] **Step 4: Commit**

```bash
git add apps/api/Dockerfile .dockerignore
git commit -m "feat: add Dockerfile for FastAPI backend deployment"
```

---

### Task 2: Make CORS origins configurable

**Files:**
- Modify: `apps/api/config.py` (add CORS_ORIGINS field)
- Modify: `apps/api/main.py` (lines 35-41, use config)

- [ ] **Step 1: Add CORS_ORIGINS to config.py**

In `apps/api/config.py`, add this field to the `Settings` class:

```python
CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"
```

This is a comma-separated string because environment variables are strings. It will be split at usage.

- [ ] **Step 2: Update CORS in main.py**

Replace lines 35-41 in `apps/api/main.py`:

```python
from config import settings

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 3: Verify the app still starts locally**

```bash
cd "d:/OneDrive - Morale AI/model_bridge2"
python -m uvicorn apps.api.main:app --reload
```

Expected: starts without error, CORS still works for localhost

- [ ] **Step 4: Commit**

```bash
git add apps/api/config.py apps/api/main.py
git commit -m "feat: make CORS origins configurable via CORS_ORIGINS env var"
```

---

### Task 3: Configure Next.js static export and env-based API URL

**Files:**
- Modify: `apps/web/next.config.ts`
- Modify: `apps/web/src/app/lib/api.ts` (line 1)

- [ ] **Step 1: Update next.config.ts for static export**

Replace `apps/web/next.config.ts`:

```typescript
import type { NextConfig } from "next";
const nextConfig: NextConfig = {
  output: "export",
};
export default nextConfig;
```

- [ ] **Step 2: Update API URL to use environment variable**

Replace line 1 in `apps/web/src/app/lib/api.ts`:

```typescript
export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
```

`NEXT_PUBLIC_` prefix makes it available at build time in the browser bundle.

- [ ] **Step 3: Test static build locally**

```bash
cd "d:/OneDrive - Morale AI/model_bridge2/apps/web"
npm run build
```

Expected: creates `out/` directory with static HTML files. If there are errors about incompatible Next.js features, they need to be fixed.

- [ ] **Step 4: Commit**

```bash
git add apps/web/next.config.ts apps/web/src/app/lib/api.ts
git commit -m "feat: configure Next.js static export and env-based API URL"
```

---

### Task 4: Create DigitalOcean App Platform spec

**Files:**
- Create: `.do/app.yaml`

- [ ] **Step 1: Create `.do/app.yaml`**

```yaml
spec:
  name: stat-research-assistant
  region: sfo

  databases:
    - name: db
      engine: PG
      version: "16"
      production: false

  services:
    - name: api
      dockerfile_path: apps/api/Dockerfile
      source_dir: /
      http_port: 8080
      instance_count: 1
      instance_size_slug: basic-xxs
      envs:
        - key: DATABASE_URL
          scope: RUN_TIME
          value: ${db.DATABASE_URL}
        - key: OPENAI_API_KEY
          scope: RUN_TIME
          type: SECRET
        - key: JWT_SECRET_KEY
          scope: RUN_TIME
          type: SECRET
        - key: CORS_ORIGINS
          scope: RUN_TIME
          value: ${_self.PUBLIC_URL}

  static_sites:
    - name: web
      build_command: npm install && npm run build
      source_dir: /apps/web
      output_dir: out
      envs:
        - key: NEXT_PUBLIC_API_URL
          scope: BUILD_TIME
          value: ${api.PUBLIC_URL}
```

Notes:
- `production: false` = Dev Database (free)
- `basic-xxs` = $5/mo smallest instance
- `${db.DATABASE_URL}` = auto-injected connection string
- `${api.PUBLIC_URL}` = the auto-generated URL for the api service
- `${_self.PUBLIC_URL}` = the static site's own URL, used for CORS

- [ ] **Step 2: Commit**

```bash
git add .do/app.yaml
git commit -m "feat: add DigitalOcean App Platform deployment spec"
```

---

### Task 5: Push to GitHub and deploy

- [ ] **Step 1: Create GitHub repo**

```bash
gh repo create model_bridge2 --private --source=. --push
```

Or manually: go to github.com → New repository → name it `model_bridge2` → private → create.

Then:

```bash
git remote add origin git@github.com:YOUR_USERNAME/model_bridge2.git
git push -u origin main
```

- [ ] **Step 2: Connect DigitalOcean to GitHub**

1. Go to https://cloud.digitalocean.com/apps
2. Click "Create App"
3. Select "GitHub" as source
4. Authorize DigitalOcean to access your GitHub repo
5. Select the `model_bridge2` repo
6. App Platform should detect `.do/app.yaml` and auto-configure

- [ ] **Step 3: Set secret environment variables**

In the App Platform dashboard, go to the `api` component settings:
1. Set `OPENAI_API_KEY` to your actual OpenAI API key
2. Set `JWT_SECRET_KEY` to a random string (e.g. run `python -c "import secrets; print(secrets.token_hex(32))"`)

- [ ] **Step 4: Deploy and verify**

1. Click "Deploy" in the App Platform dashboard
2. Wait for build to complete (first deploy takes ~5 minutes)
3. Test the backend health endpoint: `curl https://<your-app-url>/health`
4. Open the static site URL in browser — should show the frontend
5. Test login and chat functionality

- [ ] **Step 5: Update CORS if needed**

After deployment, if the static site URL differs from what `${_self.PUBLIC_URL}` resolved to, update `CORS_ORIGINS` in the api component env vars to match the actual static site URL.
