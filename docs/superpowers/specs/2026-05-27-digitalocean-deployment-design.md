# DigitalOcean App Platform Deployment Design

## Overview

Deploy the stat-research-assistant (model_bridge2) to DigitalOcean App Platform with three components: FastAPI backend as a Web Service, Next.js frontend as a Static Site, and a Dev Database for PostgreSQL.

## Architecture

```
GitHub Repo (model_bridge2)
        |
        v
  DigitalOcean App Platform
  +-----------------------------------+
  |                                   |
  |  [Static Site] Next.js frontend   |  Free
  |    static export -> CDN           |
  |    calls API via env var URL      |
  |                                   |
  |  [Web Service] FastAPI backend    |  $5/mo Basic
  |    Dockerfile-based build         |
  |    /api/* endpoints               |
  |                                   |
  |  [Dev Database] PostgreSQL        |  Free (1GB)
  |    auto-injected DATABASE_URL     |
  |                                   |
  +-----------------------------------+
```

## Code Changes Required

### 1. apps/api/Dockerfile (new)

Dockerfile for the FastAPI backend:
- Python 3.12 slim base
- Install dependencies from requirements.txt
- Run uvicorn on port 8080 (App Platform default)

### 2. apps/web/next.config.ts (modify)

Add output: 'export' to enable static HTML export for App Platform Static Site hosting.

### 3. apps/web/src/app/lib/api.ts (modify)

Change hardcoded http://localhost:8000 to use process.env.NEXT_PUBLIC_API_URL with localhost as fallback for local dev.

### 4. .do/app.yaml (new)

App Platform spec file defining:
- api: Web Service component, Dockerfile-based, source apps/api/, env vars bound to dev DB
- web: Static Site component, source apps/web/, build command npm run build, output dir out/
- db: Dev Database, PostgreSQL 16, engine PG

## Environment Variables

| Variable | Component | Source |
|---|---|---|
| DATABASE_URL | api | Auto-injected from Dev Database binding |
| OPENAI_API_KEY | api | Manual (secret) |
| JWT_SECRET_KEY | api | Manual (secret) |
| NEXT_PUBLIC_API_URL | web (build-time) | Set to api public URL |

## Deployment Flow

1. Push code to GitHub repo
2. Connect DigitalOcean App Platform to GitHub repo
3. App Platform reads .do/app.yaml
4. Auto builds and deploys both components
5. Future git push triggers auto-redeploy

## Cost

| Item | Cost |
|---|---|
| FastAPI Web Service (Basic) | $5/mo |
| Static Site (Starter) | Free |
| Dev Database (1GB) | Free |
| Total | ~$5/mo |

## Constraints

- Dev Database: 1GB storage limit, no automated backups
- Static Site: no SSR or API routes in Next.js (client-side only)
- Single region deployment
