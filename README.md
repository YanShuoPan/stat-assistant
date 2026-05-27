# Statistical Research Assistant (MVP)

A minimal statistical research assistant with GPT-style chat and researcher method upload.

## Quick Start

### 1. Start PostgreSQL
```bash
docker-compose up -d
```

### 2. Start API
```bash
cd apps/api
pip install -r requirements.txt
DATABASE_URL="sqlite:///./dev.db" uvicorn main:app --reload --port 8000
```

### 3. Start Web
```bash
cd apps/web
npm install
npm run dev
```

- Web: http://localhost:3000
- API Docs: http://localhost:8000/docs
