import logging
import os
import sys
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)

# Make packages/ importable across all routers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine

from routers.auth import router as auth_router
from routers.chat import router as chat_router
from routers.methods import router as methods_router

import models  # noqa: F401
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Statistical Research Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(methods_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
