"""Shared test fixtures — SQLite in-memory DB, FastAPI TestClient, auth helpers."""

import sys
import os

# Make apps/api and packages importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

# Must set env vars BEFORE importing app modules
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["OPENAI_API_KEY"] = "sk-test-fake-key"
os.environ["JWT_SECRET_KEY"] = "test-secret"

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from database import Base, get_db
from main import app

engine = create_engine("sqlite:///./test.db", connect_args={"check_same_thread": False})
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    return TestClient(app)


def create_user(client: TestClient, username: str, password: str, role: str = "viewer", admin_token: str | None = None) -> dict:
    """Helper to register a user and return the user data."""
    headers = {}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"
    res = client.post(
        "/api/auth/register",
        json={"username": username, "password": password, "role": role},
        headers=headers,
    )
    assert res.status_code == 201, res.json()
    return res.json()


def login_user(client: TestClient, username: str, password: str) -> str:
    """Helper to login and return the access token."""
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200
    return res.json()["access_token"]


def auth_header(token: str) -> dict:
    """Return auth header dict."""
    return {"Authorization": f"Bearer {token}"}
