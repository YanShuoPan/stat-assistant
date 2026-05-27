from conftest import create_user, login_user, auth_header
import pytest

@pytest.fixture(autouse=True)
def mock_embedding(monkeypatch):
    """Mock compute_embedding to avoid real OpenAI API calls."""
    fake = lambda text, api_key: [0.0] * 1536
    monkeypatch.setattr("chat.embeddings.compute_embedding", fake)
    monkeypatch.setattr("routers.methods.compute_embedding", fake)

SAMPLE_UNITS = {
    "units": [
        {
            "source_type": "paper",
            "title": "Adaptive Lasso",
            "section": "Method Overview",
            "knowledge_type": "definition",
            "topic_tags": ["feature selection", "high-dimensional", "regularization"],
            "question_intent_tags": ["what_is_it", "when_to_use"],
            "content": "Adaptive Lasso uses weighted L1 regularization where weights are derived from an initial consistent estimator, achieving oracle property under certain conditions.",
            "evidence_span": "Zou (2006), Theorem 2",
            "dependencies": ["initial estimator", "L1 penalty"],
            "limitations": "Depends on quality of initial estimator; may fail if initial weights are poor.",
            "confidence": "high",
            "reusable_for_questions": [
                "What is adaptive lasso?",
                "When should I use adaptive lasso over regular lasso?",
            ],
        },
        {
            "source_type": "paper",
            "title": "Adaptive Lasso",
            "section": "Assumptions",
            "knowledge_type": "assumption",
            "topic_tags": ["sparsity", "regularity conditions"],
            "question_intent_tags": ["how_it_works"],
            "content": "Requires approximate sparsity and that the initial estimator is root-n consistent.",
            "evidence_span": "Zou (2006), Section 3",
            "dependencies": ["root-n consistency"],
            "limitations": "Not verified under heavy-tailed error distributions.",
            "confidence": "high",
            "reusable_for_questions": [
                "What assumptions does adaptive lasso require?",
            ],
        },
    ]
}


def _admin_headers(client):
    create_user(client, "admin1", "pass")
    return auth_header(login_user(client, "admin1", "pass"))


def _researcher_headers(client, admin_token):
    create_user(client, "researcher1", "pass", role="researcher", admin_token=admin_token)
    return auth_header(login_user(client, "researcher1", "pass"))


def test_upload_requires_auth(client):
    res = client.post("/api/knowledge/upload", json=SAMPLE_UNITS)
    assert res.status_code in (401, 403)


def test_upload_knowledge(client):
    headers = _admin_headers(client)
    res = client.post("/api/knowledge/upload", json=SAMPLE_UNITS, headers=headers)
    assert res.status_code == 201
    data = res.json()
    assert len(data) == 2
    assert data[0]["title"] == "Adaptive Lasso"
    assert data[0]["knowledge_type"] == "definition"
    assert data[1]["knowledge_type"] == "assumption"
    assert data[0]["id"] is not None
    assert data[0]["uploaded_by"] is not None


def test_list_knowledge(client):
    headers = _admin_headers(client)
    client.post("/api/knowledge/upload", json=SAMPLE_UNITS, headers=headers)
    res = client.get("/api/knowledge", headers=headers)
    assert res.status_code == 200
    assert len(res.json()) == 2


def test_get_knowledge_unit(client):
    headers = _admin_headers(client)
    create = client.post("/api/knowledge/upload", json=SAMPLE_UNITS, headers=headers)
    uid = create.json()[0]["id"]

    res = client.get(f"/api/knowledge/{uid}", headers=headers)
    assert res.status_code == 200
    assert res.json()["title"] == "Adaptive Lasso"


def test_get_knowledge_unit_not_found(client):
    headers = _admin_headers(client)
    res = client.get("/api/knowledge/9999", headers=headers)
    assert res.status_code == 404


def test_delete_knowledge_unit_admin(client):
    headers = _admin_headers(client)
    create = client.post("/api/knowledge/upload", json=SAMPLE_UNITS, headers=headers)
    uid = create.json()[0]["id"]

    res = client.delete(f"/api/knowledge/{uid}", headers=headers)
    assert res.status_code == 204


def test_researcher_can_upload(client):
    admin_h = _admin_headers(client)
    admin_token = login_user(client, "admin1", "pass")
    researcher_h = _researcher_headers(client, admin_token)

    res = client.post("/api/knowledge/upload", json=SAMPLE_UNITS, headers=researcher_h)
    assert res.status_code == 201


def test_researcher_can_delete_own(client):
    admin_h = _admin_headers(client)
    admin_token = login_user(client, "admin1", "pass")
    researcher_h = _researcher_headers(client, admin_token)

    create = client.post("/api/knowledge/upload", json=SAMPLE_UNITS, headers=researcher_h)
    uid = create.json()[0]["id"]

    res = client.delete(f"/api/knowledge/{uid}", headers=researcher_h)
    assert res.status_code == 204


def test_researcher_cannot_delete_others(client):
    admin_h = _admin_headers(client)
    admin_token = login_user(client, "admin1", "pass")

    # Admin uploads knowledge
    create = client.post("/api/knowledge/upload", json=SAMPLE_UNITS, headers=admin_h)
    uid = create.json()[0]["id"]

    # Researcher tries to delete admin's unit
    researcher_h = _researcher_headers(client, admin_token)
    res = client.delete(f"/api/knowledge/{uid}", headers=researcher_h)
    assert res.status_code == 403


def test_viewer_cannot_upload(client):
    admin_h = _admin_headers(client)
    admin_token = login_user(client, "admin1", "pass")

    create_user(client, "viewer1", "pass", role="viewer", admin_token=admin_token)
    viewer_h = auth_header(login_user(client, "viewer1", "pass"))

    res = client.post("/api/knowledge/upload", json=SAMPLE_UNITS, headers=viewer_h)
    assert res.status_code == 403


def test_viewer_can_browse(client):
    admin_h = _admin_headers(client)
    admin_token = login_user(client, "admin1", "pass")

    client.post("/api/knowledge/upload", json=SAMPLE_UNITS, headers=admin_h)

    create_user(client, "viewer1", "pass", role="viewer", admin_token=admin_token)
    viewer_h = auth_header(login_user(client, "viewer1", "pass"))

    res = client.get("/api/knowledge", headers=viewer_h)
    assert res.status_code == 200
    assert len(res.json()) == 2
