from conftest import create_user, login_user, auth_header


def test_register_first_user_becomes_admin(client):
    """First registered user is always admin regardless of requested role."""
    res = client.post("/api/auth/register", json={
        "username": "first", "password": "pass123", "role": "viewer",
    })
    assert res.status_code == 201
    assert res.json()["role"] == "admin"


def test_login_success(client):
    create_user(client, "alice", "secret123")
    res = client.post("/api/auth/login", json={"username": "alice", "password": "secret123"})
    assert res.status_code == 200
    assert "access_token" in res.json()


def test_login_wrong_password(client):
    create_user(client, "alice", "secret123")
    res = client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
    assert res.status_code == 401


def test_me_endpoint(client):
    create_user(client, "admin1", "pass")
    token = login_user(client, "admin1", "pass")
    res = client.get("/api/auth/me", headers=auth_header(token))
    assert res.status_code == 200
    assert res.json()["username"] == "admin1"
    assert res.json()["role"] == "admin"


def test_register_second_user_requires_admin(client):
    """Non-first registration without admin token should fail."""
    create_user(client, "admin1", "pass")

    # Try without token
    res = client.post("/api/auth/register", json={
        "username": "bob", "password": "pass", "role": "researcher",
    })
    assert res.status_code == 403


def test_admin_can_create_users(client):
    create_user(client, "admin1", "pass")
    token = login_user(client, "admin1", "pass")

    res = client.post(
        "/api/auth/register",
        json={"username": "researcher1", "password": "pass", "role": "researcher"},
        headers=auth_header(token),
    )
    assert res.status_code == 201
    assert res.json()["role"] == "researcher"


def test_duplicate_username(client):
    create_user(client, "admin1", "pass")
    token = login_user(client, "admin1", "pass")

    res = client.post(
        "/api/auth/register",
        json={"username": "admin1", "password": "other", "role": "viewer"},
        headers=auth_header(token),
    )
    assert res.status_code == 409
