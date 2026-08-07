"""Authentication lifecycle tests using an isolated database."""

from fastapi.testclient import TestClient

import app.auth as auth_module
from app.main import app


def test_register_login_session_and_logout(
    monkeypatch,
    tmp_path,
):
    test_database = tmp_path / "authentication.db"

    monkeypatch.setattr(
        auth_module,
        "DATABASE_PATH",
        test_database,
    )

    with TestClient(app) as client:
        registration = client.post(
            "/auth/register",
            json={
                "display_name": "Test Analyst",
                "email": "analyst@example.com",
                "password": "Secure123",
            },
        )

        assert registration.status_code == 201
        assert client.get("/auth/me").status_code == 200

        logout = client.post("/auth/logout")

        assert logout.status_code == 200
        assert client.get("/auth/me").status_code == 401

        login = client.post(
            "/auth/login",
            json={
                "email": "analyst@example.com",
                "password": "Secure123",
            },
        )

        assert login.status_code == 200
        assert client.get("/auth/me").status_code == 200