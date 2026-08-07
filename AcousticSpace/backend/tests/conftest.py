"""Shared backend test configuration and fixtures."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.auth import require_authenticated_user
from app.main import app


@pytest.fixture
def authenticated_client():
    """Provide a client with an authenticated test user."""

    app.dependency_overrides[
        require_authenticated_user
    ] = lambda: {
        "id": 1,
        "email": "analyst@example.com",
        "display_name": "Test Analyst",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(
            require_authenticated_user,
            None,
        )