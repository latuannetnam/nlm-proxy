# tests/test_openai_proxy.py
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint():
    from notebooklm_mcp.openai_proxy import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
