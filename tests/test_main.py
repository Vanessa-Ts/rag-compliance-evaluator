from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_index_returns_200():
    response = client.get("/")
    assert response.status_code == 200

def test_index_contains_app_name():
    response = client.get("/")
    assert "docker-dev-template" in response.text