from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_index_returns_200():
    response = client.get("/")
    assert response.status_code == 200

def test_index_contains_app_name():
    response = client.get("/")
    assert "rag-compliance-evaluator" in response.text