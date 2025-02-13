import pytest
from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_unbias_post():
    """Test the POST request for bias detection"""
    response = client.post("/unbias", json={"text": "[MASK] is the CEO of a multinational corporation."})
    assert response.status_code == 200
    assert "Final Statement" in response.json()

def test_unbias_get():
    """Test the GET request for bias detection"""
    response = client.get("/unbias", params={"text": "[MASK] is the CEO of a multinational corporation."})
    assert response.status_code == 200
    assert "Final Statement" in response.json()
