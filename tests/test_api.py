import pytest
import sys
sys.path.append('..')

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_register_user():
    response = client.post("/api/auth/register", json={
        "username": "testuser",
        "password": "testpass123"
    })
    assert response.status_code == 200
    assert "message" in response.json()

def test_login_user():
    # First register
    client.post("/api/auth/register", json={
        "username": "logintest",
        "password": "testpass123"
    })
    
    # Then login
    response = client.post("/api/auth/login", json={
        "username": "logintest",
        "password": "testpass123"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_mood_entry():
    # Register and login first
    client.post("/api/auth/register", json={
        "username": "moodtest",
        "password": "testpass123"
    })
    login_response = client.post("/api/auth/login", json={
        "username": "moodtest",
        "password": "testpass123"
    })
    token = login_response.json()["access_token"]

    # Test mood entry
    response = client.post(
        "/api/mood/entry",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "Feeling great!", "mood_score": 8}
    )
    assert response.status_code == 200
    assert "mood_score" in response.json()
