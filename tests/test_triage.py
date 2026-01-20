from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_triage_invoke_success():
    r = client.post("/triage/invoke", json={"ticket_text": "My order ORD1001 is late"})
    assert r.status_code == 200
    data = r.json()
    assert data["order_id"] == "ORD1001"
    assert data["issue_type"] != ""
    assert "order" in data
    assert "reply_text" in data

def test_triage_invoke_missing_order_id():
    r = client.post("/triage/invoke", json={"ticket_text": "My order is late"})
    assert r.status_code == 400
