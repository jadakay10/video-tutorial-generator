import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from io import BytesIO


@pytest.fixture
def client():
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_upload_rejects_non_mp4(client):
    data = {"file": (BytesIO(b"fake"), "video.avi")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    body = json.loads(response.data)
    assert "error" in body


def test_upload_rejects_missing_file(client):
    response = client.post("/upload", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    body = json.loads(response.data)
    assert "error" in body


def test_upload_accepts_mp4(client, tmp_path):
    with patch("app.threading.Thread") as mock_thread, \
         patch("app.UPLOAD_DIR", tmp_path):
        mock_thread.return_value.start = MagicMock()
        data = {"file": (BytesIO(b"fake video"), "demo.mp4")}
        response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    body = json.loads(response.data)
    assert "job_id" in body


def test_progress_returns_404_for_unknown_job(client):
    response = client.get("/progress/unknown-job-id")
    assert response.status_code == 404
