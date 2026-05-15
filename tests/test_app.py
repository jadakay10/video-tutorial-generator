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


@pytest.fixture(autouse=True)
def reset_job_state():
    import app as app_module
    app_module._active_job.clear()
    app_module._jobs.clear()
    yield
    app_module._active_job.clear()
    app_module._jobs.clear()


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


def test_upload_accepts_screenshots(client, tmp_path):
    with patch("app.threading.Thread") as mock_thread, \
         patch("app.UPLOAD_DIR", tmp_path):
        mock_thread.return_value.start = MagicMock()
        data = {
            "file": (BytesIO(b"fake video"), "demo.mp4"),
            "screenshots": (BytesIO(b"fake img"), "01-30.png"),
        }
        response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    body = json.loads(response.data)
    assert "job_id" in body
    # screenshots dir should have been created
    job_dirs = list(tmp_path.iterdir())
    assert len(job_dirs) == 1
    assert (job_dirs[0] / "screenshots").exists()


def test_progress_returns_404_for_unknown_job(client):
    response = client.get("/progress/unknown-job-id")
    assert response.status_code == 404


def test_pipeline_emits_skipped_when_screenshots_provided(tmp_path):
    import queue as queue_module
    from unittest.mock import patch, MagicMock
    from pathlib import Path

    # Create a fake screenshot file
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    (screenshots_dir / "01-30.png").write_bytes(b"fake")

    from app import _run_pipeline, _jobs
    job_id = "test-skip-job"
    q = queue_module.Queue()
    _jobs[job_id] = q

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")

    fake_frames = [{"path": screenshots_dir / "01-30.png", "seconds": 90.0, "label": "1m 30s"}]

    with patch("app.extract_audio"), \
         patch("app.load_screenshots", return_value=fake_frames), \
         patch("app.sample_frames", return_value=[screenshots_dir / "01-30.png"]), \
         patch("app.encode_frames", return_value=[]), \
         patch("app.transcribe_audio", return_value=("transcript", [])), \
         patch("app.generate_tutorial", return_value="# Tutorial"), \
         patch("app._active_job", [job_id]):
        _run_pipeline(job_id, video_path, screenshots_dir)

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    step2_event = next(e for e in events if e.get("step") == 2)
    assert step2_event.get("skipped") is True
    assert step2_event["label"] == "Extracting frames"

    del _jobs[job_id]


def test_pipeline_emits_step2_and_calls_extract_frames_when_no_screenshots(tmp_path):
    import queue as queue_module

    from app import _run_pipeline, _jobs
    job_id = "test-extract-job"
    q = queue_module.Queue()
    _jobs[job_id] = q

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")

    fake_frame = tmp_path / "frame_0001.jpg"
    fake_frame.write_bytes(b"fake")

    with patch("app.extract_audio"), \
         patch("app.extract_frames", return_value=[fake_frame]) as mock_extract, \
         patch("app.sample_frames", return_value=[fake_frame]), \
         patch("app.encode_frames", return_value=[]), \
         patch("app.transcribe_audio", return_value=("transcript", [])), \
         patch("app.generate_tutorial", return_value="# Tutorial"), \
         patch("app._active_job", [job_id]):
        _run_pipeline(job_id, video_path, screenshots_dir=None)

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    step2_event = next(e for e in events if e.get("step") == 2)
    assert "skipped" not in step2_event
    mock_extract.assert_called_once()

    del _jobs[job_id]


def test_pipeline_uses_project_root_screenshots_folder_as_fallback(tmp_path):
    import queue as queue_module

    from app import _run_pipeline, _jobs
    job_id = "test-root-screenshots-job"
    q = queue_module.Queue()
    _jobs[job_id] = q

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")

    root_screenshots = tmp_path / "screenshots"
    root_screenshots.mkdir()
    (root_screenshots / "01-00.png").write_bytes(b"fake")

    fake_shots = [{"path": root_screenshots / "01-00.png", "seconds": 60.0, "label": "1m 0s"}]
    fake_paired = [{"path": root_screenshots / "01-00.png", "seconds": 60.0, "label": "1m 0s", "context": "hello"}]

    with patch("app.extract_audio"), \
         patch("app.load_screenshots", return_value=fake_shots), \
         patch("app.match_screenshots_to_transcript", return_value=fake_paired), \
         patch("app.transcribe_audio", return_value=("transcript", [])), \
         patch("app.generate_tutorial", return_value="# Tutorial") as mock_gen, \
         patch("app.Path", side_effect=lambda p: root_screenshots if p == "screenshots" else Path(p)), \
         patch("app._active_job", [job_id]):
        _run_pipeline(job_id, video_path, screenshots_dir=None)

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    step2_event = next(e for e in events if e.get("step") == 2)
    assert step2_event.get("skipped") is True
    assert mock_gen.call_args[0][3] == fake_paired

    del _jobs[job_id]
