import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_extract_audio_calls_ffmpeg(tmp_path):
    from video_to_tutorial import extract_audio
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    audio = tmp_path / "v.mp3"
    with patch("video_to_tutorial.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        extract_audio(video, audio)
    args = mock_run.call_args[0][0]
    assert "ffmpeg" in args
    assert str(audio) in args


def test_extract_frames_calls_ffmpeg(tmp_path):
    from video_to_tutorial import extract_frames
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    frames_dir = tmp_path / "frames"
    with patch("video_to_tutorial.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        # Create a fake frame file so glob returns it
        frames_dir.mkdir()
        (frames_dir / "frame_0001.jpg").write_bytes(b"fake")
        result = extract_frames(video, frames_dir)
    args = mock_run.call_args[0][0]
    assert "fps=1/5" in " ".join(args)
    assert len(result) == 1
    assert result[0].name == "frame_0001.jpg"


def test_transcribe_audio_returns_string(tmp_path):
    from video_to_tutorial import transcribe_audio
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": " Hello world"}
    with patch("video_to_tutorial.whisper.load_model", return_value=mock_model):
        result = transcribe_audio(audio)
    assert result == "Hello world"


def test_encode_frames_returns_image_blocks(tmp_path):
    from video_to_tutorial import encode_frames
    frame = tmp_path / "frame_0001.jpg"
    frame.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header
    blocks = encode_frames([frame])
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["type"] == "base64"
    assert blocks[0]["source"]["media_type"] == "image/jpeg"


def test_generate_tutorial_returns_string():
    from video_to_tutorial import generate_tutorial
    mock_client = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["# Title\n", "Step 1"])
    mock_client.messages.stream.return_value = mock_stream
    with patch("video_to_tutorial.anthropic.Anthropic", return_value=mock_client):
        result = generate_tutorial("transcript text", [])
    assert result == "# Title\nStep 1"
