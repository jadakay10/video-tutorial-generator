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


def test_transcribe_audio_returns_tuple(tmp_path):
    from video_to_tutorial import transcribe_audio
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": " Hello world",
        "segments": [{"start": 0.0, "end": 1.5, "text": " Hello world"}],
    }
    with patch("video_to_tutorial.whisper.load_model", return_value=mock_model):
        text, segments = transcribe_audio(audio)
    assert text == "Hello world"
    assert len(segments) == 1
    assert segments[0] == {"start": 0.0, "end": 1.5, "text": "Hello world"}
    mock_model.transcribe.assert_called_once_with(str(audio), verbose=False)


def test_encode_frames_returns_image_blocks(tmp_path):
    from video_to_tutorial import encode_frames
    frame = tmp_path / "frame_0001.jpg"
    frame.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header
    blocks = encode_frames([frame])
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["type"] == "base64"
    assert blocks[0]["source"]["media_type"] == "image/jpeg"


def test_load_screenshots_parses_mm_ss(tmp_path):
    from video_to_tutorial import load_screenshots
    (tmp_path / "01-30.png").write_bytes(b"")
    results = load_screenshots(tmp_path)
    assert len(results) == 1
    assert results[0]["seconds"] == 90.0
    assert results[0]["label"] == "1m 30s"
    assert results[0]["path"] == tmp_path / "01-30.png"


def test_load_screenshots_parses_hh_mm_ss(tmp_path):
    from video_to_tutorial import load_screenshots
    (tmp_path / "01-02-30.jpg").write_bytes(b"")
    results = load_screenshots(tmp_path)
    assert len(results) == 1
    assert results[0]["seconds"] == 3750.0
    assert results[0]["label"] == "1h 2m 30s"


def test_load_screenshots_sorted_by_seconds(tmp_path):
    from video_to_tutorial import load_screenshots
    (tmp_path / "02-00.png").write_bytes(b"")
    (tmp_path / "00-30.jpg").write_bytes(b"")
    (tmp_path / "01-00.jpeg").write_bytes(b"")
    results = load_screenshots(tmp_path)
    assert [r["seconds"] for r in results] == [30.0, 60.0, 120.0]


def test_load_screenshots_skips_bad_names(tmp_path, capsys):
    from video_to_tutorial import load_screenshots
    (tmp_path / "screenshot.png").write_bytes(b"")
    (tmp_path / "01-30.png").write_bytes(b"")
    results = load_screenshots(tmp_path)
    assert len(results) == 1
    captured = capsys.readouterr()
    assert "screenshot.png" in captured.out


def test_load_screenshots_ignores_non_image_files(tmp_path):
    from video_to_tutorial import load_screenshots
    (tmp_path / "01-30.txt").write_bytes(b"")
    (tmp_path / "01-30.mp4").write_bytes(b"")
    (tmp_path / "01-30.png").write_bytes(b"")
    results = load_screenshots(tmp_path)
    assert len(results) == 1


def _make_shot(seconds, tmp_path):
    from pathlib import Path
    from video_to_tutorial import _seconds_to_label
    return {"path": tmp_path / "shot.png", "seconds": float(seconds), "label": _seconds_to_label(seconds)}


def _seg(start, text):
    return {"start": float(start), "end": float(start + 5), "text": text}


def test_match_screenshots_includes_segments_within_30s(tmp_path):
    from video_to_tutorial import match_screenshots_to_transcript
    shot = _make_shot(60, tmp_path)
    segments = [_seg(30, "thirty"), _seg(90, "ninety")]  # |30-60|=30, |90-60|=30 — both on boundary
    result = match_screenshots_to_transcript([shot], segments)
    assert len(result) == 1
    assert "thirty" in result[0]["context"]
    assert "ninety" in result[0]["context"]


def test_match_screenshots_excludes_segments_beyond_30s(tmp_path):
    from video_to_tutorial import match_screenshots_to_transcript
    shot = _make_shot(60, tmp_path)
    segments = [_seg(29, "too early"), _seg(91, "too late")]  # |29-60|=31, |91-60|=31
    result = match_screenshots_to_transcript([shot], segments)
    assert result[0]["context"] == "(no spoken audio at this moment)"


def test_match_screenshots_concatenates_multiple_segments(tmp_path):
    from video_to_tutorial import match_screenshots_to_transcript
    shot = _make_shot(60, tmp_path)
    segments = [_seg(50, "first"), _seg(55, "second"), _seg(60, "third")]
    result = match_screenshots_to_transcript([shot], segments)
    assert result[0]["context"] == "first second third"


def test_match_screenshots_fallback_when_no_segments_match(tmp_path):
    from video_to_tutorial import match_screenshots_to_transcript
    shot = _make_shot(60, tmp_path)
    result = match_screenshots_to_transcript([shot], [])
    assert result[0]["context"] == "(no spoken audio at this moment)"


def test_match_screenshots_preserves_path_seconds_label(tmp_path):
    from video_to_tutorial import match_screenshots_to_transcript
    shot = _make_shot(90, tmp_path)
    result = match_screenshots_to_transcript([shot], [])
    assert result[0]["path"] == shot["path"]
    assert result[0]["seconds"] == 90.0
    assert result[0]["label"] == "1m 30s"


def test_match_screenshots_empty_inputs(tmp_path):
    from video_to_tutorial import match_screenshots_to_transcript
    assert match_screenshots_to_transcript([], []) == []


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
