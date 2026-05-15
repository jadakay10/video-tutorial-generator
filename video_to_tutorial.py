import sys
import os
import re
import subprocess
import base64
import glob
from pathlib import Path

from dotenv import load_dotenv
import whisper
import anthropic

load_dotenv()


def _run_ffmpeg(args: list[str], step_label: str) -> None:
    try:
        subprocess.run(["ffmpeg"] + args + ["-y"], check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed during {step_label}:\n{e.stderr.decode()}")


def extract_audio(video_path: Path, audio_path: Path) -> None:
    _run_ffmpeg(
        ["-i", str(video_path), "-vn", "-acodec", "libmp3lame", str(audio_path)],
        "audio extraction",
    )


def extract_frames(video_path: Path, frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        ["-i", str(video_path), "-vf", "fps=1/5", str(frames_dir / "frame_%04d.jpg")],
        "frame extraction",
    )
    return sorted(frames_dir.glob("frame_*.jpg"))


def transcribe_audio(audio_path: Path) -> tuple[str, list[dict]]:
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path), verbose=False)
    text = result["text"].strip()
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    return text, segments


MAX_FRAMES = 80


def sample_frames(frame_files: list[Path]) -> list[Path]:
    if len(frame_files) <= MAX_FRAMES:
        return frame_files
    step = len(frame_files) / MAX_FRAMES
    return [frame_files[int(i * step)] for i in range(MAX_FRAMES)]


_MM_SS = re.compile(r"^(\d{2})-(\d{2})$")
_HH_MM_SS = re.compile(r"^(\d{2})-(\d{2})-(\d{2})$")


def _seconds_to_label(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def load_screenshots(folder: Path) -> list[dict]:
    results = []
    for path in folder.iterdir():
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        stem = path.stem
        m = _HH_MM_SS.match(stem)
        if m:
            h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            seconds = float(h * 3600 + mn * 60 + s)
        else:
            m = _MM_SS.match(stem)
            if m:
                mn, s = int(m.group(1)), int(m.group(2))
                seconds = float(mn * 60 + s)
            else:
                print(f"Warning: skipping '{path.name}' — name does not match MM-SS or HH-MM-SS format.")
                continue
        results.append({"path": path, "seconds": seconds, "label": _seconds_to_label(seconds)})
    return sorted(results, key=lambda x: x["seconds"])


def match_screenshots_to_transcript(
    screenshots: list[dict], segments: list[dict]
) -> list[dict]:
    result = []
    for shot in screenshots:
        ts = shot["seconds"]
        matching = [s for s in segments if abs(s["start"] - ts) <= 30]
        context = (
            " ".join(s["text"] for s in matching).strip()
            if matching
            else "(no spoken audio at this moment)"
        )
        result.append({
            "path": shot["path"],
            "seconds": shot["seconds"],
            "label": shot["label"],
            "context": context,
        })
    return result


def encode_frames(frame_files: list[Path]) -> list[dict]:
    blocks = []
    for frame_file in frame_files:
        try:
            raw = frame_file.read_bytes()
        except OSError as e:
            raise RuntimeError(f"Could not read frame '{frame_file}': {e}")
        data = base64.standard_b64encode(raw).decode("utf-8")
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
            }
        )
    return blocks


def _image_block_from_path(path: Path) -> dict:
    ext = path.suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    try:
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    except OSError as e:
        raise RuntimeError(f"Could not read screenshot '{path}': {e}")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def generate_tutorial(
    transcript: str,
    image_blocks: list[dict],
    segments: list[dict] | None = None,
    paired_screenshots: list[dict] | None = None,
) -> str:
    client = anthropic.Anthropic()

    if paired_screenshots is not None:
        content: list[dict] = []
        for shot in paired_screenshots:
            content.append({
                "type": "text",
                "text": f'Screenshot at {shot["label"]} — what was being said: "{shot["context"]}"',
            })
            content.append(_image_block_from_path(shot["path"]))
        prompt_text = (
            f"Above are screenshots paired with the spoken audio at each moment.\n\n"
            f"**Full transcript for reference:**\n{transcript}\n\n"
            f"Using the screenshots and their paired spoken context above, write a detailed "
            f"step-by-step tutorial in Markdown format. Each image is paired with what was "
            f"being said at that moment — use both the visual and the spoken context together "
            f"when writing each step. The tutorial should:\n"
            f"- Start with a clear `# Title`\n"
            f"- Include a short introduction explaining what the viewer will learn\n"
            f"- List any prerequisites or tools needed\n"
            f"- Number every step clearly, describing what is shown and what was being said\n"
            f"- Highlight important tips, warnings, or common mistakes\n"
            f"- End with a brief summary or next steps"
        )
        content.append({"type": "text", "text": prompt_text})
    else:
        prompt_text = (
            f"Below are frames captured every 5 seconds from a video, followed by the "
            f"full audio transcript.\n\n"
            f"**Transcript:**\n{transcript}\n\n"
            f"Using the visual frames and the transcript above, write a detailed "
            f"step-by-step tutorial in Markdown format. The tutorial should:\n"
            f"- Start with a clear `# Title`\n"
            f"- Include a short introduction explaining what the viewer will learn\n"
            f"- List any prerequisites or tools needed\n"
            f"- Number every step clearly and describe what is shown on screen\n"
            f"- Reference relevant frames where helpful (e.g., 'As shown at ~10s...')\n"
            f"- Highlight important tips, warnings, or common mistakes\n"
            f"- End with a brief summary or next steps"
        )
        content = image_blocks + [{"type": "text", "text": prompt_text}]

    parts: list[str] = []
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        for text in stream.text_stream:
            parts.append(text)
    return "".join(parts)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python video_to_tutorial.py <video.mp4>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: '{video_path}' does not exist.")
        sys.exit(1)

    video_stem = video_path.stem
    audio_path = video_path.parent / f"{video_stem}.mp3"
    frames_dir = video_path.parent / f"{video_stem}_frames"
    output_path = Path(f"{video_stem}.md")

    print(f"[1/5] Extracting audio from '{video_path.name}'...")
    extract_audio(video_path, audio_path)
    print(f"      Audio saved to '{audio_path.name}'.")

    print("[2/5] Extracting one frame every 5 seconds...")
    frame_files = extract_frames(video_path, frames_dir)
    print(f"      Extracted {len(frame_files)} frames into '{frames_dir}/'.")

    print("[3/5] Transcribing audio with Whisper (base model)...")
    transcript, segments = transcribe_audio(audio_path)
    print(f"      Transcription complete ({len(transcript)} characters, {len(segments)} segments).")

    frame_files = sample_frames(frame_files)
    print(f"[4/5] Encoding {len(frame_files)} frames for the API...")
    image_blocks = encode_frames(frame_files)
    print("      Frames encoded.")

    print("[5/5] Sending to Claude to generate tutorial (streaming)...\n")
    print("─" * 60)
    tutorial = generate_tutorial(transcript, image_blocks, segments)
    print("\n" + "─" * 60)

    if not tutorial.strip():
        print("Error: Claude returned an empty response. The output file was not written.")
        sys.exit(1)

    markdown_output = f"## Transcript\n\n{transcript}\n\n---\n\n{tutorial}"
    output_path.write_text(markdown_output, encoding="utf-8")
    print(f"\nTutorial saved to '{output_path}'.")


if __name__ == "__main__":
    main()
