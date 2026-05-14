import sys
import os
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


def transcribe_audio(audio_path: Path) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


def encode_frames(frame_files: list[Path]) -> list[dict]:
    blocks = []
    for frame_file in frame_files:
        data = base64.standard_b64encode(frame_file.read_bytes()).decode("utf-8")
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
            }
        )
    return blocks


def generate_tutorial(transcript: str, image_blocks: list[dict]) -> str:
    client = anthropic.Anthropic()
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
    frames_dir = Path("frames")
    output_path = Path(f"{video_stem}.md")

    print(f"[1/5] Extracting audio from '{video_path.name}'...")
    extract_audio(video_path, audio_path)
    print(f"      Audio saved to '{audio_path.name}'.")

    print("[2/5] Extracting one frame every 5 seconds...")
    frame_files = extract_frames(video_path, frames_dir)
    print(f"      Extracted {len(frame_files)} frames into '{frames_dir}/'.")

    print("[3/5] Transcribing audio with Whisper (base model)...")
    transcript = transcribe_audio(audio_path)
    print(f"      Transcription complete ({len(transcript)} characters).")

    print(f"[4/5] Encoding {len(frame_files)} frames for the API...")
    image_blocks = encode_frames(frame_files)
    print("      Frames encoded.")

    print("[5/5] Sending to Claude to generate tutorial (streaming)...\n")
    print("─" * 60)
    tutorial = generate_tutorial(transcript, image_blocks)
    print("\n" + "─" * 60)

    markdown_output = f"## Transcript\n\n{transcript}\n\n---\n\n{tutorial}"
    output_path.write_text(markdown_output, encoding="utf-8")
    print(f"\nTutorial saved to '{output_path}'.")


if __name__ == "__main__":
    main()
