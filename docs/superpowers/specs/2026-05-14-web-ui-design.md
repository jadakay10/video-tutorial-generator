# Web UI Design — Video Tutorial Generator

**Date:** 2026-05-14
**Status:** Approved

## Overview

Add a local web UI to the video tutorial generator so the user can drag and drop an `.mp4` file into a browser page instead of using the command line. The UI shows a step-by-step progress bar as the pipeline runs, then renders the finished tutorial as formatted HTML with a Markdown download button.

## Goals

- Replace the CLI workflow with a drag-and-drop browser page for local use
- Show real-time progress for each of the 5 pipeline steps
- Display the generated tutorial rendered as HTML in the browser
- Offer a button to download the tutorial as a `.md` file
- Support one upload at a time

## Non-Goals

- Multi-user or networked access
- Authentication
- Storing past results
- Changing the underlying pipeline (Whisper, ffmpeg, Claude)

## Files

### New files

| File | Purpose |
|---|---|
| `app.py` | Flask server — upload endpoint, SSE progress stream, page serving |
| `templates/index.html` | Single-page UI — drop zone, progress bar, tutorial output |

### Modified files

| File | Change |
|---|---|
| `video_to_tutorial.py` | Refactor pipeline steps into importable functions so `app.py` can call them directly. CLI `main()` entry point stays intact. |
| `requirements.txt` | Add `flask`, `markdown` |

## Architecture

```
Browser
  │
  ├── POST /upload         → Flask saves .mp4 to uploads/, starts background thread, returns job_id
  ├── GET  /progress/<id>  → SSE stream — thread pushes step events + final done/error event
  └── GET  /              → Serves index.html
```

The pipeline runs in a `threading.Thread`. Progress events are put into a `queue.Queue` per job. The `/progress/<id>` SSE route reads from that queue and yields events to the browser until it receives `done` or `error`.

## Data Flow

1. User drops `.mp4` → browser POSTs to `/upload`
2. Flask saves file to `uploads/<job_id>/video.mp4`, starts background thread, returns `{"job_id": "..."}`
3. Browser opens `/progress/<job_id>` SSE stream
4. Thread runs pipeline steps in order, pushing `{"step": N, "label": "..."}` events after each step
5. On completion, thread pushes `{"done": true, "markdown": "..."}` event
6. Browser closes SSE, renders Markdown as HTML, shows download button
7. On failure, thread pushes `{"error": "..."}` event; UI shows message and resets drop zone

## Pipeline Steps (SSE labels)

| Step | Label |
|---|---|
| 1 | Extracting audio |
| 2 | Extracting frames |
| 3 | Transcribing with Whisper |
| 4 | Encoding frames |
| 5 | Generating tutorial with Claude |

## UI Behaviour

- Drop zone accepts `.mp4` only; rejects other file types immediately with an inline error
- While a job is running the drop zone is disabled
- Progress bar advances one segment per completed step (5 segments total)
- When `done` arrives, progress bar fills completely and the tutorial section appears below
- Tutorial is rendered from Markdown to HTML using the `marked.js` CDN library (no server-side rendering needed)
- Download button triggers a browser download of the raw Markdown as `<video-name>.md`
- If an error event arrives, a red error message replaces the progress bar and the drop zone resets

## Error Handling

| Scenario | Behaviour |
|---|---|
| Non-.mp4 file dropped | Rejected on the client before upload; inline error message shown |
| ffmpeg not found | Thread catches `FileNotFoundError`, pushes error event |
| Whisper / API failure | Thread catches exception, pushes error event with message |
| Job already running | Drop zone disabled until current job completes |

## Dependencies to Add

```
flask
```

`marked.js` is loaded from CDN in `index.html` — no install needed.

## File Cleanup

Uploaded videos and extracted frames are stored under `uploads/<job_id>/` for the duration of the job. They are not automatically deleted (out of scope for v1) — the user can manually clear the folder.
