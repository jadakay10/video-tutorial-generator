# Screenshot Upload Design

**Date:** 2026-05-14
**Status:** Approved

## Overview

Add an optional second drop zone to the web UI that lets the user supply their own timestamped screenshots instead of relying on ffmpeg frame extraction. When screenshots are provided, the extraction step is skipped and `load_screenshots()` is used instead. The progress bar marks step 2 as skipped rather than hiding it.

## Goals

- Let users provide their own screenshots named with timestamps (`MM-SS` or `HH-MM-SS`)
- Skip ffmpeg frame extraction when screenshots are present
- Show step 2 as greyed out / skipped in the progress bar
- Fall back to auto-extraction if screenshots are uploaded but none have valid names

## Non-Goals

- Mixing user screenshots with auto-extracted frames
- Renaming or reprocessing uploaded screenshots
- Accepting screenshot folders via a filesystem path input

## Screenshot Naming Convention

Files must be named with timestamps so `load_screenshots()` can parse them:

| Format | Example | Parsed as |
|--------|---------|-----------|
| `MM-SS` | `01-30.png` | 1 minute 30 seconds |
| `HH-MM-SS` | `00-01-30.jpg` | 1 minute 30 seconds |

Files that don't match either format are skipped with a warning printed to the server log.

## File Map

| Action | Path | Change |
|--------|------|--------|
| Modify | `templates/index.html` | Add second drop zone, skipped-step style, screenshot file handling |
| Modify | `app.py` | Accept screenshots in `/upload`, emit `skipped` SSE event, call `load_screenshots()` |

`video_to_tutorial.py` and `load_screenshots()` are unchanged.

## Architecture

```
Browser
  POST /upload  (multipart)
    - file: video.mp4
    - screenshots[]: 01-30.png, 02-00.jpg, ...  (optional, multiple)

Flask /upload
  → saves video to uploads/<job_id>/video.mp4
  → saves screenshots to uploads/<job_id>/screenshots/
  → passes has_screenshots flag to pipeline thread

_run_pipeline
  → step 1: extract audio (always)
  → step 2: if has_screenshots → load_screenshots(), emit {skipped: true, step: 2}
             else → extract_frames() as today
  → steps 3-5: unchanged
```

## SSE Events

New event type for step 2 when screenshots are used:

```json
{"skipped": true, "step": 2, "label": "Extracting frames"}
```

The browser marks step 2 with a `skipped` CSS class (grey text, `—` icon) and advances the progress bar past it.

## Fallback Behaviour

If screenshots are uploaded but `load_screenshots()` returns an empty list (no valid names), the pipeline falls back to ffmpeg frame extraction and emits a normal step 2 event. No error is shown to the user.

## UI Changes

- Second drop zone below the video zone, labelled "Screenshots (optional)"
- Subtitle: "Name files as `01-30.png` or `00-01-30.jpg` (timestamp format)"
- Shows a file count badge when files are selected (`3 files selected`)
- Cleared automatically when the video drop zone resets after a job completes
- Accepts `.png`, `.jpg`, `.jpeg` only; silently ignores other file types client-side

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Screenshots uploaded with no valid names | Fall back to auto-extraction silently |
| Screenshot file unreadable | `load_screenshots()` skips it with a server-side warning |
| No screenshots provided | Pipeline runs as today — no change |
