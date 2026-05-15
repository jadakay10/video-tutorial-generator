# Screenshot Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional screenshot drop zone to the web UI that replaces ffmpeg frame extraction when timestamp-named images are provided.

**Architecture:** `/upload` accepts optional `screenshots` files alongside the video, saves them to `uploads/<job_id>/screenshots/`, and passes a `screenshots_dir` path to `_run_pipeline`. The pipeline calls `load_screenshots()` when that dir is provided; if it returns results, step 2 is skipped via a `{"skipped": true, "step": 2}` SSE event; otherwise it falls back to ffmpeg extraction.

**Tech Stack:** Flask (multipart file upload), existing `load_screenshots()` in `video_to_tutorial.py`, vanilla JS (second drop zone + `skipped` SSE event handling)

---

## File Map

| Action | Path | Change |
|--------|------|--------|
| Modify | `app.py` | Import `load_screenshots`, update `_run_pipeline` signature, save screenshots in `/upload`, pass `screenshots_dir` to thread |
| Modify | `templates/index.html` | Second drop zone, count badge, `skipped` step CSS + JS, clear on reset |
| Modify | `tests/test_app.py` | Add test for screenshot upload path |

---

## Task 1: Update app.py to handle screenshots

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_app.py` after `test_upload_accepts_mp4`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```
cd c:\Users\jendicott\Documents\GitHub\video-tutorial-generator && venv\Scripts\pytest tests/test_app.py::test_upload_accepts_screenshots -v
```

Expected: FAIL — `screenshots` dir not created yet.

- [ ] **Step 3: Update app.py imports and _run_pipeline**

Replace the import block and `_run_pipeline` function in `app.py`:

```python
from video_to_tutorial import (
    extract_audio,
    extract_frames,
    load_screenshots,
    sample_frames,
    transcribe_audio,
    encode_frames,
    generate_tutorial,
)
```

Replace `_run_pipeline`:

```python
def _run_pipeline(job_id: str, video_path: Path, screenshots_dir: Path | None = None) -> None:
    q = _jobs[job_id]
    try:
        audio_path = video_path.parent / (video_path.stem + ".mp3")

        q.put({"step": 1, "label": "Extracting audio"})
        extract_audio(video_path, audio_path)

        if screenshots_dir is not None:
            user_frames = load_screenshots(screenshots_dir)
            frame_files = [s["path"] for s in user_frames]
        else:
            frame_files = []

        if frame_files:
            q.put({"skipped": True, "step": 2, "label": "Extracting frames"})
        else:
            frames_dir = video_path.parent / (video_path.stem + "_frames")
            q.put({"step": 2, "label": "Extracting frames"})
            frame_files = extract_frames(video_path, frames_dir)

        q.put({"step": 3, "label": "Transcribing with Whisper"})
        transcript, segments = transcribe_audio(audio_path)

        q.put({"step": 4, "label": "Encoding frames"})
        frame_files = sample_frames(frame_files)
        image_blocks = encode_frames(frame_files)

        q.put({"step": 5, "label": "Generating tutorial with Claude"})
        tutorial = generate_tutorial(transcript, image_blocks, segments)

        markdown = f"## Transcript\n\n{transcript}\n\n---\n\n{tutorial}"
        q.put({
            "done": True,
            "markdown": markdown,
            "transcript": transcript,
            "tutorial": tutorial,
            "filename": video_path.stem,
        })
    except Exception as exc:
        q.put({"error": str(exc)})
    finally:
        if _active_job and _active_job[0] == job_id:
            _active_job.clear()
```

- [ ] **Step 4: Update /upload to save screenshots and pass screenshots_dir**

Replace the `upload()` route in `app.py`:

```python
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file provided."}), 400
    if not file.filename.lower().endswith(".mp4"):
        return jsonify({"error": "Only .mp4 files are supported."}), 400

    screenshot_files = request.files.getlist("screenshots")

    with _lock:
        if _active_job:
            return jsonify({"error": "A job is already running. Please wait."}), 409

        job_id = str(uuid.uuid4())
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(parents=True)
        video_path = job_dir / secure_filename(file.filename)
        file.save(video_path)

        screenshots_dir = None
        if screenshot_files:
            screenshots_dir = job_dir / "screenshots"
            screenshots_dir.mkdir()
            for sf in screenshot_files:
                name = secure_filename(sf.filename)
                if name:
                    sf.save(screenshots_dir / name)

        _jobs[job_id] = queue.Queue()
        _active_job.append(job_id)

    try:
        thread = threading.Thread(
            target=_run_pipeline,
            args=(job_id, video_path, screenshots_dir),
            daemon=True,
        )
        thread.start()
    except Exception:
        with _lock:
            if _active_job and _active_job[0] == job_id:
                _active_job.clear()
        return jsonify({"error": "Failed to start processing thread."}), 500

    return jsonify({"job_id": job_id})
```

- [ ] **Step 5: Run all tests — all must pass**

```
cd c:\Users\jendicott\Documents\GitHub\video-tutorial-generator && venv\Scripts\pytest tests/ -v
```

Expected: 16 passed (10 existing + 1 new app test + 5 pipeline tests).

- [ ] **Step 6: Commit**

```
cd c:\Users\jendicott\Documents\GitHub\video-tutorial-generator && git add app.py tests/test_app.py && git commit -m "feat: accept screenshots in /upload, skip frame extraction when provided"
```

---

## Task 2: Update index.html with screenshot drop zone and skipped-step UI

**Files:**
- Modify: `templates/index.html`

*(UI changes — verified manually after Task 1 is complete)*

- [ ] **Step 1: Add CSS for the screenshot zone and skipped step**

In `templates/index.html`, add the following inside the `<style>` block, after the `.step.done` rule:

```css
    .step.skipped { color: #aaa; }

    #screenshot-zone {
      width: 100%;
      max-width: 560px;
      margin-top: 16px;
      border: 1px dashed #ccc;
      border-radius: 12px;
      padding: 20px 24px;
      background: #fff;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
    }
    #screenshot-zone.drag-over { border-color: #4f46e5; background: #eef2ff; }
    #screenshot-zone.disabled { opacity: 0.5; pointer-events: none; }
    #screenshot-zone p { color: #777; font-size: 0.82rem; margin-top: 4px; }
    #screenshot-zone code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
    #screenshot-input { display: none; }
    #screenshot-count {
      display: inline-block;
      margin-top: 8px;
      font-size: 0.82rem;
      color: #4f46e5;
      font-weight: 600;
    }
```

- [ ] **Step 2: Add the screenshot drop zone to the HTML body**

In `templates/index.html`, add the screenshot zone after `<div id="error-msg"></div>`:

```html
  <div id="screenshot-zone">
    <strong>Screenshots (optional)</strong>
    <p>Name files as <code>01-30.png</code> or <code>00-01-30.jpg</code> (timestamp format)</p>
    <input type="file" id="screenshot-input" accept=".png,.jpg,.jpeg" multiple />
    <div id="screenshot-count"></div>
  </div>
```

- [ ] **Step 3: Update the JavaScript**

In `templates/index.html`, make the following JS changes:

**3a. Add `screenshotZone`, `screenshotInput`, `screenshotCount` constants** after the existing DOM constants:

```javascript
    const screenshotZone = document.getElementById('screenshot-zone');
    const screenshotInput = document.getElementById('screenshot-input');
    const screenshotCount = document.getElementById('screenshot-count');
```

**3b. Update `setStepState` to handle `'skipped'`** — replace the existing function:

```javascript
    function setStepState(stepNum, state) {
      const el = document.querySelector(`.step[data-step="${stepNum}"]`);
      if (!el) return;
      el.className = 'step ' + state;
      el.querySelector('.step-icon').textContent =
        state === 'done' ? '✓' : state === 'active' ? '▶' : state === 'skipped' ? '—' : '○';
    }
```

**3c. Update `listenForProgress` to handle the `skipped` SSE event** — replace the existing function:

```javascript
    function listenForProgress(jobId, filename) {
      const es = new EventSource('/progress/' + jobId);
      es.onmessage = (e) => {
        const event = JSON.parse(e.data);
        if (event.skipped) {
          setStepState(event.step, 'skipped');
          progressFill.style.width = (event.step / 5 * 100) + '%';
        } else if (event.step) {
          setProgress(event.step);
        } else if (event.done) {
          es.close();
          for (let i = 1; i <= 5; i++) {
            const el = document.querySelector(`.step[data-step="${i}"]`);
            if (el && !el.classList.contains('skipped')) setStepState(i, 'done');
          }
          progressFill.style.width = '100%';
          showResult(event.markdown, event.transcript, event.tutorial, event.filename);
          dropZone.classList.remove('disabled');
          screenshotZone.classList.remove('disabled');
          screenshotInput.value = '';
          screenshotCount.textContent = '';
        } else if (event.error) {
          es.close();
          handleError(event.error);
        }
      };
      es.onerror = () => { es.close(); handleError('Connection lost. Please try again.'); };
    }
```

**3d. Update `startUpload` to include screenshots in FormData and disable the screenshot zone** — replace the existing function:

```javascript
    function startUpload(file) {
      clearError();
      if (!file.name.toLowerCase().endsWith('.mp4')) {
        showError('Only .mp4 files are supported.');
        return;
      }

      dropZone.classList.add('disabled');
      screenshotZone.classList.add('disabled');
      progressSection.style.display = 'block';
      resultSection.style.display = 'none';
      setProgress(1);

      const formData = new FormData();
      formData.append('file', file);
      for (const f of screenshotInput.files) {
        if (f.name.toLowerCase().match(/\.(png|jpg|jpeg)$/)) {
          formData.append('screenshots', f);
        }
      }

      fetch('/upload', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
          if (data.error) { handleError(data.error); return; }
          listenForProgress(data.job_id, file.name.replace('.mp4', ''));
        })
        .catch(err => handleError(err.message));
    }
```

**3e. Update `handleError` to also re-enable the screenshot zone** — replace the existing function:

```javascript
    function handleError(msg) {
      progressSection.style.display = 'none';
      dropZone.classList.remove('disabled');
      screenshotZone.classList.remove('disabled');
      showError('Error: ' + msg);
    }
```

**3f. Wire up screenshot zone interactions** — add at the end of the script, after the existing drag-and-drop listeners:

```javascript
    // Screenshot zone
    screenshotZone.addEventListener('dragover', (e) => { e.preventDefault(); screenshotZone.classList.add('drag-over'); });
    screenshotZone.addEventListener('dragleave', () => screenshotZone.classList.remove('drag-over'));
    screenshotZone.addEventListener('drop', (e) => {
      e.preventDefault();
      screenshotZone.classList.remove('drag-over');
      const files = Array.from(e.dataTransfer.files).filter(f => f.name.toLowerCase().match(/\.(png|jpg|jpeg)$/));
      if (files.length) {
        const dt = new DataTransfer();
        files.forEach(f => dt.items.add(f));
        screenshotInput.files = dt.files;
        screenshotCount.textContent = `${files.length} file${files.length > 1 ? 's' : ''} selected`;
      }
    });
    screenshotZone.addEventListener('click', () => screenshotInput.click());
    screenshotInput.addEventListener('change', () => {
      const n = screenshotInput.files.length;
      screenshotCount.textContent = n ? `${n} file${n > 1 ? 's' : ''} selected` : '';
    });
```

- [ ] **Step 4: Manual verification**

Start the server:
```
cd c:\Users\jendicott\Documents\GitHub\video-tutorial-generator && venv\Scripts\python app.py
```

Open `http://localhost:5000` and verify:
1. Screenshot zone appears below the video drop zone
2. Clicking or dropping images onto it shows the file count badge
3. Drop a video with no screenshots → step 2 shows as active then done normally
4. Drop a video with timestamp-named screenshots → step 2 shows as greyed out with `—` icon
5. Drop a video with screenshots that have invalid names → step 2 runs normally (fallback)

- [ ] **Step 5: Commit**

```
cd c:\Users\jendicott\Documents\GitHub\video-tutorial-generator && git add templates/index.html && git commit -m "feat: add screenshot drop zone with skipped-step UI"
```

---

## Task 3: Push and update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Regenerate requirements.txt**

```
cd c:\Users\jendicott\Documents\GitHub\video-tutorial-generator && venv\Scripts\pip freeze > requirements.txt
```

- [ ] **Step 2: Run all tests one final time**

```
cd c:\Users\jendicott\Documents\GitHub\video-tutorial-generator && venv\Scripts\pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit and push**

```
cd c:\Users\jendicott\Documents\GitHub\video-tutorial-generator && git add requirements.txt && git commit -m "chore: update requirements.txt" && git push origin main
```
