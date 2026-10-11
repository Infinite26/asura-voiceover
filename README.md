# Asura Voiceover — comic chapter → narrated video

Give it a comic chapter URL and it produces a **voiced video** of that chapter —
no manual editing.

```
chapter URL ─► scrape panels ─► OCR each panel (tesseract)
            ─► LLM writes a narration script ─► TTS per panel (edge-tts)
            ─► ffmpeg builds one video, audio synced to the panels
```

## Requirements

- Python 3.10+ with `edge-tts` (installed in `.venv`)
- `tesseract-ocr`, `ffmpeg`, and `curl` on `PATH`
- An LLM endpoint for the narration step (the script calls the local OpenCode
  server by default — point it at any OpenAI-compatible API)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install edge-tts
# system deps (Arch):
sudo pacman -S tesseract tesseract-data-eng ffmpeg
```

## Usage

Run the whole pipeline:

```bash
./run_detached.sh <chapter_url>          # defaults to the full "video" step
# or a single step:
python asura_voiceover.py run    <chapter_url>
python asura_voiceover.py scrape <chapter_url>
python asura_voiceover.py script <chapter_url>
python asura_voiceover.py tts    <chapter_url>
python asura_voiceover.py video  <chapter_url>
```

Add `--voice en-US-AndrewNeural` to pick a different voice.

`run_detached.sh` / `run_video.sh` run the job **detached from the agent's
cgroup and memory-capped**, so a long build can't get the session OOM-killed.

`work/` and `.venv/` are git-ignored.
