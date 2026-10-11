#!/usr/bin/env python3
"""Asura Scans -> voiced comic video pipeline.

Given an Asura Scans chapter URL it:
  1. scrapes the chapter panel images
  2. OCRs every panel (tesseract)
  3. asks OpenCode (deepseek-v4.1-flash) to write a narration script
  4. synthesizes TTS per panel (edge-tts)
  5. builds one video with the audio synced to the panels (ffmpeg)

Usage:
  asura_voiceover.py run     <chapter_url> [--voice en-US-AndrewNeural]
  asura_voiceover.py scrape  <chapter_url>
  asura_voiceover.py script  <chapter_url>
  asura_voiceover.py tts     <chapter_url>
  asura_voiceover.py video   <chapter_url>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import edge_tts  # only available under ./.venv/bin/python
except ImportError:  # noqa: BLE001
    edge_tts = None

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
MODEL = os.environ.get("AV_MODEL", "opencode-go/deepseek-v4.1-flash")
OPENCODE = shutil.which("opencode") or "opencode"
TESSERACT = shutil.which("tesseract") or "tesseract"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
EDGE_TTS = str(ROOT / ".venv" / "bin" / "edge-tts")
if not Path(EDGE_TTS).exists():
    EDGE_TTS = shutil.which("edge-tts") or "edge-tts"

W, H = 1080, 1920
FPS = 60            # overridden at startup to divide the display refresh rate (see pick_fps)
GAP = 0.35          # silence padding after each panel (seconds)
DEFAULT_HOLD = 1.6  # hold time for a panel with no narration
MAX_SCROLL = 900    # max vertical pan speed (px/s at 1080 width) for tall webtoon strips


def pick_fps() -> int:
    """Pick a frame rate that divides the monitor's refresh rate so pans don't judder.

    30 fps on a 144 Hz panel (144/30 = 4.8) judders badly; 72 or 48 divide it exactly.
    """
    env = os.environ.get("AV_FPS")
    if env:
        return int(env)
    refresh = 0
    try:
        out = subprocess.run(["hyprctl", "monitors"], capture_output=True, text=True, timeout=3).stdout
        for m in re.finditer(r"@([0-9.]+)Hz", out):
            refresh = max(refresh, int(round(float(m.group(1)))))
    except Exception:  # noqa: BLE001
        pass
    if refresh:
        for r in (72, 60, 48, 30, 24):
            if refresh % r == 0:
                return r
    return 60


def log(msg: str) -> None:
    print(f"[asura] {msg}", flush=True)


def http_get(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://asurascans.com/"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_chapter(url: str) -> dict:
    m = re.search(r"/comics/([^/]+)/chapter/([0-9.]+)", url)
    if not m:
        m2 = re.search(r"/comics/([^/?#]+)", url)
        if not m2:
            sys.exit(f"Not a chapter URL: {url}")
        sys.exit(f"URL must be a chapter URL (…/chapter/N): {url}")
    slug, chapter = m.group(1), m.group(2)
    html = http_get(url).decode("utf-8", "ignore")
    title = re.search(r"<title>([^<]+)</title>", html)
    title = title.group(1).split("|")[0].strip() if title else slug
    imgs = re.findall(
        r'https://cdn\.asurascans\.com/asura-images/chapters/[^"\' <>\\]+?\.(?:webp|jpg|jpeg|png)',
        html, re.I)
    seen, panel_urls = set(), []
    for u in imgs:
        if u not in seen:
            seen.add(u)
            panel_urls.append(u)
    if not panel_urls:
        sys.exit("No chapter images found (site layout may have changed).")
    return {"slug": slug, "chapter": chapter, "title": title, "url": url,
            "images": panel_urls,
            "workdir": str(WORK / slug / f"chapter_{chapter}")}


def load_meta(url: str) -> dict:
    meta_file = None
    slug = re.search(r"/comics/([^/?#]+)", url).group(1)
    chapter = re.search(r"/chapter/([0-9.]+)", url)
    chapter = chapter.group(1) if chapter else "1"
    meta_file = WORK / slug / f"chapter_{chapter}" / "meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text())
    meta = parse_chapter(url)
    Path(meta["workdir"]).mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------- scrape
def cmd_scrape(url: str) -> dict:
    meta = parse_chapter(url)
    wd = Path(meta["workdir"])
    (wd / "images").mkdir(parents=True, exist_ok=True)
    ext = {"webp": ".webp", "jpg": ".jpg", "jpeg": ".jpg", "png": ".png"}
    for i, u in enumerate(meta["images"], 1):
        e = ext.get(u.rsplit(".", 1)[-1].lower(), ".img")
        out = wd / "images" / f"{i:03d}{e}"
        if out.exists() and out.stat().st_size > 0:
            continue
        try:
            out.write_bytes(http_get(u))
            log(f"downloaded panel {i:03d} ({out.stat().st_size//1024} KiB)")
        except Exception as exc:  # noqa: BLE001
            log(f"WARN panel {i:03d} failed: {exc}")
    meta["workdir"] = str(wd)
    (wd / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"scraped {len(list((wd/'images').glob('*')))} panels -> {wd/'images'}")
    return meta


# ---------------------------------------------------------------- OCR
def ocr_panels(meta: dict) -> list[dict]:
    wd = Path(meta["workdir"])
    ocr_file = wd / "ocr.json"
    imgs = sorted(p for p in (wd / "images").glob("*") if p.is_file())
    out = []
    for i, p in enumerate(imgs, 1):
        try:
            txt = subprocess.run([TESSERACT, str(p), "stdout"],
                                 capture_output=True, text=True, timeout=120).stdout
        except Exception as exc:  # noqa: BLE001
            txt = ""
            log(f"WARN ocr panel {i}: {exc}")
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in txt.splitlines()]
        lines = [ln for ln in lines
                 if len(re.sub(r"[^A-Za-z0-9]", "", ln)) >= 2
                 and sum(c.isalnum() for c in ln) / max(len(ln), 1) > 0.5]
        out.append({"index": i, "image": str(p), "ocr": "\n".join(lines)})
    ocr_file.write_text(json.dumps(out, indent=2))
    log(f"OCR'd {len(out)} panels -> {ocr_file}")
    return out


# ---------------------------------------------------------------- script (LLM)
SCRIPT_PROMPT = """You are a professional comic voice-over scriptwriter.
Each numbered item below is one tall vertical strip of a webtoon chapter. The OCR text for a \
strip is in reading order and mixes speech-bubble dialogue, sound effects, and UI/screen text.

Write the narration a single voice actor reads aloud while the strip scrolls by. Rules:
- Exactly one entry per strip index, in order, covering everything that happens in that strip.
- Turn dialogue into spoken lines; keep it natural and dramatic but concise.
- Weave in brief atmospheric narration to connect the events.
- Strip length is proportional to how much happens in it: a short strip gets ~15-30 words, a
  long strip gets ~60-120 words. Stay within roughly one word per 65 vertical pixels of strip.
- Ignore unreadable OCR garbage, translator credits, watermarks and page numbers.
- Never mention "panel", "OCR", "image" or the reader.
- Output STRICT JSON only, no markdown, exactly this schema:
{"panels":[{"index":1,"text":"..."}, ...]}
"""


def run_opencode(prompt: str, attach: list[str] | None = None) -> str:
    cmd = [OPENCODE, "run", "--model", MODEL]
    for f in attach or []:
        cmd += ["-f", f]
    cmd.append(prompt)
    log(f"asking {MODEL} ...")
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=900)
    if p.returncode != 0:
        log(p.stderr[-2000:])
        raise RuntimeError(f"opencode run failed ({p.returncode})")
    return p.stdout


def extract_json(text: str):
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def cmd_script(url: str) -> list[dict]:
    meta = load_meta(url)
    wd = Path(meta["workdir"])
    ocr_file = wd / "ocr.json"
    ocr = json.loads(ocr_file.read_text()) if ocr_file.exists() else ocr_panels(meta)

    payload = "\n\n".join(f"### PANEL {p['index']}\n{p['ocr'] or '(no readable text)'}"
                          for p in ocr)
    prompt = (f"Series: {meta['title']}\nChapter: {meta['chapter']}\n\n"
              f"{SCRIPT_PROMPT}\n\n---- OCR ----\n{payload}\n---- END ----")
    (wd / "llm_prompt.txt").write_text(prompt)

    raw = run_opencode(prompt)
    try:
        data = extract_json(raw)
    except Exception:  # noqa: BLE001 retry once with a stricter nudge
        raw = run_opencode(prompt + "\n\nRespond with ONLY the raw JSON object.")
        data = extract_json(raw)

    by_idx = {}
    for e in data.get("panels", []):
        try:
            by_idx[int(e["index"])] = e
        except (KeyError, TypeError, ValueError):
            continue
    script = []
    for p in ocr:
        e = by_idx.get(p["index"], {})
        script.append({
            "index": p["index"],
            "image": p["image"],
            "speaker": (e.get("speaker") or "Narrator").strip(),
            "text": (e.get("text") or "").strip(),
        })
    (wd / "script.json").write_text(json.dumps(script, indent=2))
    flat = "\n\n".join((f"{s['speaker']}: {s['text']}" if s["speaker"] != "Narrator"
                        else s["text"]) for s in script if s["text"])
    (wd / "script.txt").write_text(flat)
    log(f"script written ({sum(1 for s in script if s['text'])}/{len(script)} panels narrated) -> {wd/'script.txt'}")
    return script


# ---------------------------------------------------------------- TTS
VOICE_STYLES = {
    "Narrator": [],
    "Baron": ["--pitch=-8Hz", "--rate=-2%"],
    "Narrator2": ["--pitch=+10Hz"],
}


async def _synth_async(text: str, voice: str, media: Path) -> list[dict]:
    """Synthesize `text` to `media`, returning per-word timings (seconds).

    edge-tts now emits SentenceBoundary (not WordBoundary); per-word times are
    interpolated inside each sentence proportionally to word length.
    """
    comm = edge_tts.Communicate(text, voice)
    boundaries: list[dict] = []
    with open(media, "wb") as fh:
        async for ch in comm.stream():
            if ch["type"] == "audio":
                fh.write(ch["data"])
            elif ch["type"] in ("SentenceBoundary", "WordBoundary"):
                boundaries.append(ch)
    words: list[dict] = []
    for b in boundaries:
        start = b["offset"] / 1e7
        dur = b["duration"] / 1e7
        toks = [t for t in re.split(r"\s+", b["text"].strip()) if t]
        if not toks:
            continue
        weights = [len(t) + 1 for t in toks]
        total = sum(weights)
        t = start
        for tok, wgt in zip(toks, weights):
            span = dur * wgt / total
            words.append({"text": tok, "start": round(t, 3), "end": round(t + span, 3)})
            t += span
    return words


def synth(text: str, voice: str, media: Path) -> list[dict]:
    return asyncio.run(_synth_async(text, voice, media))


def cmd_tts(url: str, voice: str, force: bool = False) -> list[dict]:
    meta = load_meta(url)
    wd = Path(meta["workdir"])
    script = json.loads((wd / "script.json").read_text())
    adir = wd / "audio"
    wdir = wd / "words"
    adir.mkdir(exist_ok=True)
    wdir.mkdir(exist_ok=True)
    for s in script:
        out = adir / f"{s['index']:03d}.mp3"
        wjson = wdir / f"{s['index']:03d}.json"
        if s["text"] and (force or not out.exists() or not wjson.exists()):
            if edge_tts is None:
                # CLI fallback: audio only, no word timings
                cmd = [EDGE_TTS, "--voice", voice, "--text", s["text"],
                       "--write-media", str(out)] + VOICE_STYLES.get(s["speaker"], [])
                subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                wjson.write_text("[]")
            else:
                try:
                    words = synth(s["text"], voice, out)
                    wjson.write_text(json.dumps(words, indent=2))
                    log(f"tts+words panel {s['index']:03d} ({len(words)} words)")
                except Exception as exc:  # noqa: BLE001
                    log(f"WARN tts panel {s['index']}: {exc}")
        s["audio"] = str(out) if out.exists() else None
        s["duration"] = probe_duration(out) if out.exists() else 0.0
        s["words"] = str(wjson) if wjson.exists() else None
    (wd / "script.json").write_text(json.dumps(script, indent=2))
    log(f"TTS done -> {adir}")
    return script


def probe_duration(path: Path) -> float:
    try:
        p = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        return float(p.stdout.strip())
    except Exception:  # noqa: BLE001
        return 0.0


def probe_size(path: Path) -> tuple[int, int]:
    try:
        p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0",
                            str(path)], capture_output=True, text=True)
        w, h = p.stdout.strip().split(",")[:2]
        return int(w), int(h)
    except Exception:  # noqa: BLE001
        return W, H


def panel_duration(img: Path, audio_dur: float, has_text: bool) -> float:
    """Audio drives the clip; tall strips get extra time so the pan stays watchable."""
    if not has_text:
        return DEFAULT_HOLD
    dur = audio_dur + GAP
    w, h = probe_size(img)
    scaled_h = h * W / max(w, 1)
    if scaled_h > H:
        dur = max(dur, scaled_h / MAX_SCROLL)
    return dur


# ---------------------------------------------------------------- subtitles
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{fs},&H00FFFFFF,&H0000D7FF,&H00000000,&H78000000,-1,0,0,0,100,100,0,0,1,4,2,2,70,70,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def words_to_ass(words: list[dict], out: Path, max_words: int = 7,
                 max_chars: int = 40, fontsize: int = 58) -> Path | None:
    """Write an ASS file whose words light up (karaoke sweep) in time with the audio."""
    if not words:
        return None
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        cand = cur + [w]
        if cur and (len(cand) > max_words or
                    len(" ".join(x["text"] for x in cand)) > max_chars):
            lines.append(cur)
            cur = [w]
        else:
            cur = cand
    if cur:
        lines.append(cur)
    events = []
    for ln in lines:
        start, end = ln[0]["start"], ln[-1]["end"]
        parts = []
        for i, w in enumerate(ln):
            nxt = ln[i + 1]["start"] if i + 1 < len(ln) else w["end"]
            k = max(1, int(round((nxt - w["start"]) * 100)))
            parts.append("{\\kf%d}%s" % (k, w["text"]))
        events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
                      f"Default,,0,0,0,,{' '.join(parts)}")
    out.write_text(ASS_HEADER.format(w=W, h=H, fs=fontsize) + "\n".join(events) + "\n")
    return out


# ---------------------------------------------------------------- video
def build_clip(img: Path, audio: Path | None, dur: float, out: Path,
               subs: Path | None = None) -> None:
    dur = round(max(dur, 0.8), 3)
    iw, ih = probe_size(img)
    scaled_h = ih * W / max(iw, 1)
    if scaled_h > H + 2:
        # crop a 16:9 window out of the ORIGINAL image first, then upscale it.
        # (scaling the full 16000px strip first needs >2GB; this needs ~MBs)
        crop_h = max(2, int(round(iw * H / W)))
        vf = (f"crop={iw}:{crop_h}:0:'(ih-{crop_h})*min(1\\,t/{dur})',"
              f"scale={W}:{H}:flags=lanczos,fps={FPS},format=yuv420p,"
              f"tmix=frames=2:weights='1 2'")
    else:
        vf = (f"scale={W}:-2:flags=lanczos,"
              f"pad={W}:{H}:0:'(oh-ih)/2',"
              f"fps={FPS},format=yuv420p")
    if subs and subs.exists():
        vf += f",ass={subs}"
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-loop", "1", "-framerate", str(FPS), "-t", str(dur), "-i", str(img)]
    has_audio = bool(audio and audio.exists())
    if has_audio:
        cmd += ["-i", str(audio), "-af", "apad", "-shortest"]
    else:
        cmd += ["-f", "lavfi", "-t", str(dur), "-i", "anullsrc=r=24000:cl=mono"]
    cmd += ["-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", "-map", "1:a",
            "-threads", "2", "-filter_threads", "1",
            "-c:v", "libx264", "-preset", "superfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "160k", "-ar", "24000", "-ac", "1",
            "-t", str(dur), str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg clip failed for {img.name}: {r.stderr[-500:]}")


def cmd_video(url: str, voice: str) -> Path:
    meta = load_meta(url)
    wd = Path(meta["workdir"])
    script_path = wd / "script.json"
    if not script_path.exists():
        cmd_script(url)
    script = json.loads(script_path.read_text())
    need_tts = any(not s.get("audio") for s in script)
    if need_tts:
        script = cmd_tts(url, voice)

    clips_dir = wd / "clips"
    clips_dir.mkdir(exist_ok=True)
    subs_dir = wd / "subs"
    subs_dir.mkdir(exist_ok=True)
    list_file = wd / "concat.txt"
    lines = []
    for s in script:
        img = Path(s["image"])
        audio = Path(s["audio"]) if s.get("audio") else None
        dur = panel_duration(img, s.get("duration") or 0.0, bool(s.get("text")))
        clip = clips_dir / f"{s['index']:03d}.mp4"
        subs = None
        wjson = Path(s["words"]) if s.get("words") else None
        if wjson and wjson.exists():
            try:
                words = json.loads(wjson.read_text())
                subs = words_to_ass(words, subs_dir / f"{s['index']:03d}.ass")
            except Exception as exc:  # noqa: BLE001
                log(f"WARN subs panel {s['index']}: {exc}")
        if not clip.exists():
            build_clip(img, audio, dur, clip, subs=subs)
            log(f"clip {s['index']:03d} ({dur:.1f}s){' +subs' if subs else ''}")
        lines.append(f"file '{clip}'")
    list_file.write_text("\n".join(lines) + "\n")

    out = wd / f"{meta['slug']}_ch{meta['chapter']}.mp4"
    r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(list_file), "-c", "copy", "-movflags", "+faststart", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                            "-i", str(list_file), "-c:v", "libx264", "-crf", "20",
                            "-c:a", "aac", "-movflags", "+faststart", str(out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"final concat failed: {r.stderr[-800:]}")
    log(f"VIDEO -> {out}")
    return out


# ---------------------------------------------------------------- cli
def main() -> None:
    global FPS
    FPS = pick_fps()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["run", "scrape", "ocr", "script", "tts", "subs", "video"])
    ap.add_argument("url")
    ap.add_argument("--voice", default="en-US-AndrewNeural")
    ap.add_argument("--force", action="store_true", help="re-synthesize TTS/subs")
    a = ap.parse_args()
    log(f"fps={FPS} (matched to display refresh; override with AV_FPS)")
    if a.step == "scrape":
        cmd_scrape(a.url)
    elif a.step == "ocr":
        ocr_panels(load_meta(a.url))
    elif a.step == "script":
        cmd_scrape(a.url) if not (Path(load_meta(a.url)["workdir"]) / "images").exists() else None
        cmd_script(a.url)
    elif a.step == "tts":
        cmd_tts(a.url, a.voice, force=a.force)
    elif a.step == "subs":
        cmd_tts(a.url, a.voice, force=True)
    elif a.step == "video":
        cmd_video(a.url, a.voice)
    else:
        cmd_scrape(a.url)
        cmd_script(a.url)
        cmd_tts(a.url, a.voice)
        cmd_video(a.url, a.voice)


if __name__ == "__main__":
    main()
