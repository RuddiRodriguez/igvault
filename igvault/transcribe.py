from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .metadata import load_records
from .utils import banner, die, log, resolve, transcript_path, warn


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def run_transcribe(
    cfg,
    *,
    model_name: str | None = None,
    limit: int = 0,
    force: bool = False,
    language: str | None = None,
) -> int:
    """Transcribe the audio track of every downloaded video with Whisper.

    Returns the number of transcripts written.
    """
    banner("Transcribe audio")

    enabled = cfg.get("transcribe.enabled", "auto")
    if enabled is False or str(enabled).lower() == "false":
        log("Transcription disabled in config.")
        return 0

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        if enabled is True or str(enabled).lower() == "true":
            die(
                "faster-whisper is required (transcribe.enabled: true). "
                "Install it with: pip install -r requirements-transcribe.txt"
            )
        warn(
            "faster-whisper is not installed — skipping audio transcription. "
            "Install it with: pip install -r requirements-transcribe.txt"
        )
        return 0

    if not shutil.which("ffmpeg"):
        die("ffmpeg is required to extract audio. Install it with: brew install ffmpeg")

    records = load_records(cfg)
    if not records:
        warn("no indexed posts found — run `python -m igvault collect` first")
        return 0

    targets = [r for r in records if r.get("video")]
    if force:
        pending = targets
    else:
        pending = [r for r in targets if not transcript_path(cfg, r["shortcode"]).is_file()]
    if limit:
        pending = pending[:limit]

    if not pending:
        log("Every video already has a transcript.")
        return 0

    name = model_name or cfg.get("transcribe.model", "small")
    device = cfg.get("transcribe.device", "auto")
    compute_type = cfg.get("transcribe.compute_type", "int8")
    lang = language or cfg.get("transcribe.language", "auto")
    lang = None if str(lang).lower() in {"", "auto", "none"} else str(lang)
    min_chars = int(cfg.get("transcribe.min_chars", 20) or 0)

    log(f"Loading Whisper model '{name}' (device={device}, compute_type={compute_type})…")
    model = WhisperModel(name, device=device, compute_type=compute_type)

    out_dir = cfg.path_for("paths.work") / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cfg.path_for("paths.work") / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for index, record in enumerate(pending, start=1):
        shortcode = record["shortcode"]
        video = resolve(cfg.root, record.get("video"))
        target = transcript_path(cfg, shortcode)
        prefix = f"[{index:3d}/{len(pending):3d}] {shortcode}"

        if not video or not video.is_file():
            warn(f"{prefix}: video file missing, skipped")
            continue

        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{shortcode}_", suffix=".wav", dir=tmp_dir)
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(video),
                    "-vn", "-ac", "1", "-ar", "16000",
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                warn(f"{prefix}: no usable audio ({proc.stderr.strip().splitlines()[-1] if proc.stderr else 'ffmpeg error'})")
                continue

            segments, _info = model.transcribe(
                str(tmp_path),
                beam_size=5,
                vad_filter=True,
                language=lang,
            )
            text = _clean(" ".join(segment.text.strip() for segment in segments if segment.text))
            target.write_text(text, encoding="utf-8")
            written += 1
            state = f"{len(text)} chars" if len(text) >= min_chars else "no speech detected"
            log(f"{prefix}: {state}")
        except KeyboardInterrupt:
            warn("interrupted — completed transcripts are kept")
            raise SystemExit(130)
        except Exception as exc:  # noqa: BLE001
            warn(f"{prefix}: transcription failed: {exc}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    log(f"Done. {written} transcript(s) written to {out_dir}")
    return written
