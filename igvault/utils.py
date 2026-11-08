from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, NoReturn, Sequence


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def log(message: str = "", *, end: str = "\n") -> None:
    print(message, end=end, flush=True)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr, flush=True)


def die(message: str, code: int = 1) -> NoReturn:
    print(f"error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def banner(title: str) -> None:
    print()
    print(f"=== {title} " + "=" * max(0, 66 - len(title)))
    print(flush=True)


# --------------------------------------------------------------------------- #
# json helpers
# --------------------------------------------------------------------------- #
def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    tmp.replace(p)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(p)


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def resolve(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def rel(root: Path, value: Path) -> str:
    try:
        return os.path.relpath(value, root)
    except ValueError:
        return str(value)


def transcript_path(cfg, shortcode: str) -> Path:
    return cfg.path_for("paths.work") / "transcripts" / f"{shortcode}.txt"


# --------------------------------------------------------------------------- #
# strings
# --------------------------------------------------------------------------- #
def folder_name(value: str, fallback: str = "Topic", max_len: int = 60) -> str:
    """Turn an arbitrary label into a safe, readable folder name."""
    if not value:
        return fallback
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r'[/\\:*?"<>|\x00-\x1f]', " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if len(value) > max_len:
        value = value[:max_len].rstrip(" .")
    return value or fallback


def slug(value: str, fallback: str = "topic") -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip().lower()
    value = re.sub(r"[\s_-]+", "-", value).strip("-")
    return value or fallback


def one_line(value: str, limit: int = 200) -> str:
    value = re.sub(r"\s+", " ", (value or "")).strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso_from_timestamp(value: Any) -> str:
    try:
        ts = int(value)
    except (TypeError, ValueError):
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return ""


def human_duration(seconds: Any) -> str:
    try:
        seconds = int(round(float(seconds)))
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def count_files(folder: Path, suffixes: tuple[str, ...]) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
