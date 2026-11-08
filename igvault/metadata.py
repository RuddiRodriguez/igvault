from __future__ import annotations

import json
import lzma
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import (
    banner,
    iso_from_timestamp,
    log,
    rel,
    transcript_path,
    warn,
    write_jsonl,
)

HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
MEDIA_SUFFIXES = {".mp4", ".mov", ".m4v", ".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}


def _read_node(path: Path) -> dict[str, Any]:
    if path.name.endswith(".xz"):
        with lzma.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    if isinstance(data, dict) and isinstance(data.get("node"), dict):
        return data["node"]
    if not isinstance(data, dict):
        raise ValueError("post metadata is not a JSON object")
    return data


def _caption(node: dict[str, Any]) -> str:
    edges = (node.get("edge_media_to_caption") or {}).get("edges") or []
    if edges:
        text = ((edges[0] or {}).get("node") or {}).get("text")
        if text:
            return str(text)
    caption = node.get("caption")
    if isinstance(caption, str):
        return caption
    if isinstance(caption, dict) and caption.get("text"):
        return str(caption["text"])
    iphone_caption = (node.get("iphone_struct") or {}).get("caption")
    if isinstance(iphone_caption, str):
        return iphone_caption
    if isinstance(iphone_caption, dict) and iphone_caption.get("text"):
        return str(iphone_caption["text"])
    return ""


def _owner(node: dict[str, Any]) -> tuple[str, str]:
    owner = node.get("owner") or {}
    username = owner.get("username") or node.get("owner_username") or ""
    owner_id = owner.get("id") or owner.get("pk") or node.get("owner_id") or ""
    if not username:
        iphone_user = (node.get("iphone_struct") or {}).get("user") or {}
        username = iphone_user.get("username") or ""
        owner_id = owner_id or iphone_user.get("pk") or ""
    return str(username or "unknown"), str(owner_id or "")


def _posted_at(node: dict[str, Any]) -> str:
    for key in ("taken_at_timestamp", "taken_at"):
        if node.get(key):
            iso = iso_from_timestamp(node[key])
            if iso:
                return iso
    iphone = node.get("iphone_struct") or {}
    for key in ("taken_at", "taken_at_timestamp"):
        if iphone.get(key):
            iso = iso_from_timestamp(iphone[key])
            if iso:
                return iso
    return ""


def _media_files(folder: Path, stem: str) -> list[Path]:
    matches: list[Path] = []
    if not folder.is_dir():
        return matches
    for candidate in folder.iterdir():
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        if candidate.stem == stem or candidate.name.startswith(stem + "_"):
            matches.append(candidate)
    return sorted(matches, key=lambda p: p.name)


def build_records(cfg, *, quiet: bool = False) -> list[dict[str, Any]]:
    """Turn Instaloader's JSON sidecars into one normalized record per post."""
    if not quiet:
        banner("Index downloaded posts")

    raw_dir = cfg.path_for("paths.raw")
    work_dir = cfg.path_for("paths.work")
    work_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.is_dir():
        warn(f"raw directory does not exist yet: {raw_dir}")
        return []

    json_files = sorted(raw_dir.rglob("*.json.xz")) + sorted(raw_dir.rglob("*.json"))
    json_files = [p for p in json_files if p.is_file()]

    records: dict[str, dict[str, Any]] = {}
    errors = 0

    for path in json_files:
        try:
            node = _read_node(path)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            warn(f"skipping unreadable metadata {path.name}: {exc}")
            continue

        if path.name.endswith(".json.xz"):
            stem = path.name[: -len(".json.xz")]
        else:
            stem = path.stem

        shortcode = str(node.get("shortcode") or "").strip()
        if not shortcode:
            # fall back to the last "_" segment of the filename
            shortcode = stem.rsplit("_", 1)[-1]

        caption = _caption(node)
        author, author_id = _owner(node)
        hashtags = sorted({tag.lower() for tag in HASHTAG_RE.findall(caption)})
        media = _media_files(path.parent, stem)
        videos = [p for p in media if p.suffix.lower() in VIDEO_SUFFIXES]
        product_type = str(node.get("product_type") or (node.get("iphone_struct") or {}).get("product_type") or "")
        is_video = bool(node.get("is_video")) or bool(videos)
        duration = node.get("video_duration") or (node.get("iphone_struct") or {}).get("video_duration") or 0

        transcript = transcript_path(cfg, shortcode)
        has_speech = False
        if transcript.is_file():
            try:
                has_speech = len(transcript.read_text(encoding="utf-8").strip()) >= 3
            except OSError:
                has_speech = False

        record = {
            "shortcode": shortcode,
            "media_id": str(node.get("id") or node.get("pk") or ""),
            "url": (
                f"https://www.instagram.com/reel/{shortcode}/"
                if product_type == "clips"
                else f"https://www.instagram.com/p/{shortcode}/"
            ),
            "author": author,
            "author_id": author_id,
            "posted_at": _posted_at(node),
            "caption": caption,
            "hashtags": hashtags,
            "typename": str(node.get("__typename") or ("GraphVideo" if is_video else "")),
            "product_type": product_type,
            "is_video": is_video,
            "duration": float(duration or 0),
            "stem": stem,
            "media": [rel(cfg.root, p) for p in media],
            "video": rel(cfg.root, videos[0]) if videos else "",
            "transcript": rel(cfg.root, transcript) if transcript.is_file() else "",
            "has_speech": has_speech,
            "metadata_file": rel(cfg.root, path),
        }

        previous = records.get(shortcode)
        if previous is None or len(node) > previous.get("_node_size", 0):
            record["_node_size"] = len(node)
            records[shortcode] = record

    rows = list(records.values())
    for row in rows:
        row.pop("_node_size", None)
    rows.sort(key=lambda r: (r.get("posted_at") or "", r.get("shortcode") or ""), reverse=True)

    output = work_dir / "records.jsonl"
    write_jsonl(output, rows)

    if not quiet:
        videos = sum(1 for r in rows if r.get("video"))
        with_caption = sum(1 for r in rows if r.get("caption"))
        log(f"{len(rows)} post(s) indexed, {videos} with video, {with_caption} with caption")
        if errors:
            log(f"{errors} unreadable metadata file(s) skipped")
        log(f"Wrote {output}")

    return rows


def load_records(cfg) -> list[dict[str, Any]]:
    from .utils import read_jsonl

    path = cfg.path_for("paths.work") / "records.jsonl"
    rows = read_jsonl(path)
    if not rows:
        rows = build_records(cfg)
    return rows


def top_hashtags(records: list[dict[str, Any]], limit: int = 10) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.get("hashtags") or [])
    return counter.most_common(limit)
