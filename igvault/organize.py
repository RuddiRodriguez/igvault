from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from .metadata import load_records
from .utils import (
    banner,
    die,
    folder_name,
    human_duration,
    iso_now,
    log,
    one_line,
    read_json,
    rel,
    resolve,
    transcript_path,
    warn,
    write_json,
)


def _info_markdown(cfg, record: dict[str, Any], info: dict[str, Any], topic: str) -> str:
    lines: list[str] = []
    title = one_line(record.get("caption") or record.get("shortcode") or "Saved video", 90)
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **Topic:** {topic}")
    lines.append(f"- **Author:** @{record.get('author') or 'unknown'}")
    if record.get("posted_at"):
        lines.append(f"- **Posted:** {record['posted_at'][:10]}")
    if record.get("duration"):
        lines.append(f"- **Duration:** {human_duration(record['duration'])}")
    lines.append(f"- **Source:** {record.get('url') or ''}")
    if info.get("tags"):
        lines.append(f"- **Tags:** {', '.join(info['tags'])}")
    if info.get("confidence"):
        lines.append(f"- **Topic confidence:** {info['confidence']}")
    lines.append("")

    if record.get("caption"):
        lines.append("## Caption")
        lines.append("")
        lines.append(record["caption"].strip())
        lines.append("")

    transcript_file = transcript_path(cfg, record["shortcode"])
    if transcript_file.is_file():
        try:
            transcript = transcript_file.read_text(encoding="utf-8").strip()
        except OSError:
            transcript = ""
        if transcript:
            lines.append("## Transcript")
            lines.append("")
            lines.append(transcript)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _place(source: Path, destination: Path, mode: str) -> str:
    if mode == "symlink":
        if destination.is_symlink() or destination.exists():
            return "exists"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve())
        return "linked"

    if destination.exists():
        try:
            if destination.stat().st_size == source.stat().st_size:
                return "exists"
        except OSError:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(source), str(destination))
        return "moved"
    shutil.copy2(source, destination)
    return "copied"


def run_organize(
    cfg,
    *,
    dry_run: bool = False,
    mode: str | None = None,
    prune: bool = False,
) -> dict[str, Any]:
    banner("Organize into topic folders")

    records = load_records(cfg)
    if not records:
        warn("no indexed posts — run `python -m igvault collect` first")
        return {"placed": 0}

    work_dir = cfg.path_for("paths.work")
    classification = read_json(work_dir / "classification.json")
    if not classification:
        die("classification.json not found — run `python -m igvault classify` first")

    items: dict[str, dict[str, Any]] = classification.get("items") or {}
    topics = classification.get("topics") or []
    library = cfg.path_for("paths.library")

    mode = (mode or cfg.get("organize.mode", "copy")).lower()
    if mode not in {"copy", "move", "symlink"}:
        die(f"invalid organize mode: {mode} (use copy, move or symlink)")

    unsorted_name = folder_name(cfg.get("organize.unsorted_name", "Unsorted"), "Unsorted")
    write_info = bool(cfg.get("organize.write_info_md", True))

    plan: list[dict[str, Any]] = []
    planned: dict[str, dict[str, Any]] = {}
    topic_records: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        shortcode = record["shortcode"]
        info = items.get(shortcode) or {}
        topic = folder_name(info.get("topic") or unsorted_name, unsorted_name)
        media = record.get("media") or []
        if not media:
            continue

        folder = library / topic
        date = (record.get("posted_at") or "")[:10] or "unknown"
        author = (record.get("author") or "unknown").lower()
        base = f"{date}_{shortcode}_{author}"
        stem = record.get("stem") or ""

        media_destinations: list[Path] = []
        for media_path in media:
            source = resolve(cfg.root, media_path)
            if not source or not source.is_file():
                warn(f"{shortcode}: missing media file {media_path}")
                continue
            name = Path(media_path).name
            suffix = name[len(stem) :] if stem and name.startswith(stem) else Path(name).suffix
            destination = folder / f"{base}{suffix}"
            plan.append({"source": source, "destination": destination, "kind": "media"})
            media_destinations.append(destination)

        info_destination = folder / f"{base}.md"
        if write_info:
            plan.append(
                {
                    "source": None,
                    "destination": info_destination,
                    "kind": "info",
                    "record": record,
                }
            )

        planned[shortcode] = {
            "topic": topic,
            "folder": folder,
            "base": base,
            "media": media_destinations,
            "info": info_destination if write_info else None,
        }
        topic_records.setdefault(topic, []).append(record)

    media_plan = [entry for entry in plan if entry["kind"] == "media"]
    info_plan = [entry for entry in plan if entry["kind"] == "info"]

    if dry_run:
        log(f"[dry run] would place {len(media_plan)} media file(s) and {len(info_plan)} info file(s)")
        log(f"[dry run] {len(topic_records)} topic folder(s) under {library}")
        for topic in sorted(topic_records, key=lambda name: (-len(topic_records[name]), name)):
            log(f"  {len(topic_records[topic]):4d}  {topic}")
        for entry in media_plan[:10]:
            log(f"  {entry['source'].name} -> {rel(cfg.root, entry['destination'])}")
        if len(media_plan) > 10:
            log(f"  … and {len(media_plan) - 10} more")
        return {"placed": 0, "planned": len(media_plan), "topics": len(topic_records)}

    library.mkdir(parents=True, exist_ok=True)
    stats = {"copied": 0, "moved": 0, "linked": 0, "exists": 0, "info": 0}

    for entry in media_plan:
        outcome = _place(entry["source"], entry["destination"], mode)
        stats[outcome] = stats.get(outcome, 0) + 1

    for entry in info_plan:
        destination: Path = entry["destination"]
        record = entry["record"]
        info = items.get(record["shortcode"]) or {}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            _info_markdown(cfg, record, info, planned[record["shortcode"]]["topic"]),
            encoding="utf-8",
        )
        stats["info"] += 1

    # Per-topic summary files
    topic_meta = {topic["name"]: topic for topic in topics}
    about_files: list[Path] = []
    for topic, members in topic_records.items():
        meta = topic_meta.get(topic) or {}
        about = [
            f"# {topic}",
            "",
            meta.get("description") or f"{len(members)} saved videos.",
            "",
            f"**Videos:** {len(members)}",
        ]
        if meta.get("tags"):
            about.append(f"**Tags:** {', '.join(meta['tags'])}")
        about.append("")
        about.append(f"_Generated by igvault on {iso_now()}._")
        about_path = library / topic / "_about.md"
        about_path.write_text("\n".join(about) + "\n", encoding="utf-8")
        about_files.append(about_path)

    # Index files
    rows: list[dict[str, Any]] = []
    for record in records:
        shortcode = record["shortcode"]
        info = items.get(shortcode) or {}
        entry = planned.get(shortcode)
        if not entry:
            continue
        rows.append(
            {
                "shortcode": shortcode,
                "topic": entry["topic"],
                "posted_at": (record.get("posted_at") or "")[:10],
                "author": record.get("author") or "",
                "duration_seconds": int(record.get("duration") or 0),
                "url": record.get("url") or "",
                "caption": one_line(record.get("caption") or "", 400),
                "hashtags": " ".join(f"#{tag}" for tag in record.get("hashtags") or []),
                "tags": ", ".join(info.get("tags") or []),
                "confidence": info.get("confidence", ""),
                "video_file": rel(cfg.root, entry["media"][0]) if entry["media"] else "",
                "info_file": rel(cfg.root, entry["info"]) if entry.get("info") else "",
                "source_file": record.get("video") or "",
            }
        )

    rows.sort(key=lambda row: (row["topic"].lower(), row["posted_at"], row["shortcode"]), reverse=False)

    csv_path = library / "_index.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["shortcode"])
        writer.writeheader()
        writer.writerows(rows)

    new_files = [entry["destination"] for entry in plan]
    new_files.append(csv_path)
    new_files.extend(about_files)
    new_files.append(library / "README.md")

    if prune:
        previous = read_json(library / "_index.json", default={}) or {}
        old_files = {resolve(cfg.root, value) for value in previous.get("files") or []}
        new_set = {path.resolve() for path in new_files if path}
        stale = [path for path in old_files if path and path.resolve() not in new_set]
        removed = 0
        for path in stale:
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                    removed += 1
            except OSError as exc:
                warn(f"could not remove stale file {path}: {exc}")
        for folder in sorted(library.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if folder.is_dir():
                try:
                    folder.rmdir()
                except OSError:
                    pass
        if removed:
            log(f"Pruned {removed} stale file(s) from previous runs")

    write_json(
        library / "_index.json",
        {
            "generated_at": iso_now(),
            "mode": mode,
            "topics": topics,
            "items": rows,
            "files": [rel(cfg.root, path) for path in new_files if path],
        },
    )

    _write_readme(cfg, library, topics, topic_records)

    log(
        f"Done. copied={stats.get('copied', 0)} moved={stats.get('moved', 0)} "
        f"linked={stats.get('linked', 0)} unchanged={stats.get('exists', 0)} "
        f"info={stats.get('info', 0)}"
    )
    log(f"Library: {library}")
    return {"placed": len(media_plan), "topics": len(topic_records), **stats}


def _write_readme(cfg, library: Path, topics: list[dict[str, Any]], topic_records: dict[str, list]) -> None:
    total = sum(len(members) for members in topic_records.values())
    lines = [
        "# Saved Instagram Library",
        "",
        f"{total} videos organized into {len(topic_records)} topic folders.",
        f"Generated by igvault on {iso_now()}.",
        "",
        "| Topic | Videos | Description |",
        "| --- | ---: | --- |",
    ]
    meta = {topic["name"]: topic for topic in topics}
    for topic in sorted(topic_records, key=lambda name: (-len(topic_records[name]), name)):
        description = (meta.get(topic) or {}).get("description") or ""
        description = description.replace("|", "\\|")
        link = topic.replace(" ", "%20")
        lines.append(f"| [{topic}](./{link}/) | {len(topic_records[topic])} | {description} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `_index.csv` — spreadsheet with every video, topic, author, tags and links.",
            "- `_index.json` — the same index plus the full topic metadata.",
            "- `<topic>/_about.md` — what each folder contains.",
            "- `<topic>/<date>_<shortcode>_<author>.md` — caption + transcript for each video.",
            "",
            "Re-run `python -m igvault classify` then `python -m igvault organize` "
            "after saving new videos.",
            "",
        ]
    )
    (library / "README.md").write_text("\n".join(lines), encoding="utf-8")
