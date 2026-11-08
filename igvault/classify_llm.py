"""LLM-first topic classification.

The old approach (embed the raw caption -> k-means -> name each cluster) failed
badly on this kind of data: Spanish and English posts clustered by *language*,
off-topic videos were forced into the nearest of k buckets, and every video in
a cluster received the cluster's tags instead of its own.

This module classifies each video on its own merits, in three passes:

1. describe  - the LLM reads each video (caption, hashtags, author, transcript,
               and a few video frames when there is little text) and writes an
               English one-line summary plus specific tags. Cached per video.
2. taxonomy  - the LLM reads all summaries and proposes the topic folders
               (or uses the fixed list from ``classify.topics`` in config.yaml).
3. assign    - the LLM puts every video into exactly one topic (or "Other").
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .llm import LLM
from .utils import chunked, folder_name, log, one_line, read_json, resolve, warn, write_json

DESCRIBE_VERSION = 2
OTHER = "Other"

DESCRIBE_SYSTEM = (
    "You catalogue a personal library of saved Instagram videos. For each video you get "
    "its author handle, caption, hashtags and sometimes a speech transcript or still frames. "
    "Work out what the video is actually ABOUT (its subject matter), ignoring engagement bait "
    "like 'follow for more', 'comment YES', generic hashtags (#viral #fyp #reels) and emojis. "
    "Captions can be in any language; always answer in English. "
    "If it is an ad, describe the product category (e.g. 'men's t-shirt ad'). "
    "Never guess wildly: if there is truly no information, say so."
)

TAXONOMY_SYSTEM = (
    "You design the folder structure for a personal library of saved Instagram videos. "
    "Folders must reflect the subject matter a person would browse by, be mutually "
    "exclusive, and be specific enough to be useful but broad enough to hold several videos."
)

ASSIGN_SYSTEM = (
    "You file saved Instagram videos into an existing set of topic folders. "
    "Choose the single folder whose subject matches the video best. Use 'Other' only "
    "when no folder fits reasonably."
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _video_text(record: dict[str, Any], transcript: str) -> dict[str, Any]:
    caption = (record.get("caption") or "").strip()
    return {
        "author": "@" + (record.get("author") or "unknown"),
        "caption": one_line(caption, 900) or "(none)",
        "hashtags": " ".join(f"#{t}" for t in (record.get("hashtags") or [])[:25]) or "(none)",
        "transcript": one_line(transcript, 900) if transcript else "(none)",
    }


def _is_low_text(record: dict[str, Any], transcript: str) -> bool:
    caption = (record.get("caption") or "").strip()
    # strip hashtags / mentions / emojis to measure real words
    words = [w for w in caption.split() if w[:1] not in "#@" and any(c.isalpha() for c in w)]
    return len(transcript) < 40 and len(words) < 12


def _frames(cfg, record: dict[str, Any], count: int = 3) -> list[str]:
    """Return a few JPEG frames of the video as base64 data URLs (needs ffmpeg)."""
    video = resolve(cfg.root, record.get("video") or "")
    if not video or not video.is_file():
        return []
    duration = float(record.get("duration") or 0) or 10.0
    stamps = [max(0.5, duration * f) for f in (0.15, 0.5, 0.85)][:count]
    out: list[str] = []
    for ts in stamps:
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-ss", f"{ts:.2f}", "-i", str(video),
                    "-frames:v", "1", "-vf", "scale=448:-2", "-f", "image2pipe",
                    "-vcodec", "mjpeg", "-q:v", "6", "-",
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return out
        if proc.returncode == 0 and proc.stdout:
            out.append("data:image/jpeg;base64," + base64.b64encode(proc.stdout).decode())
    return out


def _cache_key(text: dict[str, Any], with_frames: bool, model: str) -> str:
    raw = json.dumps([DESCRIBE_VERSION, model, with_frames, text], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _clean_tags(raw: Any, limit: int = 6) -> list[str]:
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list):
        return []
    tags: list[str] = []
    for tag in raw:
        tag = str(tag).strip().lower().lstrip("#")
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:limit]


# --------------------------------------------------------------------------- #
# pass 1: describe each video
# --------------------------------------------------------------------------- #
DESCRIBE_RULES = (
    "For each video return:\n"
    '- "summary": one English sentence (max 25 words) saying concretely what the video is about.\n'
    '- "subject": the broad subject area in 1-3 English words (e.g. "volleyball", "home workouts", '
    '"cooking", "personal finance", "programming", "local politics", "menswear").\n'
    '- "tags": 3-6 specific lowercase English keywords about the content (no generic words like '
    '"video", "tips", "viral", "reels", "content").\n'
    '- "unclear": true only if there is not enough information to tell what it is about.\n'
)


def _describe_batch(llm: LLM, batch: list[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    lines = [f"[{sid}] " + json.dumps(text, ensure_ascii=False) for sid, text in batch]
    prompt = (
        "Videos:\n" + "\n".join(lines) + "\n\n" + DESCRIBE_RULES
        + '\nReturn JSON: {"videos": [{"id": "<id in brackets>", "summary": "...", '
        '"subject": "...", "tags": ["..."], "unclear": false}, ...]} with one entry per video.'
    )
    answer = llm.chat_json(DESCRIBE_SYSTEM, prompt, max_tokens=250 * len(batch) + 200) or {}
    return _index_by_id(answer.get("videos"))


def _describe_with_frames(llm: LLM, sid: str, text: dict[str, Any], frames: list[str]) -> dict[str, Any] | None:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Video [{sid}] " + json.dumps(text, ensure_ascii=False)
                + "\nThe caption has little information, so here are frames from the video.\n\n"
                + DESCRIBE_RULES
                + '\nReturn JSON: {"id": "...", "summary": "...", "subject": "...", "tags": ["..."], "unclear": false}'
            ),
        }
    ]
    content += [{"type": "image_url", "image_url": {"url": url, "detail": "low"}} for url in frames]
    return llm.chat_json(DESCRIBE_SYSTEM, content, max_tokens=400)


def _index_by_id(rows: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("id") is not None:
                out[str(row["id"]).strip().strip("[]")] = row
    return out


def describe_videos(
    cfg, llm: LLM, records: list[dict[str, Any]], transcripts: list[str]
) -> dict[str, dict[str, Any]]:
    cache_path = cfg.path_for("paths.work") / "descriptions.json"
    cache: dict[str, dict[str, Any]] = read_json(cache_path, default={}) or {}
    use_frames = bool(cfg.get("classify.use_frames", True))
    batch_size = int(cfg.get("classify.describe_batch", 12) or 12)

    results: dict[str, dict[str, Any]] = {}
    text_jobs: list[tuple[str, dict[str, Any], str]] = []
    frame_jobs: list[tuple[str, dict[str, Any], str, dict[str, Any]]] = []

    for record, transcript in zip(records, transcripts):
        sc = record["shortcode"]
        text = _video_text(record, transcript)
        with_frames = use_frames and _is_low_text(record, transcript)
        key = _cache_key(text, with_frames, llm.chat_model)
        cached = cache.get(sc)
        if cached and cached.get("key") == key:
            results[sc] = cached
            continue
        if with_frames:
            frame_jobs.append((sc, text, key, record))
        else:
            text_jobs.append((sc, text, key))

    todo = len(text_jobs) + len(frame_jobs)
    if todo:
        log(f"Describing {todo} video(s) with {llm.chat_model} "
            f"({len(frame_jobs)} using video frames; {len(results)} cached)…")

    def store(sc: str, key: str, row: dict[str, Any] | None) -> None:
        if not row or not str(row.get("summary") or "").strip():
            return
        entry = {
            "key": key,
            "summary": one_line(str(row.get("summary")), 220),
            "subject": one_line(str(row.get("subject") or ""), 40).lower(),
            "tags": _clean_tags(row.get("tags")),
            "unclear": bool(row.get("unclear")),
        }
        results[sc] = entry
        cache[sc] = entry

    # text-only, batched
    numbered = list(text_jobs)
    for i, batch in enumerate(chunked(numbered, batch_size)):
        ids = {str(n): job for n, job in enumerate(batch, 1)}
        answers = _describe_batch(llm, [(n, job[1]) for n, job in ids.items()])
        for n, (sc, _text, key) in ids.items():
            store(sc, key, answers.get(n))
        # retry any the model skipped, one at a time
        for n, (sc, text, key) in ids.items():
            if sc not in results:
                store(sc, key, _describe_batch(llm, [("1", text)]).get("1"))
        log(f"  described {min((i + 1) * batch_size, len(numbered))}/{len(numbered)} text videos")
        write_json(cache_path, cache)

    # low-text videos, with frames
    for sc, text, key, record in frame_jobs:
        frames = _frames(cfg, record)
        if frames:
            store(sc, key, _describe_with_frames(llm, "1", text, frames))
        if sc not in results:
            store(sc, key, _describe_batch(llm, [("1", text)]).get("1"))
    if frame_jobs:
        log(f"  described {len(frame_jobs)} low-text videos from frames")
    write_json(cache_path, cache)

    missing = [r["shortcode"] for r in records if r["shortcode"] not in results]
    if missing:
        warn(f"could not describe {len(missing)} video(s): {', '.join(missing[:8])}")
    return results


# --------------------------------------------------------------------------- #
# pass 2: taxonomy
# --------------------------------------------------------------------------- #
def build_taxonomy(
    cfg, llm: LLM, descriptions: dict[str, dict[str, Any]], *, target: int | None, max_topics: int, language: str
) -> list[dict[str, Any]]:
    fixed = cfg.get("classify.topics") or []
    if isinstance(fixed, list) and fixed:
        topics = []
        for entry in fixed:
            if isinstance(entry, dict):
                topics.append({"name": str(entry.get("name", "")).strip(),
                               "description": str(entry.get("description", "")).strip()})
            else:
                topics.append({"name": str(entry).strip(), "description": ""})
        topics = [t for t in topics if t["name"]]
        log(f"Using {len(topics)} fixed topic(s) from config.yaml")
        return topics

    subjects = Counter(d["subject"] for d in descriptions.values() if d.get("subject") and not d.get("unclear"))
    lines = [f"- {d['subject']}: {d['summary']}" for d in descriptions.values() if not d.get("unclear")]
    count_rule = (
        f"Create exactly {target} folders." if target
        else f"Create between 4 and {max_topics} folders — as many as the content genuinely needs."
    )
    min_size = int(cfg.get("classify.min_cluster_size", 3) or 3)
    prompt = (
        f"Here are {len(lines)} saved videos (subject: summary).\n\n"
        + "\n".join(lines)
        + "\n\nSubject frequency: "
        + ", ".join(f"{s} ({n})" for s, n in subjects.most_common(60))
        + "\n\nDesign the topic folders.\n"
        f"- {count_rule}\n"
        f"- Every folder should hold at least {min_size} of these videos; tiny one-off subjects go to 'Other' (do not create it, it exists already).\n"
        "- Folders must not overlap (e.g. do not make both 'Volleyball Drills' and 'Volleyball Training'). "
        "Merge closely related subjects into one folder (e.g. workouts, fitness programs and exercise tips -> one "
        "fitness folder; AI tools, ML theory and data science may be one or two folders, not three). "
        "Split a big subject only when each part has 10+ videos and they are clearly different things.\n"
        "- Aim for fewer than 10% of videos to end up in 'Other': prefer a broader folder (e.g. 'Books & Literature', "
        "'Gadgets & Tech Products', 'Home & Organization') over leaving a recurring theme uncovered.\n"
        "- Group by subject matter, never by language or by the fact that something is an ad.\n"
        f"- name: 1-4 words, Title Case, written in {language}, no emojis or slashes, max 40 characters.\n"
        "- description: one sentence saying exactly what belongs there (and, if useful, what does not).\n"
        'Return JSON: {"topics": [{"name": "...", "description": "..."}]}'
    )
    answer = llm.chat_json(TAXONOMY_SYSTEM, prompt, max_tokens=1500) or {}
    return _clean_topics(answer.get("topics"))


# --------------------------------------------------------------------------- #
# pass 3: assignment
# --------------------------------------------------------------------------- #
def assign_topics(
    cfg, llm: LLM, topics: list[dict[str, Any]], descriptions: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    names = [t["name"] for t in topics]
    lookup = {n.lower(): n for n in names}
    lookup[OTHER.lower()] = OTHER
    folder_list = "\n".join(f"- {t['name']}: {t['description']}" for t in topics) + f"\n- {OTHER}: nothing above fits."
    batch_size = int(cfg.get("classify.assign_batch", 30) or 30)

    def run(batch: list[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        ids = {str(n): sc for n, (sc, _d) in enumerate(batch, 1)}
        lines = [f"[{n}] ({d['subject']}) {d['summary']} | tags: {', '.join(d['tags'])}"
                 for n, (_sc, d) in zip(ids, batch)]
        prompt = (
            "Folders:\n" + folder_list + "\n\nVideos:\n" + "\n".join(lines)
            + "\n\nFor each video pick exactly one folder name from the list (copy it exactly) and "
            "a confidence between 0 and 1. Judge by the video's subject: e.g. a volleyball highlight "
            "belongs in a volleyball folder, a UI/CSS design demo in a programming/web folder. "
            f"Use '{OTHER}' only if no folder is even loosely related.\n"
            'Return JSON: {"assignments": [{"id": "...", "topic": "...", "confidence": 0.9}]}'
        )
        answer = llm.chat_json(ASSIGN_SYSTEM, prompt, max_tokens=40 * len(batch) + 200) or {}
        out: dict[str, dict[str, Any]] = {}
        for n, row in _index_by_id(answer.get("assignments")).items():
            sc = ids.get(n)
            topic = lookup.get(str(row.get("topic") or "").strip().lower())
            if sc and topic:
                try:
                    conf = max(0.0, min(1.0, float(row.get("confidence", 0.5))))
                except (TypeError, ValueError):
                    conf = 0.5
                out[sc] = {"topic": topic, "confidence": round(conf, 2)}
        return out

    pending = [(sc, d) for sc, d in descriptions.items() if not d.get("unclear")]
    results: dict[str, dict[str, Any]] = {}
    for batch in chunked(pending, batch_size):
        results.update(run(list(batch)))
    retry = [(sc, d) for sc, d in pending if sc not in results]
    for batch in chunked(retry, 5):
        results.update(run(list(batch)))
    return results


# --------------------------------------------------------------------------- #
# pass 2b: refine — add folders for what ended up in "Other"
# --------------------------------------------------------------------------- #
def refine_taxonomy(
    cfg, llm: LLM, topics: list[dict[str, Any]], leftovers: dict[str, dict[str, Any]], *, max_topics: int, language: str
) -> list[dict[str, Any]]:
    room = max_topics - len(topics)
    if room <= 0 or not leftovers:
        return []
    min_size = int(cfg.get("classify.min_cluster_size", 3) or 3)
    existing = "\n".join(f"- {t['name']}: {t['description']}" for t in topics)
    lines = [f"- {d['subject']}: {d['summary']}" for d in leftovers.values()]
    prompt = (
        "Existing folders:\n" + existing
        + f"\n\nThese {len(lines)} videos did not fit any existing folder:\n" + "\n".join(lines)
        + f"\n\nPropose up to {room} NEW folders that together cover as many of these videos as possible. "
        f"Each new folder must hold at least {min_size} of them, so prefer broader themes "
        "(e.g. 'Books & Literature', 'Gadgets & Tech Products', 'Home & Organization', 'Music & Events', "
        "'Movies & TV', 'Travel'). Do not duplicate or overlap an existing folder. "
        f"Names: 1-4 words, Title Case, in {language}, max 40 characters.\n"
        'Return JSON: {"topics": [{"name": "...", "description": "..."}]}'
    )
    answer = llm.chat_json(TAXONOMY_SYSTEM, prompt, max_tokens=1000) or {}
    return _clean_topics(answer.get("topics"), {t["name"].lower() for t in topics})[:room]


def _clean_topics(rows: Any, taken: set[str] | None = None) -> list[dict[str, Any]]:
    seen = set(taken or ())
    topics: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = folder_name(str(row.get("name") or "").strip(), "")
        low = name.lower()
        if not name or low in seen or low.startswith(OTHER.lower()) or "misc" in low or "general" in low:
            continue
        seen.add(low)
        topics.append({"name": name, "description": one_line(str(row.get("description") or ""), 200)})
    return topics
