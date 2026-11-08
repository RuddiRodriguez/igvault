from __future__ import annotations

import getpass
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .utils import banner, die, log, warn, write_json


# --------------------------------------------------------------------------- #
# login
# --------------------------------------------------------------------------- #
def _password(username: str) -> str:
    password = os.getenv("IGVAULT_PASSWORD")
    if password:
        return password
    try:
        return getpass.getpass(f"Instagram password for @{username} (not stored): ")
    except EOFError:
        die(
            "no Instagram password available. Set IGVAULT_PASSWORD or run the "
            "command in an interactive terminal."
        )


def _ask_code(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        die("Instagram wants a verification code; run the command interactively.")


def make_client(username: str, work_dir: Path):
    """Return a logged-in instagrapi client, reusing a saved session when possible."""
    try:
        from instagrapi import Client
        from instagrapi.exceptions import (
            BadPassword,
            ChallengeRequired,
            TwoFactorRequired,
        )
    except ImportError as exc:  # noqa: BLE001
        die(f"instagrapi could not be imported ({exc}). Run: pip install instagrapi")

    settings_path = work_dir / "igapi_session.json"
    client = Client()
    client.request_timeout = 60
    client.delay_range = [1.5, 3.5]

    def code_handler(_username: str, choice: str) -> str:
        log(f"Instagram wants a {choice} verification code.")
        return _ask_code("Code: ")

    client.challenge_code_handler = code_handler

    if settings_path.is_file():
        try:
            client.load_settings(settings_path)
            client.account_info()
            log(f"Reused Instagram API session for @{username}")
            return client, settings_path
        except Exception as exc:  # noqa: BLE001
            warn(f"saved API session is unusable ({exc}); logging in again")

    password = _password(username)
    log(f"Logging in to Instagram as @{username}…")
    try:
        client.login(username, password)
    except TwoFactorRequired:
        code = _ask_code("Two-factor authentication code: ")
        try:
            client.login(username, password, verification_code=code)
        except Exception as exc:  # noqa: BLE001
            die(f"2FA login failed: {exc}")
    except ChallengeRequired as exc:
        die(
            "Instagram wants to verify this login (challenge_required). Open "
            "Instagram in the app or a browser, approve the login attempt, then "
            f"retry. Details: {exc}"
        )
    except BadPassword:
        die("Instagram rejected the password. Log in once in the app, then retry.")
    except Exception as exc:  # noqa: BLE001
        die(f"Instagram login failed: {exc}")

    client.dump_settings(settings_path)
    try:
        os.chmod(settings_path, 0o600)
    except OSError:
        pass
    log(f"Session saved to {settings_path}")
    return client, settings_path


# --------------------------------------------------------------------------- #
# saved posts
# --------------------------------------------------------------------------- #
def iter_saved(client, limit: int | None = None) -> Iterator[Any]:
    """Yield saved posts newest-first using the private feed/saved/posts/ endpoint."""
    seen: set[str] = set()
    max_id = ""
    while True:
        medias, max_id = client.collection_medias_v1_chunk("saved", max_id=max_id)
        if not medias:
            return
        for media in medias:
            key = str(media.pk)
            if key in seen:
                continue
            seen.add(key)
            yield media
            if limit and len(seen) >= limit:
                return
        if not max_id:
            return
        time.sleep(random.uniform(0.4, 1.2))


def video_parts(media) -> list[tuple[int, str]]:
    """Return [(index, url)] for every video inside a post (1 for a Reel, N for a carousel)."""
    if getattr(media, "media_type", None) == 2 and media.video_url:
        return [(0, str(media.video_url))]
    if getattr(media, "media_type", None) == 8:
        parts: list[tuple[int, str]] = []
        for index, resource in enumerate(media.resources or []):
            if getattr(resource, "video_url", None):
                parts.append((index, str(resource.video_url)))
        return parts
    return []


def photo_parts(media) -> list[tuple[int, str]]:
    if getattr(media, "media_type", None) == 1 and media.thumbnail_url:
        return [(0, str(media.thumbnail_url))]
    if getattr(media, "media_type", None) == 8:
        parts: list[tuple[int, str]] = []
        for index, resource in enumerate(media.resources or []):
            if getattr(resource, "media_type", None) == 1 and resource.thumbnail_url:
                parts.append((index, str(resource.thumbnail_url)))
        return parts
    return []


def _stem(media) -> str:
    taken = getattr(media, "taken_at", None)
    if isinstance(taken, datetime):
        taken = taken.astimezone(timezone.utc)
        day = taken.strftime("%Y-%m-%d")
    else:
        day = "1970-01-01"
    code = str(getattr(media, "code", "") or getattr(media, "pk", ""))
    return f"{day}_{code}"


def _sidecar(media, videos: list[tuple[int, str]]) -> dict[str, Any]:
    taken = getattr(media, "taken_at", None)
    timestamp = int(taken.timestamp()) if isinstance(taken, datetime) else 0
    user = getattr(media, "user", None)
    typename = {2: "GraphVideo", 8: "GraphSidecar"}.get(getattr(media, "media_type", 0), "GraphImage")
    return {
        "node": {
            "__typename": typename,
            "id": str(getattr(media, "pk", "")),
            "shortcode": str(getattr(media, "code", "")),
            "taken_at_timestamp": timestamp,
            "edge_media_to_caption": {
                "edges": [{"node": {"text": str(getattr(media, "caption_text", "") or "")}}]
            },
            "owner": {
                "id": str(getattr(user, "pk", "") or ""),
                "username": str(getattr(user, "username", "") or "unknown"),
            },
            "is_video": bool(videos),
            "video_duration": float(getattr(media, "video_duration", 0) or 0),
            "product_type": str(getattr(media, "product_type", "") or ""),
            "video_url": videos[0][1] if videos else "",
        }
    }


def _download(client, url: str, target: Path) -> bool:
    """Download url to target. Returns False when the file was already there."""
    if target.is_file() and target.stat().st_size > 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{target.stem}.part{target.suffix}"
    downloaded = client.video_download_by_url(
        url, filename=tmp_name, folder=target.parent, overwrite=True
    )
    Path(downloaded).replace(target)
    return True


def run_collect_api(
    cfg,
    *,
    login: str | None = None,
    limit: int | None = None,
    include_photos: bool = False,
) -> int:
    """Download saved posts through Instagram's private mobile API (instagrapi)."""
    banner("Collect saved posts (Instagram private API)")

    raw_dir = cfg.path_for("paths.raw")
    work_dir = cfg.path_for("paths.work")
    raw_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    username = (login or cfg.get("collect.login") or os.getenv("IGVAULT_LOGIN") or "").strip()
    if not username:
        try:
            username = input("Instagram username: ").strip()
        except EOFError:
            username = ""
    if not username:
        die("an Instagram username is required")

    videos_only = bool(cfg.get("collect.videos_only", True)) and not include_photos
    max_count = limit if limit is not None else int(cfg.get("collect.limit", 0) or 0)
    max_count = max_count or None

    client, _ = make_client(username, work_dir)

    log(f"Fetching saved posts (limit={max_count or 'all'}, videos_only={videos_only})…")

    posts = 0
    videos = 0
    new_files = 0
    skipped = 0

    try:
        for media in iter_saved(client, max_count):
            posts += 1
            videos_in_post = video_parts(media)
            photos_in_post = photo_parts(media) if not videos_only else []
            if not videos_in_post and not photos_in_post:
                skipped += 1
                continue

            stem = _stem(media)
            total = len(videos_in_post) + len(photos_in_post)
            try:
                for index, url in videos_in_post:
                    name = f"{stem}.mp4" if total == 1 else f"{stem}_{index + 1}.mp4"
                    if _download(client, url, raw_dir / name):
                        new_files += 1
                        videos += 1
                    else:
                        videos += 1
                for index, url in photos_in_post:
                    name = f"{stem}.jpg" if total == 1 else f"{stem}_{index + 1}.jpg"
                    if _download(client, url, raw_dir / name):
                        new_files += 1
                write_json(raw_dir / f"{stem}.json", _sidecar(media, videos_in_post))
            except Exception as exc:  # noqa: BLE001
                warn(f"skipping {getattr(media, 'code', media.pk)}: {exc}")
                continue

            if posts % 10 == 0:
                log(f"  … {posts} post(s) seen, {new_files} new file(s)")
            time.sleep(random.uniform(0.5, 1.5))
    except KeyboardInterrupt:
        warn("interrupted — already downloaded posts are kept; run again to resume")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        die(
            f"download failed: {exc}\n"
            "Common causes: expired session (delete data/work/igapi_session.json), "
            "challenge_required (open Instagram in the app and approve), or too many "
            "requests (wait a few hours)."
        )

    log(
        f"Done. {posts} saved post(s) seen, {videos} video(s), "
        f"{new_files} new file(s), {skipped} post(s) without video"
    )
    return videos
