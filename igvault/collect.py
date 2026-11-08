from __future__ import annotations

import getpass
import os
from pathlib import Path

from .utils import banner, die, log, warn


def _login(loader, username: str, session_path: Path) -> None:
    import instaloader

    password = os.getenv("IGVAULT_PASSWORD") or getpass.getpass(
        f"Instagram password for @{username} (not stored): "
    )
    try:
        loader.login(username, password)
    except instaloader.TwoFactorAuthRequiredException:
        try:
            code = input("Two-factor authentication code: ").strip()
        except EOFError:
            die(
                "2FA is enabled on this account. Run the command interactively "
                "(`python -m igvault collect`) and enter the code when asked."
            )
        try:
            loader.two_factor_login(code)
        except Exception as exc:  # noqa: BLE001
            die(f"2FA login failed: {exc}")
    except instaloader.BadCredentialsException:
        die("Instagram rejected the username/password. If your account uses a linked login, log in once in the app and retry.")
    except Exception as exc:  # noqa: BLE001
        die(f"Instagram login failed: {exc}")

    loader.save_session_to_file(str(session_path))
    try:
        os.chmod(session_path, 0o600)
    except OSError:
        pass
    log(f"Session saved to {session_path}")


def run_collect(
    cfg,
    *,
    login: str | None = None,
    sessionfile: str | None = None,
    limit: int | None = None,
    fast_update: bool | None = None,
    include_photos: bool = False,
) -> int:
    """Download every saved post with Instaloader.

    Returns the number of media files present in the raw directory afterwards.
    """
    banner("Collect saved posts (Instaloader)")

    try:
        import instaloader
    except ImportError as exc:
        die(f"instaloader could not be imported ({exc}). Run: pip install -r requirements.txt")

    raw_dir = cfg.path_for("paths.raw")
    work_dir = cfg.path_for("paths.work")
    raw_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    username = (login or cfg.get("collect.login") or os.getenv("IGVAULT_LOGIN") or "").strip()
    if not username:
        username = input("Instagram username: ").strip()
    if not username:
        die("an Instagram username is required")

    session_value = sessionfile or cfg.get("collect.sessionfile") or ""
    session_path = Path(session_value).expanduser() if session_value else work_dir / "ig_session"
    if not session_path.is_absolute():
        session_path = cfg.root / session_path
    session_path.parent.mkdir(parents=True, exist_ok=True)

    videos_only = bool(cfg.get("collect.videos_only", True)) and not include_photos
    max_count = limit if limit is not None else int(cfg.get("collect.limit", 0) or 0)
    max_count = max_count or None
    fast = cfg.get("collect.fast_update", False) if fast_update is None else fast_update

    loader = instaloader.Instaloader(
        dirname_pattern=cfg.get("collect.dirname_pattern", "saved"),
        filename_pattern=cfg.get("collect.filename_pattern", "{date_utc:%Y-%m-%d}_{shortcode}"),
        download_pictures=not videos_only,
        download_videos=True,
        download_video_thumbnails=False,
        download_comments=False,
        save_metadata=True,
        compress_json=True,
        post_metadata_txt_pattern="{caption}",
        quiet=False,
    )

    if session_path.is_file():
        try:
            loader.load_session_from_file(username, str(session_path))
            log(f"Loaded Instagram session for @{username}")
        except Exception as exc:  # noqa: BLE001
            warn(f"could not reuse session ({exc}); logging in again")
            _login(loader, username, session_path)
    else:
        _login(loader, username, session_path)

    def post_filter(post) -> bool:
        if not videos_only:
            return True
        try:
            return bool(post.is_video)
        except Exception:  # noqa: BLE001
            return True

    log(f"Fetching saved posts (limit={max_count or 'all'}, videos_only={videos_only})…")
    log("Instagram rate-limits aggressively; this may take a while for large libraries.")

    previous_cwd = os.getcwd()
    os.chdir(raw_dir)
    try:
        loader.download_saved_posts(
            max_count=max_count,
            fast_update=bool(fast),
            post_filter=post_filter if videos_only else None,
        )
    except KeyboardInterrupt:
        warn("interrupted — already downloaded posts are kept; run again to resume")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        die(
            "download failed: "
            f"{exc}\n"
            "Common causes: expired session (delete data/work/ig_session), "
            "checkpoint_required (open Instagram in a browser and clear the check), "
            "or too many requests (wait a few hours)."
        )
    finally:
        os.chdir(previous_cwd)

    media = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov"}]
    log(f"Done. {len(media)} video file(s) in {raw_dir}")
    return len(media)
