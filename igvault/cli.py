from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from . import __version__
from .classify import run_classify
from .collect import run_collect
from .collect_api import run_collect_api
from .config import load_config
from .metadata import build_records
from .organize import run_organize
from .transcribe import run_transcribe
from .utils import banner, log, read_json, read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="igvault",
        description="Organize your saved Instagram videos into AI-discovered topic folders.",
    )
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--version", action="version", version=f"igvault {__version__}")

    sub = parser.add_subparsers(dest="command")

    collect = sub.add_parser("collect", help="download saved posts")
    collect.add_argument("--login", help="Instagram username (defaults to IGVAULT_LOGIN)")
    collect.add_argument(
        "--engine",
        choices=["api", "instaloader"],
        default="api",
        help="api = private mobile API (default, more reliable); instaloader = web GraphQL",
    )
    collect.add_argument("--sessionfile", help="reuse an Instaloader session file")
    collect.add_argument("--limit", type=int, help="stop after N posts (useful for a test run)")
    collect.add_argument("--fast-update", action="store_true", help="stop at the first already-downloaded post")
    collect.add_argument("--include-photos", action="store_true", help="also download photo posts")

    sub.add_parser("index", help="rebuild data/work/records.jsonl from downloaded metadata")

    transcribe = sub.add_parser("transcribe", help="transcribe video audio with Whisper")
    transcribe.add_argument("--model", help="Whisper model: tiny, base, small, medium, large-v3")
    transcribe.add_argument("--limit", type=int, default=0, help="transcribe at most N videos")
    transcribe.add_argument("--force", action="store_true", help="re-transcribe existing transcripts")
    transcribe.add_argument("--language", help="force a language code (default: auto)")

    classify = sub.add_parser("classify", help="cluster videos and name the topics with an LLM")
    classify.add_argument("--clusters", type=int, help="force a fixed number of topics")
    classify.add_argument("--max-clusters", type=int, help="upper bound when auto-choosing k")
    classify.add_argument("--no-llm", action="store_true", help="use TF-IDF and hashtag names only")

    organize = sub.add_parser("organize", help="copy videos into topic folders and write the index")
    organize.add_argument("--dry-run", action="store_true", help="show the plan without touching files")
    organize.add_argument("--mode", choices=["copy", "move", "symlink"], help="how to place files")
    organize.add_argument("--prune", action="store_true", help="remove files from previous runs that moved topic")

    run = sub.add_parser("run", help="collect → index → transcribe → classify → organize")
    run.add_argument("--login", help="Instagram username (defaults to IGVAULT_LOGIN)")
    run.add_argument("--engine", choices=["api", "instaloader"], default="api")
    run.add_argument("--limit", type=int, help="process at most N posts (test mode)")
    run.add_argument("--skip-collect", action="store_true")
    run.add_argument("--skip-transcribe", action="store_true")
    run.add_argument("--skip-classify", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="organize in preview mode only")
    run.add_argument("--prune", action="store_true")
    run.add_argument("--clusters", type=int)
    run.add_argument("--no-llm", action="store_true")

    sub.add_parser("status", help="show what has been collected, transcribed and classified")

    return parser


def _cmd_status(cfg) -> int:
    banner("Status")
    raw = cfg.path_for("paths.raw")
    work = cfg.path_for("paths.work")
    library = cfg.path_for("paths.library")

    metadata_files = (
        len(list(raw.rglob("*.json.xz"))) + len(list(raw.rglob("*.json"))) if raw.is_dir() else 0
    )
    raw_videos = (
        sum(1 for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov"})
        if raw.is_dir()
        else 0
    )
    records = read_jsonl(work / "records.jsonl")
    transcripts = work / "transcripts"
    transcript_files = list(transcripts.glob("*.txt")) if transcripts.is_dir() else []
    with_speech = 0
    for path in transcript_files:
        try:
            if len(path.read_text(encoding="utf-8").strip()) >= 3:
                with_speech += 1
        except OSError:
            pass

    classification = read_json(work / "classification.json")
    library_files = sum(1 for p in library.rglob("*") if p.is_file()) if library.is_dir() else 0

    log(f"config:            {cfg.path}")
    log(f"raw directory:     {raw}")
    log(f"metadata files:    {metadata_files}")
    log(f"raw videos:        {raw_videos}")
    log(f"indexed posts:     {len(records)}")
    log(f"transcripts:       {len(transcript_files)} ({with_speech} with speech)")
    if classification:
        topics = classification.get("topics") or []
        log(
            f"classification:    {len(topics)} topic(s), "
            f"{len(classification.get('items') or {})} item(s), "
            f"{classification.get('generated_at')}"
        )
    else:
        log("classification:    not run yet")
    log(f"library files:     {library_files} in {library}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    cfg = load_config(args.config)
    cfg.ensure_dirs()

    try:
        if args.command == "collect":
            if args.engine == "instaloader":
                run_collect(
                    cfg,
                    login=args.login,
                    sessionfile=args.sessionfile,
                    limit=args.limit,
                    fast_update=args.fast_update or None,
                    include_photos=args.include_photos,
                )
            else:
                run_collect_api(
                    cfg,
                    login=args.login,
                    limit=args.limit,
                    include_photos=args.include_photos,
                )
        elif args.command == "index":
            build_records(cfg)
        elif args.command == "transcribe":
            run_transcribe(
                cfg,
                model_name=args.model,
                limit=args.limit,
                force=args.force,
                language=args.language,
            )
        elif args.command == "classify":
            run_classify(cfg, clusters=args.clusters, max_clusters=args.max_clusters, use_llm=not args.no_llm)
        elif args.command == "organize":
            run_organize(cfg, dry_run=args.dry_run, mode=args.mode, prune=args.prune)
        elif args.command == "run":
            if not args.skip_collect:
                if args.engine == "instaloader":
                    run_collect(cfg, login=args.login, limit=args.limit)
                else:
                    run_collect_api(cfg, login=args.login, limit=args.limit)
            else:
                log("Skipping collection.")
            build_records(cfg)
            if not args.skip_transcribe:
                run_transcribe(cfg, limit=args.limit or 0)
            else:
                log("Skipping transcription.")
            if not args.skip_classify:
                run_classify(cfg, clusters=args.clusters, use_llm=not args.no_llm)
            else:
                log("Skipping classification.")
            run_organize(cfg, dry_run=args.dry_run, prune=args.prune)
        elif args.command == "status":
            return _cmd_status(cfg)
        else:
            parser.print_help()
            return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    return 0
