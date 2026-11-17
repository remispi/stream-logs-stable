"""Command line, used by the app window (JSON lines on stdout) and usable on its own.

    stream-log vods <account>                       an account's recent VODs, with file names
    stream-log vod <url>                            one VOD's details
    stream-log status [--mode M]                    data folder, models, what setup still needs
    stream-log setup [--mode M]                     download what's missing
    stream-log run --vod URL [--out DIR] [--name N] download, analyse, write the list
    stream-log run --file VIDEO [--vod URL]         the same for a video you already have
    stream-log relist <project> [--video V]         write the list again from a processed project
    stream-log sync A B                             line up two VODs of the same moments

Every line on stdout is one JSON object: {"type": "steps" | "step" | "log" | "done" | "error" | ...}.
Anything libraries print goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from stream_log import listing

_OUT = sys.stdout


def emit(event: dict) -> None:
    _OUT.write(json.dumps(event, ensure_ascii=False) + "\n")
    _OUT.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stream-log")
    sub = parser.add_subparsers(dest="command", required=True)

    vods = sub.add_parser("vods", help="an account's recent VODs")
    vods.add_argument("account")
    vods.add_argument("--limit", type=int, default=30)

    vod = sub.add_parser("vod", help="one VOD's details")
    vod.add_argument("url")

    for name in ("status", "setup"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=["fast", "balanced", "best"])

    run = sub.add_parser("run", help="download or open a VOD and write its list")
    run.add_argument("--vod", help="Twitch VOD link or id")
    run.add_argument("--file", type=Path, help="a video you already have")
    run.add_argument("--out", type=Path, help="folder for the download (default: Raw Streams)")
    run.add_argument("--name", help="file name for the download, without .mp4")
    run.add_argument("--list-dir", type=Path, help="folder for the list (default: the video's)")
    run.add_argument("--mode", choices=["fast", "balanced", "best"])
    run.add_argument("--profile")
    run.add_argument("--per-hour", type=int, default=listing.PER_HOUR)
    run.add_argument("--no-games", action="store_true",
                     help="don't look for TFT games (blocks follow topics only)")
    run.add_argument("--no-reuse", action="store_true",
                     help="analyse again even if this VOD was analysed before")

    relist = sub.add_parser("relist", help="write the list again from a processed project")
    relist.add_argument("project")
    relist.add_argument("--video", type=Path)
    relist.add_argument("--per-hour", type=int, default=listing.PER_HOUR)
    relist.add_argument("--no-games", action="store_true",
                        help="don't look for TFT games (blocks follow topics only)")

    sync = sub.add_parser("sync", help="line up two VODs (same voice chat or same broadcast)")
    sync.add_argument("a", help="a processed video file or project id")
    sync.add_argument("b")
    sync.add_argument("--label-a")
    sync.add_argument("--label-b")
    sync.add_argument("--per-hour", type=int, default=listing.PER_HOUR)
    sync.add_argument("--no-games", action="store_true",
                      help="don't look for TFT games (blocks follow topics only)")
    return parser


def main(argv: list[str] | None = None) -> int:
    global _OUT
    args = _parser().parse_args(argv)
    _OUT = sys.stdout
    sys.stdout = sys.stderr  # stray prints from libraries must not break the JSON lines
    try:
        if args.command == "vods":
            from stream_log import naming, twitch

            vods = twitch.list_vods(args.account, args.limit)
            emit({"type": "vods", "account": args.account,
                  "vods": [v.to_json() | {"stem": naming.file_stem(v.started, v.title)}
                           for v in vods]})
        elif args.command == "vod":
            from stream_log import naming, twitch

            vod_id = twitch.vod_id(args.url)
            if vod_id is None:
                raise ValueError(f"not a Twitch VOD link: {args.url}")
            v = twitch.vod_info(vod_id)
            emit({"type": "vod", "vod": v.to_json() | {"stem": naming.file_stem(v.started, v.title)}})
        elif args.command in ("status", "setup"):
            from stream_log import setup
            from stream_log.engine.config import get_paths

            paths = get_paths().ensure()
            if args.command == "status":
                emit(setup.status(paths, args.mode))
            else:
                setup.run_setup(paths, args.mode, emit)
        elif args.command == "run":
            from stream_log.run import Job, run

            if not args.vod and not args.file:
                raise ValueError("give --vod or --file")
            run(Job(vod=args.vod, file=args.file, out=args.out, name=args.name,
                    list_dir=args.list_dir, mode=args.mode, profile=args.profile,
                    per_hour=args.per_hour, reuse=not args.no_reuse,
                    find_games=not args.no_games), emit)
        elif args.command == "relist":
            from stream_log.run import relist as relist_

            relist_(args.project, args.video, args.per_hour, emit, find_games=not args.no_games)
        elif args.command == "sync":
            from stream_log import sync

            sync.run(args.a, args.b, args.label_a, args.label_b, emit, args.per_hour,
                     find_games=not args.no_games)
        return 0
    except KeyboardInterrupt:
        emit({"type": "error", "text": "stopped"})
        return 130
    except Exception as error:  # noqa: BLE001 - reported to the app as one line
        traceback.print_exc(file=sys.stderr)
        emit({"type": "error", "text": f"{type(error).__name__}: {error}"})
        return 1
    finally:
        sys.stdout = _OUT


if __name__ == "__main__":
    raise SystemExit(main())
