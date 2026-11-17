"""One job, start to finish: get the video (download it, or use a file you have), get the chat,
run the analysis up to its event log, find the games, write the list.

Progress goes out as events (dicts) through `emit`; the CLI prints them as JSON lines for the app.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from stream_log import games, listing, naming, output, sync, twitch

Emit = Callable[[dict], None]

# The analysis stages, in order (`engine.pipeline.STAGES`), then this app's own.
ANALYSIS_UNTIL = "events"
STEPS = [
    ("vod", "Reading the Twitch VOD"),
    ("download", "Downloading the video"),
    ("chat", "Downloading the chat"),
    ("media", "Making a preview copy"),
    ("transcribe", "Transcribing"),
    ("align", "Timing every word"),
    ("words", "Checking names and game terms"),
    ("speakers", "Telling voices apart"),
    ("audio_signals", "Listening for reactions"),
    ("chat_signals", "Reading the chat"),
    ("visual", "Looking at the video"),
    ("fusion", "Combining signals"),
    ("chapters", "Finding topic changes"),
    ("llm", "Describing what happens (local LLM)"),
    ("events", "Collecting events"),
    ("games", "Finding games"),
    ("list", "Writing the list"),
]


@dataclass
class Job:
    vod: str | None = None  # Twitch VOD URL or id: download it (unless `file` is given)
    file: Path | None = None  # a video you already have
    out: Path | None = None  # folder for a downloaded video (default: the user's Raw Streams)
    name: str | None = None  # file name for a download, without .mp4
    list_dir: Path | None = None  # where the list goes (default: next to the video)
    mode: str | None = None  # LLM mode: fast | balanced | best (default: tier's)
    profile: str | None = None  # voice profile (default: the usual one)
    per_hour: int = listing.PER_HOUR
    find_games: bool = True  # look for TFT games; off: blocks follow the topics only
    reuse: bool = True  # reuse an earlier analysis of this Twitch VOD


def default_raw_folder() -> Path:
    return Path.home() / "Videos" / "Raw Streams"


def default_profile(paths) -> str | None:
    """The voice profile most projects used, or the only one there is."""
    from stream_log.engine.project import list_projects

    profiles = sorted(p.name for p in paths.profiles.glob("*") if p.is_dir()) \
        if paths.profiles.exists() else []
    if not profiles:
        return None
    used = collections.Counter(
        project.meta.get("profile") for project in list_projects(paths)
        if project.meta.get("profile") in profiles)
    return used.most_common(1)[0][0] if used else (profiles[0] if len(profiles) == 1 else None)


def reusable_project(paths, vod: twitch.Vod):
    """An earlier project for the same Twitch VOD that already has its event log."""
    from stream_log.engine.project import list_projects

    for project in list_projects(paths):
        meta = project.meta
        same = project.id == f"twitch-{vod.id}" or meta.get("url") == vod.url \
            or str(meta.get("twitch_id")) == vod.id
        if same and project.stage_record(ANALYSIS_UNTIL) and project.path("events.json").exists() \
                and project.path("proxy.mp4").exists():
            return project
    return None


def run(job: Job, emit: Emit) -> dict:
    from stream_log.engine import pipeline
    from stream_log.engine.config import get_paths

    def step(id_: str, fraction: float | None = None, text: str = "") -> None:
        emit({"type": "step", "id": id_, "fraction": fraction, "text": text})

    def log(text: str) -> None:
        emit({"type": "log", "text": str(text)})

    emit({"type": "steps", "steps": [{"id": i, "label": label} for i, label in STEPS]})
    paths = get_paths().ensure()

    vod = None
    if job.vod:
        step("vod")
        vod_id = twitch.vod_id(job.vod)
        if vod_id is None:
            raise ValueError(f"not a Twitch VOD link: {job.vod}")
        vod = twitch.vod_info(vod_id)
        log(f"{vod.channel}: {vod.title} ({listing.minutes(vod.length)})")

    if job.file:
        video = Path(job.file)
        if not video.is_file():
            raise FileNotFoundError(f"no such video: {video}")
    elif vod:
        folder = Path(job.out) if job.out else default_raw_folder()
        if job.name and (problem := naming.bad_stem(job.name)):
            raise ValueError(f"can't save as {job.name!r}: {problem}")
        stem = job.name or naming.file_stem(vod.started, vod.title)
        skipped: list[str] = []

        def mine(path: Path) -> bool:
            same = twitch.is_same_vod(path, vod)
            if same is None:  # can't tell (no ffprobe): keep it rather than download again
                log(f"can't check that {path.name} is this VOD: using it as is")
                return True
            if not same:
                skipped.append(path.name)
            return same

        video = naming.free_path(folder, stem, reuse=mine)
        if skipped:
            log(f"{', '.join(skipped)}: another video, not this VOD")
        if video.exists():
            log(f"already downloaded: {video.name}")
        else:
            step("download", 0.0, video.name)
            video = twitch.download_video(vod, video, lambda f, t: step("download", f, t))
    else:
        raise ValueError("give a Twitch VOD or a video file")

    chat_file = None
    if vod:
        step("chat", 0.0)
        chat_file = paths.cache / "twitch-chat" / f"{vod.id}.json"
        if chat_file.exists():
            log("chat: already downloaded")
        else:
            try:
                count = twitch.download_chat(vod, chat_file, lambda f, t: step("chat", f, t))
                log(f"chat: {count} messages")
            except twitch.TwitchError as error:
                log(f"no chat: {error}")
                chat_file = None

    project = reusable_project(paths, vod) if (vod and job.reuse) else None
    if project is not None:
        log(f"reusing the earlier analysis of this VOD (project {project.id})")
    else:
        profile = job.profile or default_profile(paths)
        project = pipeline.ingest(paths, str(video), chat=chat_file, profile=profile, log=log)
        if vod:
            project.update_meta(title=vod.title, channel=vod.channel, url=vod.url,
                                started_at=int(vod.started.timestamp()), twitch_id=vod.id,
                                chapters=vod.chapters)
        settings = pipeline.Settings(mode=job.mode)
        pipeline.process(paths, project, settings, until=ANALYSIS_UNTIL, log=log,
                         progress=lambda stage: (lambda f, t: step(stage, f, t)))

    list_dir = Path(job.list_dir) if job.list_dir else video.parent
    sync.remember_video(project, video, list_dir)
    step("games", 0.0)
    if job.find_games:
        found = games.detect(paths, project, lambda f, t: step("games", f, t), log)
    else:
        found = []
        log("games: not looked for (turned off in Settings)")
    step("list", 0.0)
    stream_list = listing.build(project, found, video, per_hour=job.per_hour, chat_file=chat_file)
    if vod:  # the VOD's own details win over whatever the project had
        stream_list.title, stream_list.channel, stream_list.url = vod.title, vod.channel, vod.url
    stream_list.others = sync.others(project)
    txt, page = output.write(stream_list, list_dir)
    result = {"type": "done", "video": str(video), "text": str(txt), "html": str(page),
              "project": project.id, "games": stream_list.games,
              "lines": sum(len(b.lines) for b in stream_list.blocks)}
    emit(result)
    return result


def relist(project_ref: str, video: Path | None, per_hour: int, emit: Emit, *,
           find_games: bool = True) -> dict:
    """Write the list again from a processed project (after changing the list's settings)."""
    from stream_log.engine.config import get_paths
    from stream_log.engine.project import open_project

    paths = get_paths()
    project = sync._project_for(paths, project_ref) if Path(project_ref).suffix \
        else open_project(paths, project_ref)
    if video is None:
        known = sync._list_videos(project)
        video = Path(known[-1]) if known else project.source_path
    sync.remember_video(project, video)
    found = games.detect(paths, project, log=lambda t: emit({"type": "log", "text": t})) \
        if find_games else []
    vod_id = twitch.vod_id(project.meta.get("url") or "") or project.meta.get("twitch_id")
    chat_file = paths.cache / "twitch-chat" / f"{vod_id}.json" if vod_id else None
    stream_list = listing.build(project, found, video, per_hour=per_hour, chat_file=chat_file)
    stream_list.others = sync.others(project)
    txt, page = output.write(stream_list, sync.list_dir_for(project, video))
    result = {"type": "done", "video": str(video), "text": str(txt), "html": str(page),
              "project": project.id, "games": stream_list.games}
    emit(result)
    return result
