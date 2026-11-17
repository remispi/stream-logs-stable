"""The analysis: ingest -> media -> transcribe -> align -> words -> speakers -> audio/chat/visual
signals -> fusion -> chapters -> llm -> events (the later stages are in `analysis.py`).

Every stage is cached in the project folder (see `project.py`); `process` only re-runs what is
stale.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from stream_log.engine import analysis as A
from stream_log.engine.audio import extract_wav, read_wav
from stream_log.engine.config import Paths
from stream_log.engine.hardware import choose_tier, detect
from stream_log.engine.ingest import chat as chatlib
from stream_log.engine.ingest import sources
from stream_log.engine.llm import find_llm
from stream_log.engine.media import make_proxy, make_sprite, probe, waveform_peaks
from stream_log.engine.models import load_manifest, require
from stream_log.engine.models.manifest import ROLES
from stream_log.engine.panel import PanelMember, external_member, managed_member
from stream_log.engine.project import Project, create_project, read_json, stage_key, write_json
from stream_log.engine.speakers.profiles import profile_dir

ALIGN_MODEL = "wav2vec2-base-960h"
SPEAKER_MODEL = "wespeaker-resnet34"
VERBATIM_MODELS = {"crisperwhisper"}

Progress = Callable[[float, str], None]
Log = Callable[[str], None]


def _quiet(*_: object) -> None:
    pass


# ---------------------------------------------------------------------- ingest


def ingest(
    paths: Paths,
    ref: str,
    *,
    chat: Path | None = None,
    chat_start: datetime | None = None,
    profile: str | None = None,
    audio_track: int = 0,
    progress: Progress = _quiet,
    log: Log = _quiet,
) -> Project:
    source = sources.classify(ref)
    project = create_project(paths, source.project_id, {"source": {"kind": "local", "ref": ref}})
    source_path = Path(source.ref)

    info = probe(source_path)
    if info.audio_tracks == 0:
        raise ValueError(f"{source_path} has no audio track")
    if audio_track >= info.audio_tracks:
        raise ValueError(f"audio track {audio_track} requested, file has {info.audio_tracks}")
    project.update_meta(
        source_path=str(source_path),
        fingerprint=sources.fingerprint(source_path),
        media=info.to_dict(),
        audio_track=audio_track,
        **({"profile": profile} if profile else {}),
    )

    if chat is not None:
        import_chat(project, chat, chat_start=chat_start, log=log)
    return project


def import_chat(
    project: Project, chat_file: Path, *, chat_start: datetime | None = None, log: Log = _quiet
) -> int:
    meta = project.meta
    if chat_start is None and meta.get("started_at"):
        chat_start = datetime.fromtimestamp(meta["started_at"])  # local time, as Chatterino logs
    fmt, messages = chatlib.load_chat(
        chat_file, vod_start=chat_start, duration=meta["media"]["duration"]
    )
    chatlib.save_chat(project.path("chat.json"), fmt, str(chat_file), messages)
    project.update_meta(chat={"format": fmt, "origin": str(chat_file), "messages": len(messages),
                              "imported": time.time()})
    log(f"chat: {len(messages)} messages ({fmt})")
    return len(messages)


# ---------------------------------------------------------------------- processing


@dataclass
class Settings:
    asr_model: str | None = None  # None: the tier's verbatim model if downloaded, else its ASR
    device: str = "auto"
    proxy: bool = True
    align: bool = True
    batch_size: int = 12
    force: set[str] = field(default_factory=set)
    use_llm: bool = True
    mode: str | None = None  # fast | balanced | best; default: the tier's
    llm_models: list[str] | None = None  # every role uses these instead of the mode's models
    llm_exe: Path | None = None  # llama-server; default: STREAM_LOG_LLAMA_SERVER, PATH, runtimes/
    llm_url: str | None = None  # use a server you started instead of managing models
    llm_model: str | None = None


@dataclass
class Panel:
    """The LLMs of a run by role; one member can have several roles (loaded once per stage)."""

    events: list[PanelMember] = field(default_factory=list)


@dataclass
class StageContext:
    paths: Paths
    project: Project
    settings: Settings
    progress: Progress
    log: Log
    panel: Panel = field(default_factory=Panel)


@dataclass
class Stage:
    name: str
    version: int
    outputs: Callable[[StageContext], list[str]]
    config: Callable[[StageContext], dict]
    run: Callable[[StageContext], None]


def resolve_asr_model(paths: Paths, requested: str | None) -> str:
    manifest = load_manifest()
    if requested:
        manifest.get(requested)
        return requested
    preset = manifest.tiers[choose_tier(detect(paths.home))]
    for candidate in (preset.verbatim_asr, preset.asr):
        if manifest.get(candidate).is_downloaded(paths.models):
            return candidate
    return preset.verbatim_asr


# --- media


def _media_outputs(ctx: StageContext) -> list[str]:
    has_video = ctx.project.meta["media"].get("width") is not None
    visual = ["proxy.mp4", "sprite.jpg"] if ctx.settings.proxy and has_video else []
    return ["audio.wav", "peaks.json", *visual]


def _media_config(ctx: StageContext) -> dict:
    meta = ctx.project.meta
    return {"fingerprint": meta["fingerprint"], "audio_track": meta.get("audio_track", 0),
            "proxy": ctx.settings.proxy}


def _media_run(ctx: StageContext) -> None:
    project, meta = ctx.project, ctx.project.meta
    source = project.source_path
    ctx.progress(0.0, "extracting audio")
    extract_wav(source, project.path("audio.wav"), audio_track=meta.get("audio_track", 0))
    write_json(project.path("peaks.json"), waveform_peaks(project.path("audio.wav")), indent=None)
    if "proxy.mp4" in _media_outputs(ctx):
        ctx.progress(0.2, "encoding 540p proxy")
        from stream_log.engine.media import MediaInfo

        info = MediaInfo(**meta["media"])
        encoder = make_proxy(source, project.path("proxy.mp4"), info,
                             audio_track=meta.get("audio_track", 0))
        ctx.progress(0.9, "thumbnail sprite")
        sprite = make_sprite(project.path("proxy.mp4"), project.path("sprite.jpg"), info.duration)
        write_json(project.path("sprite.json"), sprite)
        ctx.log(f"proxy encoded with {encoder}")


# --- transcribe


def _transcribe_config(ctx: StageContext) -> dict:
    return {"model": resolve_asr_model(ctx.paths, ctx.settings.asr_model),
            "batch_size": ctx.settings.batch_size}


def _transcribe_run(ctx: StageContext) -> None:
    from stream_log.engine.asr.transcribe import transcribe

    model_id = resolve_asr_model(ctx.paths, ctx.settings.asr_model)
    model_dir = require(load_manifest().get(model_id), ctx.paths.models)
    ctx.log(f"ASR model: {model_id}")
    result = transcribe(
        ctx.project.path("audio.wav"), model_dir,
        verbatim_model=model_id in VERBATIM_MODELS,
        device=ctx.settings.device, batch_size=ctx.settings.batch_size, progress=ctx.progress,
    )
    result["model"] = model_id
    write_json(ctx.project.path("transcript.raw.json"), result, indent=None)


# --- align (+ probable fillers from unworded speech)


def _align_run(ctx: StageContext) -> None:
    from stream_log.engine.asr.align import align_transcript
    from stream_log.engine.asr.fillers import insert_gap_fillers

    data = read_json(ctx.project.path("transcript.raw.json"))
    lines = data["lines"]
    if ctx.settings.align and lines:
        model_dir = require(load_manifest().get(ALIGN_MODEL), ctx.paths.models)
        rejected = align_transcript(
            ctx.project.path("audio.wav"), lines, model_dir, data["duration"], ctx.progress
        )
        data["aligned"] = ALIGN_MODEL
        ctx.log(f"aligned {len(lines) - rejected}/{len(lines)} lines "
                f"({rejected} kept Whisper timing)")
    else:
        data["aligned"] = None
    whisper_spans = [(w["start"], w["end"]) for line in
                     read_json(ctx.project.path("transcript.raw.json"))["lines"]
                     for w in line["words"]]
    data["unrecognised"] = insert_gap_fillers(lines, [tuple(s) for s in data.pop("speech")],
                                              whisper_spans)
    fillers = sum(w["filler"] for line in lines for w in line["words"])
    probable = sum(bool(w.get("probable")) for line in lines for w in line["words"])
    data["stats"] = {"lines": len(lines), "words": sum(len(x["words"]) for x in lines),
                     "fillers": fillers, "probable_fillers": probable}
    write_json(ctx.project.path("transcript.json"), data, indent=None)


# --- words: is each heard word a normal word, a game term, a name? (wordcheck.py)


def _words_config(ctx: StageContext) -> dict:
    from stream_log.engine import wordcheck as W
    from stream_log.engine.lexicon import (
        GAME_FILE,
        GLOSSARY,
        NAMES_FILE,
        common_words,
        lexicon_path,
    )

    return {"prompt": W.PROMPT_VERSION,
            "model": [f"{m.id}:{m.identity}" for m in ctx.panel.events[:1]],
            "glossary": W.file_digest(ctx.paths.home / GLOSSARY),
            "game": W.file_digest(lexicon_path(ctx.paths, GAME_FILE)),
            "names": W.file_digest(lexicon_path(ctx.paths, NAMES_FILE)),
            "vocabulary": len(common_words(ctx.paths.models)), "note": A._channel_note(ctx)}


def _words_run(ctx: StageContext) -> None:
    from contextlib import ExitStack

    from stream_log.engine import wordcheck as W
    from stream_log.engine.lexicon import (
        GAME_FILE,
        GLOSSARY,
        NAMES_FILE,
        common_words,
        lexicon_path,
        read_glossary,
        read_list,
    )
    from stream_log.engine.project import read_cache

    project = ctx.project
    data = read_json(project.path("transcript.json"))
    lines = data["lines"]
    W.restore(lines)
    lexicon = W.build_lexicon(read_glossary(ctx.paths.home / GLOSSARY),
                              read_list(lexicon_path(ctx.paths, GAME_FILE)),
                              read_list(lexicon_path(ctx.paths, NAMES_FILE)),
                              common_words(ctx.paths.models))
    candidates = W.find(lines, lexicon)
    member = ctx.panel.events[0] if ctx.panel.events else None
    errors: list[str] = []
    with ExitStack() as stack:
        ask = None
        if member is not None:
            cache = project.path(f"words_runs/{member.id}.json")
            stored = read_cache(cache)
            llm = None

            def ask(prompt: str) -> dict:
                nonlocal llm
                key = W.cache_key(f"{member.id}:{member.identity}", prompt)
                if key not in stored:
                    llm = llm or stack.enter_context(member.session())
                    stored[key] = llm.chat_json(W.SYSTEM, prompt, W.SCHEMA, max_tokens=1500)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    write_json(cache, stored, indent=None)
                return stored[key]

        decisions = W.decide(lines, candidates, ask, note=A._channel_note(ctx),
                             progress=ctx.progress, errors=errors)
    changed = W.apply(lines, decisions)
    write_json(project.path("transcript.json"), data, indent=None)
    counts = W.summary(decisions)
    write_json(project.path("words.json"), {
        "model": member.id if member else None, "lexicon": {
            "game": len(lexicon.game), "names": len(lexicon.names),
            "ordinary": len(lexicon.common)},
        "counts": counts, "errors": errors[:20], "decisions": decisions})
    ctx.log(f"words: {len(candidates)} checked ({counts['game']} game terms, {counts['name']} "
            f"names, {counts['normal']} normal, {counts['other']} other); {changed} corrected"
            + (f" by {member.id}" if member else " by rule (no LLM)"))
    for error in errors[:5]:
        ctx.log(f"[yellow]LLM call failed[/] {error}")


# --- speakers


def _speakers_config(ctx: StageContext) -> dict:
    meta = ctx.project.meta
    profile = meta.get("profile")
    voice_file = profile_dir(ctx.paths, profile) / "voice.npy" if profile else None
    voice_stamp = voice_file.stat().st_mtime if voice_file and voice_file.exists() else None
    overrides_file = ctx.project.path("speaker_labels.json")
    return {
        "profile": profile,
        "voice": voice_stamp,
        "chat": (meta.get("chat") or {}).get("imported"),
        "labels": read_json(overrides_file) if overrides_file.exists() else {},
    }


def line_embeddings(project: Project) -> dict[int, np.ndarray]:
    path = project.path("speaker_embeddings.npz")
    if not path.exists():
        return {}
    data = np.load(path)
    return dict(zip(data["ids"].tolist(), data["vectors"], strict=True))


def _speakers_run(ctx: StageContext) -> None:
    from stream_log.engine.speakers.embed import SpeakerEmbedder
    from stream_log.engine.speakers.profiles import load_voice
    from stream_log.engine.speakers.tagging import tag_lines

    project = ctx.project
    lines = read_json(project.path("transcript.json"))["lines"]
    embedder = SpeakerEmbedder(require(load_manifest().get(SPEAKER_MODEL), ctx.paths.models))
    wav = project.path("audio.wav")
    ids, vectors = [], []
    for index, line in enumerate(lines):
        vector = embedder.embed(read_wav(wav, line["start"], line["end"]))
        if vector is not None:
            ids.append(line["id"])
            vectors.append(vector)
        if index % 25 == 0:
            ctx.progress((index + 1) / max(len(lines), 1), f"line {index + 1} / {len(lines)}")
    np.savez(project.path("speaker_embeddings.npz"), ids=np.array(ids, dtype=int),
             vectors=np.array(vectors, dtype=np.float32).reshape(-1, 256))

    config = _speakers_config(ctx)
    result = tag_lines(
        lines, dict(zip(ids, vectors, strict=True)),
        voice=load_voice(ctx.paths, config["profile"]),
        chat=chatlib.read_saved_chat(project.path("chat.json")),
        overrides=config["labels"],
    )
    result["profile"] = config["profile"]
    write_json(project.path("speakers.json"), result, indent=None)


def _no_config(ctx: StageContext) -> dict:
    return {}


def _outputs(*names: str) -> Callable[[StageContext], list[str]]:
    return lambda ctx: list(names)


STAGES = [
    Stage("media", 1, _media_outputs, _media_config, _media_run),
    Stage("transcribe", 1, _outputs("transcript.raw.json"), _transcribe_config, _transcribe_run),
    Stage("align", 2, _outputs("transcript.json"), lambda ctx: {"align": ctx.settings.align},
          _align_run),
    Stage("words", 1, _outputs("words.json"), _words_config, _words_run),
    Stage("speakers", 3, _outputs("speakers.json", "speaker_embeddings.npz"), _speakers_config,
          _speakers_run),
    Stage("audio_signals", 1, _outputs("audio_signals.npz"), _no_config, A.audio_signals_run),
    Stage("chat_signals", 1, _outputs("chat_signals.npz", "chat_interactions.json"),
          A.chat_signals_config, A.chat_signals_run),
    Stage("visual", 1, A.visual_outputs,
          lambda ctx: {"proxy": ctx.project.path("proxy.mp4").exists()}, A.visual_run),
    Stage("fusion", 1, _outputs("signals.npz", "vibe.json"), _no_config, A.fusion_run),
    Stage("chapters", 2, _outputs("chapters.json"), _no_config, A.chapters_run),
    Stage("llm", 1, _outputs("llm.json"), A.llm_config, A.llm_run),
    Stage("events", 2, _outputs("events.json"), A.events_config, A.events_run),
]
STAGE_NAMES = [stage.name for stage in STAGES]


def resolve_panel(paths: Paths, project: Project, settings: Settings, log: Log) -> Panel:
    """The LLMs this run uses, by role: the mode's models from the manifest (or
    `settings.llm_models` for every role), else a server already running."""
    from stream_log.engine.llm_server import find_exe

    if settings.llm_url:
        client = find_llm(settings.llm_url, settings.llm_model)
        if client is None:
            log(f"LLM: nothing answers at {settings.llm_url}, LLM steps skipped")
            return Panel()
        log(f"LLM: {client.label} (your server)")
        return _alone(external_member(client))
    manifest = load_manifest()
    tier = choose_tier(detect(paths.home))
    preset = manifest.tiers[tier]
    mode = settings.mode or preset.default_mode
    if mode not in preset.modes:
        log(f"LLM: the {tier.value} tier has no {mode} mode, using {preset.default_mode}")
        mode = preset.default_mode
    roles = ({role: list(settings.llm_models) for role in ROLES} if settings.llm_models
             else preset.mode_roles(mode))
    wanted = list(dict.fromkeys(i for ids in roles.values() for i in ids))
    entries = {i: manifest.get(i) for i in wanted}
    missing = [i for i in wanted if not entries[i].is_downloaded(paths.models)]
    if missing:
        log(f"LLM: not downloaded, skipped: {', '.join(missing)} (Settings > first-time setup)")
    available = [i for i in wanted if i not in missing]
    try:
        exe = find_exe(settings.llm_exe, paths.home)
    except FileNotFoundError as error:
        exe = None
        if available:
            log(f"LLM: {error}")
    members = {}
    for i in available if exe else []:
        try:
            members[i] = managed_member(entries[i], paths.models, exe, project.path("llm_runs"))
        except FileNotFoundError as error:  # e.g. the folder holds no model GGUF
            log(f"LLM: {error}, skipped")
    available = [i for i in available if i in members]
    if members:
        # A role whose models are all missing falls back to the first model that is there.
        panel = Panel(**{role: [members[i] for i in ids if i in members]
                         or [members[available[0]]] for role, ids in roles.items()})
        log(f"LLM ({'chosen models' if settings.llm_models else mode}): "
            + "; ".join(f"{role} {' + '.join(m.id for m in getattr(panel, role))}"
                        for role in ROLES) + f" via {exe}")
        return panel
    client = find_llm()
    if client is not None:
        log(f"LLM: {client.label} (already running)")
        return _alone(external_member(client))
    log("LLM: no models or server available, LLM steps skipped")
    return Panel()


def _alone(member: PanelMember) -> Panel:
    return Panel([member])


def process(
    paths: Paths,
    project: Project,
    settings: Settings | None = None,
    *,
    until: str | None = None,
    stages: list[Stage] | None = None,
    progress: Callable[[str], Progress] | None = None,
    log: Log = _quiet,
) -> list[tuple[str, str, float]]:
    """Run stale stages in order. Returns (stage, "ran" | "cached", seconds) per stage."""
    settings = settings or Settings()
    stages = stages or STAGES
    panel = Panel()
    if settings.use_llm and stages is STAGES:
        panel = resolve_panel(paths, project, settings, log)
    report = []
    upstream: list[str] = []
    dirty = False  # once a stage has run, everything after it must run too
    for stage in stages:
        ctx = StageContext(paths, project, settings,
                           progress(stage.name) if progress else _quiet, log, panel)
        key = stage_key(stage.name, stage.version, stage.config(ctx), upstream)
        outputs = stage.outputs(ctx)
        forced = dirty or stage.name in settings.force
        if not forced and project.is_fresh(stage.name, key, outputs):
            report.append((stage.name, "cached", 0.0))
        else:
            started = time.perf_counter()
            stage.run(ctx)
            seconds = time.perf_counter() - started
            project.record_stage(stage.name, key, outputs, seconds)
            report.append((stage.name, "ran", seconds))
            dirty = True
        upstream.append(key)
        if stage.name == until:
            break
    return report
