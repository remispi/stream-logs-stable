"""The later analysis stages: signals -> modes -> fusion/vibe -> chapters -> llm -> events.

Registered in `pipeline.STAGES` after the transcript and voice stages, with the same caching rules.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from stream_log.engine.ingest import chat as chatlib
from stream_log.engine.models import load_manifest, require
from stream_log.engine.project import Project, read_json, write_json

TEXT_EMBED_MODEL = "bge-small-en"
TEXT_EMOTION_MODEL = "go-emotions"
SOUND_EVENT_MODEL = "audio-events"
VISION_MODEL = "siglip-base"


def save_npz(project: Project, name: str, arrays: dict[str, np.ndarray]) -> None:
    tmp = project.path(name + ".part.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
    tmp.replace(project.path(name))


def load_npz(project: Project, name: str) -> dict[str, np.ndarray]:
    with np.load(project.path(name)) as data:
        return {k: data[k] for k in data.files}


def duration_seconds(project: Project) -> int:
    return int(np.ceil(read_json(project.path("transcript.json"))["duration"]))


def speaker_tags(project: Project) -> dict[int, str]:
    path = project.path("speakers.json")
    return {e["id"]: e["tag"] for e in read_json(path)["lines"]} if path.exists() else {}


@cache
def _model_dir(models_root, model_id):
    return require(load_manifest().get(model_id), models_root)


def text_embedder(ctx):
    from stream_log.engine.onnx_models import TextEmbedder

    return TextEmbedder(_model_dir(ctx.paths.models, TEXT_EMBED_MODEL))


# ---------------------------------------------------------------------- audio signals


def audio_signals_run(ctx) -> None:
    from stream_log.engine.signals.audio import SoundEvents, prosody, speech_signals

    project = ctx.project
    duration = duration_seconds(project)
    wav = project.path("audio.wav")
    lines = read_json(project.path("transcript.json"))["lines"]
    ctx.progress(0.0, "loudness and pitch")
    signals = prosody(wav, duration)
    signals |= speech_signals(lines, speaker_tags(project), duration)
    events = SoundEvents(_model_dir(ctx.paths.models, SOUND_EVENT_MODEL))
    signals |= events.run(wav, duration, ctx.progress)
    save_npz(project, "audio_signals.npz", signals)


# ---------------------------------------------------------------------- chat signals


def chat_signals_config(ctx) -> dict:
    return {"chat": (ctx.project.meta.get("chat") or {}).get("imported")}


def chat_signals_run(ctx) -> None:
    from stream_log.engine.onnx_models import TextEmotions
    from stream_log.engine.signals.chat import chat_signals, find_interactions

    project = ctx.project
    duration = duration_seconds(project)
    messages = chatlib.read_saved_chat(project.path("chat.json"))
    lines = read_json(project.path("transcript.json"))["lines"]
    if messages:
        embedder = text_embedder(ctx)
        emotions = TextEmotions(_model_dir(ctx.paths.models, TEXT_EMOTION_MODEL))
    else:
        embedder = emotions = None
    signals = chat_signals(messages, duration, embedder, emotions, ctx.progress)
    save_npz(project, "chat_signals.npz", signals)
    write_json(project.path("chat_interactions.json"),
               find_interactions(lines, speaker_tags(project), messages))


# ---------------------------------------------------------------------- visual + modes


def visual_outputs(ctx) -> list[str]:
    return ["visual.npz", "segments.json"]


def visual_run(ctx) -> None:
    from stream_log.engine.onnx_models import Siglip
    from stream_log.engine.signals.visual import (
        MODE_STEP,
        MODES,
        apply_audio_cues,
        frozen_spans,
        gray_frames,
        mode_probabilities,
        motion_signals,
        rgb_frames,
        segments_from_path,
        visual_event_probs,
        viterbi,
    )

    project = ctx.project
    duration = duration_seconds(project)
    audio = load_npz(project, "audio_signals.npz")
    proxy = project.path("proxy.mp4")
    steps = int(np.ceil(duration / MODE_STEP))
    visual: dict[str, np.ndarray] = {}
    frozen: list[tuple[int, int]] = []
    if proxy.exists():
        ctx.progress(0.05, "motion")
        visual |= motion_signals(gray_frames(proxy), duration)
        frozen = frozen_spans(visual["motion"])
        ctx.progress(0.2, "SigLIP frames")
        siglip = Siglip(_model_dir(ctx.paths.models, VISION_MODEL))
        frames = rgb_frames(proxy)[:steps]
        probs, images = mode_probabilities(siglip, frames)
        for name, values in visual_event_probs(siglip, images).items():
            visual[f"visual_{name}"] = values
        if len(probs) < steps:  # proxy ended early: repeat the last frame's guess
            pad = probs[-1:] if len(probs) else np.full((1, len(MODES)), 1 / len(MODES))
            probs = np.concatenate([probs, np.repeat(pad, steps - len(probs), axis=0)])
    else:
        probs = np.full((steps, len(MODES)), 1 / len(MODES), dtype=np.float32)
        probs[:, MODES.index("chatting")] = 0.6  # audio only: assume talking
        probs /= probs.sum(axis=1, keepdims=True)
        visual["motion"] = np.zeros(duration, dtype=np.float32)
    probs = apply_audio_cues(probs, audio["speech"], audio["music"])
    segments = segments_from_path(viterbi(probs), probs, duration)
    visual["mode_probs"] = probs
    save_npz(project, "visual.npz", visual)
    write_json(project.path("segments.json"), {"step": MODE_STEP, "modes": MODES,
                                               "segments": segments, "frozen": frozen})


# ---------------------------------------------------------------------- fusion + vibe


def fusion_run(ctx) -> None:
    from stream_log.engine.chapters import lore
    from stream_log.engine.signals.chat import chat_reliability, estimate_delay, shift_back
    from stream_log.engine.signals.fusion import arousal, fuse, vibe_mix, vibe_windows

    project = ctx.project
    signals = load_npz(project, "audio_signals.npz")
    chat = load_npz(project, "chat_signals.npz")
    visual = load_npz(project, "visual.npz")
    duration = len(signals["loudness"])
    energy = arousal(signals | {"speech": signals["speech"]})
    delay, corr = estimate_delay(chat["chat_rate"], energy)
    for key, values in chat.items():
        signals[key] = shift_back(values[:duration], delay)
    signals["motion"] = visual["motion"][:duration]
    reliability = chat_reliability(chat["chat_unique"])
    signals["fused"], weights = fuse(signals, reliability)
    signals["arousal"] = energy
    save_npz(project, "signals.npz", signals)

    segments = read_json(project.path("segments.json"))["segments"]
    interactions = read_json(project.path("chat_interactions.json"))
    windows = vibe_windows(signals, segments, interactions, reliability)
    messages = chatlib.read_saved_chat(project.path("chat.json"))
    lines = read_json(project.path("transcript.json"))["lines"]
    write_json(project.path("vibe.json"), {
        "chat_delay": delay, "chat_delay_correlation": round(corr, 3),
        "chat_reliability": round(reliability, 3), "weights": weights,
        "mix": vibe_mix(windows), "windows": windows,
        "lore": lore(lines, [m.text for m in messages if m.kind == "message"]),
    }, indent=None)


# ---------------------------------------------------------------------- chapters


def _channel_note(ctx) -> str:
    profile = ctx.project.meta.get("profile")
    meta = ctx.project.meta
    parts = []
    if meta.get("channel"):
        parts.append(f"Channel: {meta['channel']}.")
    if profile:
        from stream_log.engine.speakers.profiles import ProfileError, load_profile

        try:
            description = load_profile(ctx.paths, profile).get("description")
        except ProfileError:
            description = None
        if description:
            parts.append(f"About the channel: {description}")
    return " ".join(parts)


def _voices(ctx) -> tuple[dict[int, str], str]:
    """For the LLM stages: each line's speaker label, a
    voice's name when it has one, and the note saying who is who (speakers/names.py)."""
    from stream_log.engine.speakers.names import speaker_labels, voice_names, voices_note

    path = ctx.project.path("speakers.json")
    if not path.exists():
        return {}, ""
    speakers = read_json(path)
    names = voice_names(ctx.paths, ctx.project)
    return speaker_labels(speakers, names), voices_note(speakers, names)


def _with_voices(ctx, config: dict) -> dict:
    """Naming a voice re-runs the stage; with no names the key stays what it was."""
    _, note = _voices(ctx)
    return {**config, "voices": note} if note else config


def _note_and_voices(ctx, voices: str) -> str:
    return " ".join(part for part in (_channel_note(ctx), voices) if part)


def chapters_run(ctx) -> None:
    """Chapter boundaries and keyword titles; the `llm` stage adds titles and summaries."""
    from stream_log.engine.chapters import build_chapters

    project = ctx.project
    lines = read_json(project.path("transcript.json"))["lines"]
    signals = load_npz(project, "signals.npz")
    segments = read_json(project.path("segments.json"))["segments"]
    chapters = build_chapters(lines, len(signals["fused"]), text_embedder(ctx), segments,
                              project.meta.get("chapters") or [], signals["fused"])
    write_json(project.path("chapters.json"), {"chapters": chapters})


# ---------------------------------------------------------------------- LLM panel


def llm_config(ctx) -> dict:
    from stream_log.engine.panel import CHECK_VERSION, PROMPT_VERSION

    return {"panel": [f"{m.id}:{m.identity}" for m in ctx.panel.events], "prompt": PROMPT_VERSION,
            "check": CHECK_VERSION,
            "note": _channel_note(ctx)}


def llm_run(ctx) -> None:
    from stream_log.engine.panel import run_panel

    project = ctx.project
    result: dict = {"models": [], "chapters": {}, "summary": None, "events": [], "rejected": [],
                    "stats": {}, "errors": []}
    if ctx.panel.events:
        result = run_panel(
            ctx.panel.events,
            read_json(project.path("chapters.json"))["chapters"],
            read_json(project.path("transcript.json"))["lines"],
            speaker_tags(project),
            chatlib.read_saved_chat(project.path("chat.json")),
            note=_channel_note(ctx),
            cache_dir=project.path("llm_runs"),
            progress=ctx.progress,
            log=ctx.log,
        )
        stats = result["stats"]
        ctx.log(f"LLM panel: {len(result['events'])} events kept "
                f"({stats.get('agreed', 0)} found by 2+ models), "
                f"{stats.get('claims_rejected', 0)} of {stats.get('claims_checked', 0)} "
                "cross-checked claims rejected")
        for error in result["errors"][:10]:
            ctx.log(f"[yellow]LLM call failed[/] {error}")
    write_json(project.path("llm.json"), result)


def final_chapters(project: Project) -> tuple[list[dict], str | None, dict]:
    """Chapters with the panel's checked titles/summaries, the stream summary and llm.json."""
    chapters = read_json(project.path("chapters.json"))["chapters"]
    path = project.path("llm.json")
    llm = read_json(path) if path.exists() else {"models": [], "chapters": {}, "summary": None}
    for chapter in chapters:
        chapter.update(llm["chapters"].get(chapter["id"], {}))
    return chapters, llm.get("summary"), llm


# ---------------------------------------------------------------------- events


def events_config(ctx) -> dict:
    profile = ctx.project.meta.get("profile")
    custom = []
    if profile:
        from stream_log.engine.speakers.profiles import ProfileError, load_profile

        try:
            custom = load_profile(ctx.paths, profile).get("custom_events", [])
        except ProfileError:
            custom = []
    return {"custom": custom}


def events_run(ctx) -> None:
    from stream_log.engine import events as ev
    from stream_log.engine.signals.visual import MODE_STEP

    project = ctx.project
    lines = read_json(project.path("transcript.json"))["lines"]
    tags = speaker_tags(project)
    speakers = read_json(project.path("speakers.json"))
    messages = chatlib.read_saved_chat(project.path("chat.json"))
    signals = load_npz(project, "signals.npz")
    visual = load_npz(project, "visual.npz")
    seg_data = read_json(project.path("segments.json"))
    segments = seg_data["segments"]
    chapters, _, llm = final_chapters(project)
    vibe = read_json(project.path("vibe.json"))
    interactions = read_json(project.path("chat_interactions.json"))
    duration = float(len(signals["fused"]))
    twitch_chapters = project.meta.get("chapters") or []
    by_id = {line["id"]: line for line in lines}
    tts_lines = [by_id[i] for i, tag in tags.items() if tag == "tts" and i in by_id]

    found = []
    found += ev.structure_events(segments, twitch_chapters, signals["speech"], duration)
    found += ev.community_events(messages, tts_lines)
    found += ev.chat_moment_events(signals, vibe["chat_reliability"])
    visual_probs = {k.removeprefix("visual_"): v for k, v in visual.items()
                    if k.startswith("visual_")}
    found += ev.gameplay_events(visual_probs, MODE_STEP, segments)
    found += ev.health_events(signals, [tuple(f) for f in seg_data["frozen"]], segments, messages)
    found += ev.announcement_events(lines, tags)
    found += ev.conversation_events(interactions, speakers, lines)
    found += ev.custom_events({"custom_events": events_config(ctx)["custom"]}, lines, messages)
    found += llm["events"]
    events = ev.finish(ev.merge_events(found), chapters)
    write_json(project.path("events.json"), {"events": events, "llm_models": llm["models"]})


# ---------------------------------------------------------------------- moments (Stage B)
