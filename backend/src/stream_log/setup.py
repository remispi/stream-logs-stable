"""What this machine still needs before a run, and downloading it (first run of the installed app).

The speech, voice, sound, text and vision models, the LLMs of the chosen mode, llama.cpp and (on
NVIDIA GPUs) the CUDA libraries. Downloads go to the data folder (Settings).
"""

from __future__ import annotations

from collections.abc import Callable


def needed_models(preset, mode: str) -> list[str]:
    from stream_log.engine.analysis import (
        SOUND_EVENT_MODEL,
        TEXT_EMBED_MODEL,
        TEXT_EMOTION_MODEL,
        VISION_MODEL,
    )
    from stream_log.engine.pipeline import ALIGN_MODEL, SPEAKER_MODEL

    ids = [preset.asr, ALIGN_MODEL, SPEAKER_MODEL, SOUND_EVENT_MODEL, TEXT_EMBED_MODEL,
           TEXT_EMOTION_MODEL, VISION_MODEL, *preset.mode_roles(mode).get("events", [])]
    return list(dict.fromkeys(i for i in ids if i))


def plan(paths, mode: str | None = None):
    from stream_log.engine import first_run
    from stream_log.engine.hardware import choose_tier, detect
    from stream_log.engine.models import load_manifest

    info = detect(paths.home)
    tier = choose_tier(info)
    manifest = load_manifest(paths.home)
    preset = manifest.tiers[tier]
    if mode not in preset.modes:
        mode = preset.default_mode
    todo = first_run.plan(paths, info, tier)
    need = needed_models(preset, mode)
    todo.models = [m for m in todo.models if m in need]
    todo.models_gb = round(sum(manifest.get(m).size_gb for m in todo.models), 1)
    return todo, {"tier": str(tier), "mode": mode, "modes": list(preset.modes),
                  "default_mode": preset.default_mode, "free_disk_gb": round(info.free_disk_gb),
                  "gpu": getattr(info.best_gpu, "name", None)}


def status(paths, mode: str | None = None) -> dict:
    from stream_log.run import default_profile

    todo, facts = plan(paths, mode)
    profiles = sorted(p.name for p in paths.profiles.glob("*") if p.is_dir()) \
        if paths.profiles.exists() else []
    return {"type": "status", "data_home": str(paths.home), **facts,
            "ready": todo.empty, "missing": todo.to_json(),
            "download_gb": round(todo.download_gb, 1),
            "profiles": profiles, "default_profile": default_profile(paths)}


def run_setup(paths, mode: str | None, emit: Callable[[dict], None]) -> None:
    from stream_log.engine import first_run

    todo, facts = plan(paths, mode)
    if todo.empty:
        emit({"type": "done", "text": "Nothing to download."})
        return
    totals: dict[str, int] = {}

    def progress(label: str, done: int, total: int) -> None:
        totals[label] = total
        emit({"type": "step", "id": "setup", "fraction": done / total if total else None,
              "text": f"{label}: {done / 1e9:.1f} / {total / 1e9:.1f} GB"})

    first_run.run(paths, todo, progress, say=lambda text: emit({"type": "log", "text": text}))
    emit({"type": "done", "text": f"Setup done ({facts['mode']} mode)."})
