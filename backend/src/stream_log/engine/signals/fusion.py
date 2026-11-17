"""Fused score and vibe vectors.

fused(t) = Σ w_i · s_i(t), with chat weights scaled by chat reliability. It's a ranking aid for
finding moments, never a gate: calm moments still come from the transcript.
"""

from __future__ import annotations

import numpy as np

from stream_log.engine.signals.chat import VIBES
from stream_log.engine.signals.common import nan0, robust_z, smooth

BASE_WEIGHTS = {
    "loudness": 0.15,
    "pitch_std": 0.10,
    "speech_rate": 0.05,
    "laughter": 0.15,
    "scream": 0.12,
    "cheer": 0.05,
    "motion": 0.03,
    "chat_rate": 0.12,
    "chat_unique": 0.06,
    "chat_emote": 0.04,
    "chat_convergence": 0.05,
    "chat_vibe_shift": 0.08,
}
CHAT_KEYS = {k for k in BASE_WEIGHTS if k.startswith("chat_")}
PROBABILITY_KEYS = {"laughter", "scream", "cheer"}  # already 0-1; scaled instead of z-scored

WINDOW = 30
ACTIVITIES = ["banter", "focused_gameplay", "storytelling", "explaining", "music", "afk_loading",
              "hype"]


def weights(chat_reliability: float) -> dict[str, float]:
    w = {k: v * (chat_reliability if k in CHAT_KEYS else 1.0) for k, v in BASE_WEIGHTS.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def fuse(signals: dict[str, np.ndarray], chat_reliability: float) -> tuple[np.ndarray, dict]:
    w = weights(chat_reliability)
    duration = len(signals["loudness"])
    fused = np.zeros(duration, dtype=np.float32)
    for key, weight in w.items():
        if key not in signals or weight == 0:
            continue
        x = signals[key]
        z = nan0(x) * 4.0 if key in PROBABILITY_KEYS else nan0(robust_z(x))
        fused += weight * np.clip(z, -3, 6)
    return smooth(fused, 5), w


def arousal(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Voice energy: loudness + pitch movement while someone is talking."""
    z = 0.6 * nan0(robust_z(signals["loudness"])) + 0.4 * nan0(robust_z(signals["pitch_std"]))
    return smooth(np.where(signals["speech"] > 0.2, z, np.minimum(z, 0)), 5)


def _mode_at(segments: list[dict], t: float) -> str:
    for segment in segments:
        if segment["start"] <= t < segment["end"]:
            return segment["mode"]
    return segments[-1]["mode"] if segments else "chatting"


def activity(window: dict) -> str:
    """Heuristic activity tag for a 30 s window (the LLM refines these when available)."""
    if window["speech"] < 0.1:
        return "music" if window["music"] > 0.4 else "afk_loading"
    if window["fused"] > 1.2 or window["laughter"] > 0.4 or window["scream"] > 0.4:
        return "hype"
    if window["mode"] == "gaming" and window["speech"] < 0.45:
        return "focused_gameplay"
    if window["streamer_speech"] > 0.7 and window["interactions"] == 0:
        return "storytelling"
    return "banter"


def vibe_windows(
    signals: dict[str, np.ndarray], segments: list[dict], interactions: list[dict],
    chat_reliability: float,
) -> list[dict]:
    duration = len(signals["loudness"])
    energy = arousal(signals)
    inter_seconds = np.zeros(duration)
    for item in interactions:
        inter_seconds[min(int(item["t"]), duration - 1)] += 1
    windows = []
    for start in range(0, duration, WINDOW):
        end = min(start + WINDOW, duration)
        part = slice(start, end)

        def mean(key: str, part: slice = part) -> float:
            return float(np.nanmean(signals[key][part])) if key in signals else 0.0

        chat_mix = {v: mean(f"vibe_{v}") for v in VIBES}
        total = sum(chat_mix.values())
        chat_mix = {k: v / total for k, v in chat_mix.items()} if total > 0 else {}
        window = {
            "start": start, "end": end,
            "mode": _mode_at(segments, start + WINDOW / 2),
            "energy": round(float(np.mean(energy[part])), 3),
            "fused": round(float(np.mean(signals["fused"][part])), 3),
            "speech": round(mean("speech"), 3),
            "streamer_speech": round(mean("streamer_speech"), 3),
            "laughter": round(float(np.max(signals["laughter"][part])), 3),
            "scream": round(float(np.max(signals["scream"][part])), 3),
            "music": round(mean("music"), 3),
            "interactions": int(inter_seconds[part].sum()),
            "chat_vibes": {k: round(v, 3) for k, v in chat_mix.items()},
        }
        window["activity"] = activity(window)
        window["vibe"] = dominant_vibe(window, chat_reliability)
        windows.append(window)
    return windows


def dominant_vibe(window: dict, chat_reliability: float) -> str:
    """One label per window for the stream's vibe mix: voice/audio first, chat as a tiebreak."""
    scores = {
        "hype": max(window["scream"], 0.0) + max(window["energy"], 0) * 0.3,
        "laughter": window["laughter"] * 1.5,
        "calm": 0.35 if window["energy"] < 0.5 else 0.1,
        "focused": 0.4 if window["activity"] == "focused_gameplay" else 0.0,
        "storytelling": 0.45 if window["activity"] == "storytelling" else 0.0,
        "community": min(window["interactions"] * 0.2, 0.6),
        "music_afk": 0.6 if window["activity"] in ("music", "afk_loading") else 0.0,
    }
    for vibe, share in window["chat_vibes"].items():
        target = {"hype": "hype", "laughter": "laughter", "shock": "hype", "tension": "hype",
                  "wholesome": "community", "sympathy": "community",
                  "conversation": "community"}.get(vibe)
        if target:
            scores[target] += share * 0.5 * chat_reliability
    return max(scores, key=scores.get)


def vibe_mix(windows: list[dict]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for window in windows:
        counts[window["vibe"]] = counts.get(window["vibe"], 0) + (window["end"] - window["start"])
    total = sum(counts.values()) or 1
    return {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
