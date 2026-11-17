"""Visual signals from the proxy: motion / scene change, frozen video, and stream modes.

Modes: SigLIP zero-shot on one frame every 10 s, combined with audio cues, then an
HMM (Viterbi with sticky transitions) so a single odd frame doesn't flip the mode.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from stream_log.engine.audio import ffmpeg_exe

MODE_STEP = 10
MODES = ["gaming", "chatting", "irl", "reaction", "static"]

MODE_PROMPTS = {
    "gaming": ["a screenshot of a video game", "video game gameplay with a hud and a small webcam",
               "a first person shooter game", "a video game menu screen"],
    "chatting": ["a person talking to a webcam in their room", ("a streamer sitting in front of a "
                 "microphone looking at the camera"), "a close up of a person at their desk"],
    "irl": ["a person filming outdoors with a phone", "a street scene filmed while walking",
            "a selfie video outside in a city", "people at a restaurant filmed on a phone"],
    "reaction": ["a person watching a youtube video on stream", ("a video playing in a browser "
                 "with a small webcam in the corner"), "a streamer reacting to a video"],
    "static": ["a starting soon screen", "a be right back screen", "a stream ending screen",
               "a static graphic with text and no people", "a black screen"],
}

# Gameplay events from single frames; each competes against a neutral prompt in a softmax.
VISUAL_EVENTS = {
    "death": "a game over screen that says you died",
    "victory": "a victory screen that says you win",
    "boss": "a boss fight with a big health bar at the top of the screen",
    "loading": "a loading screen",
}
NEUTRAL_PROMPT = "a screenshot of normal video game gameplay"

STAY = 0.97  # HMM probability of staying in the same mode for the next 10 s


def gray_frames(proxy: Path) -> np.ndarray:
    """1 frame per second (the proxy's keyframes) as 64x36 grayscale."""
    result = subprocess.run(
        [ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-skip_frame", "nokey", "-i", str(proxy),
         "-vf", "fps=1,scale=64:36,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 36, 64)


def rgb_frames(proxy: Path, step: int = MODE_STEP, size: int = 224) -> np.ndarray:
    result = subprocess.run(
        [ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-skip_frame", "nokey", "-i", str(proxy),
         "-vf", f"fps=1/{step},scale={size}:{size}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, size, size, 3)


def motion_signals(frames: np.ndarray, duration: int) -> dict[str, np.ndarray]:
    """Mean absolute frame difference (motion) and scene-cut indicator per second."""
    motion = np.zeros(duration, dtype=np.float32)
    if len(frames) > 1:
        diffs = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2)) / 255.0
        n = min(len(diffs), duration - 1)
        motion[1 : n + 1] = diffs[:n]
    baseline = np.median(motion) if len(motion) else 0.0
    cut = (motion > max(0.15, baseline * 4)).astype(np.float32)
    return {"motion": motion, "scene_cut": cut}


def frozen_spans(motion: np.ndarray, min_seconds: int = 10) -> list[tuple[int, int]]:
    """Runs where the picture doesn't change at all (encoder froze, game hung)."""
    still = motion < 0.0015
    spans, start = [], None
    for t, value in enumerate(still):
        if value and start is None:
            start = t
        elif not value and start is not None:
            if t - start >= min_seconds:
                spans.append((start, t))
            start = None
    if start is not None and len(still) - start >= min_seconds:
        spans.append((start, len(still)))
    return spans


def mode_probabilities(siglip, frames: np.ndarray) -> np.ndarray:
    prompts = [(mode, p) for mode in MODES for p in MODE_PROMPTS[mode]]
    text = siglip.embed_text([p for _, p in prompts])
    images = siglip.embed_images(frames)
    probs = siglip.prompt_probs(images, text)
    out = np.zeros((len(frames), len(MODES)), dtype=np.float32)
    for column, (mode, _) in enumerate(prompts):
        out[:, MODES.index(mode)] += probs[:, column]
    return out, images


def visual_event_probs(siglip, image_embs: np.ndarray) -> dict[str, np.ndarray]:
    names = list(VISUAL_EVENTS)
    text = siglip.embed_text([VISUAL_EVENTS[n] for n in names] + [NEUTRAL_PROMPT])
    probs = siglip.prompt_probs(image_embs, text)
    return {name: probs[:, i] for i, name in enumerate(names)}


def apply_audio_cues(probs: np.ndarray, speech: np.ndarray, music: np.ndarray) -> np.ndarray:
    """Little speech + music over a static-looking frame is a starting-soon/BRB screen."""
    out = probs.copy()
    steps = len(probs)
    for i in range(steps):
        lo, hi = i * MODE_STEP, min((i + 1) * MODE_STEP, len(speech))
        if hi <= lo:
            continue
        talk = float(np.mean(speech[lo:hi]))
        if talk < 0.05:
            out[i, MODES.index("static")] *= 2.0 + float(np.mean(music[lo:hi]))
        elif talk > 0.3:
            out[i, MODES.index("static")] *= 0.3
    return out / out.sum(axis=1, keepdims=True)


def viterbi(probs: np.ndarray, stay: float = STAY) -> np.ndarray:
    steps, states = probs.shape
    if steps == 0:
        return np.zeros(0, dtype=int)
    move = (1 - stay) / (states - 1)
    log_trans = np.log(np.full((states, states), move) + np.eye(states) * (stay - move))
    log_emit = np.log(probs + 1e-6)
    score = np.log(np.full(states, 1 / states)) + log_emit[0]
    back = np.zeros((steps, states), dtype=int)
    for t in range(1, steps):
        candidates = score[:, None] + log_trans
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], np.arange(states)] + log_emit[t]
    path = np.zeros(steps, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(steps - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def segments_from_path(path: np.ndarray, probs: np.ndarray, duration: float) -> list[dict]:
    segments = []
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            confidence = float(probs[start:i, path[start]].mean())
            segments.append({
                "start": start * MODE_STEP,
                "end": min(i * MODE_STEP, duration),
                "mode": MODES[path[start]],
                "confidence": round(confidence, 3),
            })
            start = i
    return segments
