"""Verbatim transcription with faster-whisper, in windows so multi-hour VODs fit in memory.

Output lines (not Whisper's 30 s segments) are the unit everything downstream uses: a line ends
at a pause, at the end of a sentence or after `MAX_LINE_SECONDS`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from stream_log.engine.asr.fillers import normalize_filler
from stream_log.engine.audio import SAMPLE_RATE, read_wav, wav_duration
from stream_log.engine.gpu import register_cuda_dlls

WINDOW_SECONDS = 20 * 60
WINDOW_SEARCH = 20.0
LINE_PAUSE = 0.6
MAX_LINE_SECONDS = 12.0

# Whisper copies the style of the prompt, so a filler-heavy prompt keeps "um"/"uh" that it would
# otherwise clean up. Used for non-verbatim models.
FILLER_PROMPT = "Umm, so, uh, let me think like, hmm... Okay, uh, here's what I'm, um, thinking."

Progress = Callable[[float, str], None]


def pick_device() -> tuple[str, str]:
    register_cuda_dlls()
    import ctranslate2

    if ctranslate2.get_cuda_device_count() > 0:
        return "cuda", "float16"
    return "cpu", "int8"


def window_bounds(wav_path: Path, duration: float) -> list[tuple[float, float]]:
    """Split into ~20 min windows, cutting at the quietest half second near each boundary."""
    bounds, start = [], 0.0
    while duration - start > WINDOW_SECONDS + WINDOW_SEARCH:
        target = start + WINDOW_SECONDS
        audio = read_wav(wav_path, target - WINDOW_SEARCH, target + WINDOW_SEARCH)
        hop = SAMPLE_RATE // 2
        frames = len(audio) // hop
        energy = (audio[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1)
        cut = target - WINDOW_SEARCH + (int(np.argmin(energy)) + 0.5) * 0.5
        bounds.append((start, cut))
        start = cut
    bounds.append((start, duration))
    return bounds


def speech_regions(audio: np.ndarray, offset: float) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_speech_duration_ms=150, min_silence_duration_ms=150, speech_pad_ms=30)
    return [
        (offset + r["start"] / SAMPLE_RATE, offset + r["end"] / SAMPLE_RATE)
        for r in get_speech_timestamps(audio, options)
    ]


def _word(raw, offset: float) -> dict:
    text = raw.word.strip()
    filler = normalize_filler(text)
    return {
        "w": filler or text,
        "start": round(offset + raw.start, 3),
        "end": round(offset + raw.end, 3),
        "p": round(raw.probability, 3),
        "filler": filler is not None,
    }


def build_lines(words: list[dict]) -> list[dict]:
    lines: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            lines.append({
                "id": len(lines),
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(w["w"] for w in current),
                "words": list(current),
            })
            current.clear()

    for word in words:
        if current:
            previous = current[-1]
            pause = word["start"] - previous["end"]
            sentence_end = previous["w"][-1:] in ".?!" and pause > 0.15
            too_long = word["end"] - current[0]["start"] > MAX_LINE_SECONDS
            if pause >= LINE_PAUSE or sentence_end or too_long:
                flush()
        current.append(word)
    flush()
    return lines


def transcribe(
    wav_path: Path,
    model_dir: Path,
    *,
    verbatim_model: bool,
    device: str = "auto",
    batch_size: int = 12,
    progress: Progress | None = None,
) -> dict:
    """Transcribe the analysis WAV. Returns `{"lines", "speech", ...}` (see transcript.json)."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    compute_type = "int8"
    if device == "auto":
        device, compute_type = pick_device()
    elif device == "cuda":
        register_cuda_dlls()
        compute_type = "float16"
    model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
    pipeline = BatchedInferencePipeline(model)

    duration = wav_duration(wav_path)
    words: list[dict] = []
    speech: list[tuple[float, float]] = []
    for start, end in window_bounds(wav_path, duration):
        audio = read_wav(wav_path, start, end)
        speech.extend(speech_regions(audio, start))
        segments, _ = pipeline.transcribe(
            audio,
            language="en",
            batch_size=batch_size if device == "cuda" else 1,
            word_timestamps=True,
            initial_prompt=None if verbatim_model else FILLER_PROMPT,
            # CrisperWhisper writes fillers as [UM]/[UH]; the default suppression list drops "[".
            suppress_tokens=[] if verbatim_model else [-1],
        )
        for segment in segments:
            words.extend(_word(w, start) for w in segment.words or [])
            if progress:
                done = start + segment.end
                progress(done / duration, f"{_clock(done)} / {_clock(duration)}")
    words.sort(key=lambda w: w["start"])
    return {
        "language": "en",
        "duration": round(duration, 3),
        "device": f"{device}/{compute_type}",
        "lines": build_lines(words),
        "speech": [[round(a, 3), round(b, 3)] for a, b in speech],
    }


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
