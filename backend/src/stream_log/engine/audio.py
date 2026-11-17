"""Audio decoding through the FFmpeg executable.

We decode with FFmpeg rather than faster-whisper's PyAV decoder: FFmpeg is already required for
rendering, and PyAV releases have broken faster-whisper's decoder before.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a media file to mono float32 PCM in [-1, 1] at `sample_rate`.

    Loads the whole file into memory: fine for short files. Long VODs go through `extract_wav` + `read_wav`.
    """
    result = subprocess.run(
        [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def ffmpeg_exe() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    return ffmpeg


def extract_wav(
    source: Path, dest: Path, *, sample_rate: int = SAMPLE_RATE, audio_track: int = 0
) -> Path:
    """Write the chosen audio track of `source` as 16-bit mono WAV (the analysis copy)."""
    tmp = dest.with_suffix(".part.wav")
    subprocess.run(
        [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-map", f"0:a:{audio_track}", "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(tmp)],
        capture_output=True,
        check=True,
    )
    tmp.replace(dest)
    return dest


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def read_wav(path: Path, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """Read [start, end) seconds of a 16-bit mono WAV as float32 without loading the rest."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono WAV")
        rate = wav.getframerate()
        total = wav.getnframes()
        first = min(max(int(start * rate), 0), total)
        last = total if end is None else min(max(int(end * rate), first), total)
        wav.setpos(first)
        raw = wav.readframes(last - first)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
