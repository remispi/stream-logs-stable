"""Audio signals at 1 Hz: loudness, pitch movement, speech rate, clipping and sound events.

Voice arousal here is prosodic (loudness + pitch variation). emotion2vec+ is not
wired in yet; the sound-event model covers laughter / screams / cheering.
"""

from __future__ import annotations

import json
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from stream_log.engine.audio import SAMPLE_RATE, read_wav

FRAME = 640  # 40 ms pitch frames
HOP = 320  # 20 ms
MIN_F0, MAX_F0 = 70.0, 500.0
VOICED_CORRELATION = 0.5
CLIP_LEVEL = 0.99

AST_SECONDS = 10.24
AST_FRAMES = 1024
EVENT_HOP = 10.0

# AudioSet classes folded into the event signals we use (max over the members).
EVENT_GROUPS = {
    "laughter": ["Laughter", "Baby laughter", "Giggle", "Belly laugh", "Chuckle, chortle",
                 "Snicker"],
    "scream": ["Screaming", "Shout", "Yell", "Whoop", "Children shouting"],
    "cheer": ["Cheering", "Applause", "Crowd"],
    "music": ["Music", "Video game music", "Musical instrument", "Singing"],
    "speech": ["Speech", "Conversation", "Narration, monologue"],
    "crying": ["Crying, sobbing"],
    "gasp": ["Gasp"],
    "gunfire": ["Gunshot, gunfire", "Explosion", "Machine gun"],
    "tts_voice": ["Speech synthesizer"],
}


def _pitch_frames(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f0 (Hz, NaN when unvoiced) and RMS for 40 ms frames, via normalised autocorrelation."""
    count = 1 + (len(audio) - FRAME) // HOP
    if count <= 0:
        return np.zeros(0), np.zeros(0)
    index = np.arange(FRAME)[None, :] + HOP * np.arange(count)[:, None]
    frames = audio[index].astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    rms = np.sqrt((frames**2).mean(axis=1))
    spectrum = np.fft.rfft(frames * np.hanning(FRAME), n=2 * FRAME)
    acf = np.fft.irfft(np.abs(spectrum) ** 2)[:, :FRAME]
    acf /= acf[:, :1] + 1e-12
    lo, hi = int(SAMPLE_RATE / MAX_F0), int(SAMPLE_RATE / MIN_F0)
    lag = lo + np.argmax(acf[:, lo:hi], axis=1)
    strength = acf[np.arange(count), lag]
    f0 = np.where((strength > VOICED_CORRELATION) & (rms > 0.01), SAMPLE_RATE / lag, np.nan)
    return f0, rms


def prosody(wav_path: Path, duration: int) -> dict[str, np.ndarray]:
    """Per-second loudness (dBFS), pitch (semitones vs 100 Hz) mean/std, clipped-sample ratio."""
    loudness = np.full(duration, -90.0, dtype=np.float32)
    pitch = np.full(duration, np.nan, dtype=np.float32)
    pitch_std = np.full(duration, np.nan, dtype=np.float32)
    clipping = np.zeros(duration, dtype=np.float32)
    block = 60
    for start in range(0, duration, block):
        audio = read_wav(wav_path, start, min(start + block, duration) + FRAME / SAMPLE_RATE)
        f0, _ = _pitch_frames(audio)
        frames_per_second = SAMPLE_RATE // HOP
        for offset in range(min(block, duration - start)):
            second = audio[offset * SAMPLE_RATE : (offset + 1) * SAMPLE_RATE]
            if len(second) == 0:
                break
            t = start + offset
            loudness[t] = 20 * np.log10(np.sqrt(np.mean(second**2)) + 1e-9)
            clipping[t] = np.mean(np.abs(second) >= CLIP_LEVEL)
            voiced = f0[offset * frames_per_second : (offset + 1) * frames_per_second]
            voiced = voiced[~np.isnan(voiced)]
            if len(voiced) >= 5:
                semitones = 12 * np.log2(voiced / 100.0)
                pitch[t] = np.median(semitones)
                pitch_std[t] = np.std(semitones)
    return {"loudness": loudness, "pitch": pitch, "pitch_std": pitch_std, "clipping": clipping}


def speech_signals(lines: list[dict], tags: dict[int, str], duration: int) -> dict[str, np.ndarray]:
    """Words per second and speech coverage, for the streamer and for everyone."""
    rate = np.zeros(duration, dtype=np.float32)
    streamer_cover = np.zeros(duration, dtype=np.float32)
    any_cover = np.zeros(duration, dtype=np.float32)
    for line in lines:
        is_streamer = tags.get(line["id"], "streamer") == "streamer"
        for word in line["words"]:
            if word.get("probable"):
                continue
            t = min(int(word["start"]), duration - 1)
            if is_streamer:
                rate[t] += 1
            start, end = word["start"], max(word["end"], word["start"] + 0.05)
            for s in range(int(start), min(int(end) + 1, duration)):
                overlap = min(end, s + 1) - max(start, s)
                if overlap > 0:
                    any_cover[s] += overlap
                    if is_streamer:
                        streamer_cover[s] += overlap
    return {"speech_rate": rate, "speech": np.minimum(any_cover, 1.0),
            "streamer_speech": np.minimum(streamer_cover, 1.0)}


class SoundEvents:
    """Audio Spectrogram Transformer (AudioSet, 527 classes) on 10 s windows."""

    def __init__(self, model_dir: Path) -> None:
        from stream_log.engine.onnx_models import session

        self.session = session(model_dir / "onnx" / "model.onnx")
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        prep = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
        self.mean, self.std = prep["mean"], prep["std"]
        label_ids = {v: int(k) for k, v in config["id2label"].items()}
        self.groups = {
            name: [label_ids[c] for c in classes if c in label_ids]
            for name, classes in EVENT_GROUPS.items()
        }

    def features(self, audio: np.ndarray) -> np.ndarray:
        from stream_log.engine.speakers.embed import kaldi_fbank

        fbank = kaldi_fbank(audio, n_mels=128, window="hanning", scale=1.0)
        if len(fbank) < AST_FRAMES:
            fbank = np.pad(fbank, ((0, AST_FRAMES - len(fbank)), (0, 0)))
        return ((fbank[:AST_FRAMES] - self.mean) / (self.std * 2)).astype(np.float32)

    def run(self, wav_path: Path, duration: int, progress: Callable[[float, str], None] | None
            ) -> dict[str, np.ndarray]:
        out = {name: np.zeros(duration, dtype=np.float32) for name in self.groups}
        starts = np.arange(0, max(duration, 1), EVENT_HOP)
        batch: list[np.ndarray] = []
        batch_starts: list[float] = []

        def flush() -> None:
            logits = self.session.run(None, {"input_values": np.stack(batch)})[0]
            probs = 1 / (1 + np.exp(-logits))
            for start, row in zip(batch_starts, probs, strict=True):
                lo, hi = int(start), min(int(start + EVENT_HOP), duration)
                for name, ids in self.groups.items():
                    out[name][lo:hi] = row[ids].max() if ids else 0.0
            batch.clear()
            batch_starts.clear()

        for i, start in enumerate(starts):
            audio = read_wav(wav_path, start, start + AST_SECONDS)
            batch.append(self.features(audio))
            batch_starts.append(float(start))
            if len(batch) == 8:
                flush()
                if progress:
                    progress((i + 1) / len(starts), f"{int(start) // 60} / {duration // 60} min")
        if batch:
            flush()
        return out


def wav_seconds(wav_path: Path) -> int:
    with wave.open(str(wav_path), "rb") as wav:
        return int(np.ceil(wav.getnframes() / wav.getframerate()))
