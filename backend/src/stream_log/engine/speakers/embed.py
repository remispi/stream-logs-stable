"""Speaker embeddings with WeSpeaker ResNet34 (ONNX), fed Kaldi-compatible log-mel fbanks."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from stream_log.engine.audio import SAMPLE_RATE

N_MELS = 80
FRAME = 400  # 25 ms
SHIFT = 160  # 10 ms
N_FFT = 512
MIN_SECONDS = 0.8
MAX_SECONDS = 15.0


@lru_cache(maxsize=4)
def _mel_banks(n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Kaldi's triangular mel filters (low 20 Hz, high Nyquist), shape (n_mels, N_FFT//2 + 1)."""

    def mel(f: np.ndarray | float) -> np.ndarray:
        return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)

    bins = N_FFT // 2
    low, high = mel(20.0), mel(sample_rate / 2)
    delta = (high - low) / (n_mels + 1)
    centers = low + delta * np.arange(n_mels + 2)
    freqs = mel(np.arange(bins) * sample_rate / N_FFT)
    left, center, right = centers[:-2, None], centers[1:-1, None], centers[2:, None]
    up = (freqs - left) / (center - left)
    down = (right - freqs) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)


def kaldi_fbank(
    audio: np.ndarray, n_mels: int = N_MELS, window: str = "povey", scale: float = 32768.0
) -> np.ndarray:
    """Log mel filterbank like `torchaudio.compliance.kaldi.fbank` (dither 0, snip edges)."""
    samples = audio.astype(np.float64) * scale
    count = 1 + (len(samples) - FRAME) // SHIFT
    if count <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    index = np.arange(FRAME)[None, :] + SHIFT * np.arange(count)[:, None]
    frames = samples[index]
    frames -= frames.mean(axis=1, keepdims=True)
    emphasised = frames.copy()
    emphasised[:, 1:] -= 0.97 * frames[:, :-1]
    emphasised[:, 0] -= 0.97 * frames[:, 0]
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / (FRAME - 1))
    weights = hann**0.85 if window == "povey" else hann
    power = np.abs(np.fft.rfft(emphasised * weights, n=N_FFT)) ** 2
    energies = power @ _mel_banks(n_mels).T.astype(np.float64)
    return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)


class SpeakerEmbedder:
    def __init__(self, model_dir: Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        model = next(model_dir.glob("*.onnx"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2) - 2)
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Unit-length 256-d embedding, or None for clips shorter than MIN_SECONDS."""
        if len(audio) < MIN_SECONDS * SAMPLE_RATE:
            return None
        limit = int(MAX_SECONDS * SAMPLE_RATE)
        if len(audio) > limit:
            offset = (len(audio) - limit) // 2
            audio = audio[offset : offset + limit]
        feats = kaldi_fbank(audio)
        feats -= feats.mean(axis=0, keepdims=True)  # cepstral mean normalisation, as in WeSpeaker
        (emb,) = self.session.run(["embs"], {"feats": feats[None]})
        vector = emb[0].astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-9)

    def embed_long(self, audio: np.ndarray, window: float = 3.0) -> np.ndarray | None:
        """Average over fixed windows: used for enrollment audio of any length."""
        step = int(window * SAMPLE_RATE)
        parts = [self.embed(audio[i : i + step]) for i in range(0, len(audio), step)]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        mean = np.mean(parts, axis=0)
        return mean / (np.linalg.norm(mean) + 1e-9)
