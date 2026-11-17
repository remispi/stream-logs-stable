"""Helpers for 1 Hz signal arrays."""

from __future__ import annotations

import numpy as np

BASELINE_SECONDS = 600  # robust z-scores are against the surrounding 10 minutes
_BASELINE_STEP = 10


def smooth(x: np.ndarray, seconds: int) -> np.ndarray:
    if seconds <= 1 or len(x) == 0:
        return x.astype(np.float32)
    kernel = np.ones(seconds) / seconds
    padded = np.pad(x.astype(np.float64), (seconds // 2, seconds - 1 - seconds // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def robust_z(x: np.ndarray, window: int = BASELINE_SECONDS, clip: float = 8.0) -> np.ndarray:
    """(x - rolling median) / (1.4826 * rolling MAD), baseline computed every 10 s.

    The rolling baseline adapts to each part of the stream (a loud game vs. quiet chatting), so a
    peak means "unusual for this part of the stream". NaNs (no data) stay NaN.
    """
    n = len(x)
    if n == 0:
        return x.astype(np.float32)
    step = _BASELINE_STEP
    centres = np.arange(0, n, step)
    half = window // 2
    medians = np.empty(len(centres))
    mads = np.empty(len(centres))
    for i, c in enumerate(centres):
        chunk = x[max(0, c - half) : c + half]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) == 0:
            medians[i], mads[i] = 0.0, 1.0
            continue
        medians[i] = np.median(chunk)
        mads[i] = np.median(np.abs(chunk - medians[i]))
    positions = np.arange(n)
    median = np.interp(positions, centres, medians)
    mad = np.interp(positions, centres, mads)
    # A flat baseline (e.g. silent chat) would divide by ~0; floor the spread.
    spread = np.maximum(1.4826 * mad, np.nanstd(x) * 0.1 + 1e-6)
    return np.clip((x - median) / spread, -clip, clip).astype(np.float32)


def per_second(values: np.ndarray, times: np.ndarray, duration: int, reducer=np.mean) -> np.ndarray:
    """Bucket (time, value) samples into whole seconds; empty seconds are NaN."""
    out = np.full(duration, np.nan, dtype=np.float32)
    if len(times) == 0:
        return out
    seconds = np.clip(times.astype(int), 0, duration - 1)
    order = np.argsort(seconds, kind="stable")
    seconds, values = seconds[order], values[order]
    starts = np.flatnonzero(np.r_[True, np.diff(seconds) > 0])
    for begin, end in zip(starts, np.r_[starts[1:], len(seconds)], strict=True):
        out[seconds[begin]] = reducer(values[begin:end])
    return out


def counts_per_second(times: np.ndarray, duration: int) -> np.ndarray:
    seconds = np.clip(np.asarray(times, dtype=float).astype(int), 0, max(duration - 1, 0))
    return np.bincount(seconds, minlength=duration)[:duration].astype(np.float32)


def nan0(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0)


def peaks(x: np.ndarray, *, min_gap: int = 30, threshold: float = 0.0, limit: int = 50
          ) -> list[int]:
    """Indices of the highest values at least `min_gap` apart (greedy)."""
    order = np.argsort(-nan0(x))
    chosen: list[int] = []
    for index in order:
        if x[index] < threshold or len(chosen) >= limit:
            break
        if all(abs(index - c) >= min_gap for c in chosen):
            chosen.append(int(index))
    return sorted(chosen)
