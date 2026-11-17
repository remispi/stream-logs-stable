"""Editing/preview assets made from the source video: probe info, proxy, sprite, waveform peaks."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from stream_log.engine.audio import ffmpeg_exe

PROXY_HEIGHT = 540
PROXY_MAX_FPS = 30
SPRITE_TILE = (160, 90)
SPRITE_COLUMNS = 10
SPRITE_MAX_TILES = 2000
PEAKS_PER_SECOND = 20


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_tracks: int

    def to_dict(self) -> dict:
        return asdict(self)


def _fraction(text: str | None) -> float | None:
    if not text or text in ("0/0", "N/A"):
        return None
    num, _, den = text.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: Path) -> MediaInfo:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe not found on PATH")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next(
        (s for s in streams
         if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    audio_tracks = sum(s.get("codec_type") == "audio" for s in streams)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    return MediaInfo(
        duration=duration,
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        fps=_fraction(video.get("avg_frame_rate")) if video else None,
        video_codec=video.get("codec_name") if video else None,
        audio_tracks=audio_tracks,
    )


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def make_proxy(source: Path, dest: Path, info: MediaInfo, *, audio_track: int = 0) -> str:
    """540p H.264 proxy with a keyframe every second, for scrubbing and preview.

    Uses NVENC when it works and falls back to libx264. Returns the encoder used.
    """
    fps = min(info.fps or PROXY_MAX_FPS, PROXY_MAX_FPS)
    gop = max(1, round(fps))
    tmp = dest.with_suffix(".part.mp4")
    common_in = ["-i", str(source), "-map", "0:v:0", "-map", f"0:a:{audio_track}?"]
    video_filter = ["-vf", f"scale=-2:{PROXY_HEIGHT},fps={fps:g}", "-pix_fmt", "yuv420p"]
    audio = ["-c:a", "aac", "-b:a", "96k", "-ac", "2"]
    tail = ["-movflags", "+faststart", str(tmp)]
    encoders = {
        "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "28",
                       "-b:v", "0", "-g", str(gop)],
        "libx264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", str(gop)],
    }
    errors = []
    for name, codec in encoders.items():
        result = _run_ffmpeg([*common_in, *video_filter, *codec, *audio, *tail])
        if result.returncode == 0:
            tmp.replace(dest)
            return name
        errors.append(f"{name}: {result.stderr.strip()[-300:]}")
    tmp.unlink(missing_ok=True)
    raise RuntimeError("proxy encode failed: " + " | ".join(errors))


def make_sprite(proxy: Path, dest: Path, duration: float) -> dict:
    """Thumbnail sheet for the timeline. Reads keyframes only, so it is quick on the proxy."""
    interval = max(10.0, math.ceil(duration / SPRITE_MAX_TILES))
    count = max(1, math.ceil(duration / interval))
    rows = math.ceil(count / SPRITE_COLUMNS)
    width, height = SPRITE_TILE
    result = _run_ffmpeg(
        ["-skip_frame", "nokey", "-i", str(proxy),
         "-vf", f"fps=1/{interval:g},scale={width}:{height},tile={SPRITE_COLUMNS}x{rows}",
         "-frames:v", "1", "-q:v", "5", str(dest)]
    )
    if result.returncode != 0:
        raise RuntimeError(f"sprite failed: {result.stderr.strip()[-300:]}")
    return {"interval": interval, "count": count, "columns": SPRITE_COLUMNS,
            "tile_width": width, "tile_height": height}


def waveform_peaks(wav_path: Path, per_second: int = PEAKS_PER_SECOND) -> dict:
    """Max-abs amplitude per bucket, scaled 0-100, read in one-minute blocks."""
    peaks: list[int] = []
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        bucket = rate // per_second
        block = bucket * per_second * 60
        while True:
            raw = wav.readframes(block)
            if not raw:
                break
            samples = np.abs(np.frombuffer(raw, dtype=np.int16).astype(np.int32))
            usable = len(samples) // bucket * bucket
            if usable:
                values = samples[:usable].reshape(-1, bucket).max(axis=1)
                peaks.extend(np.round(values * 100 / 32768).astype(int).tolist())
            if usable < len(samples):
                peaks.append(round(int(samples[usable:].max()) * 100 / 32768))
    return {"per_second": per_second, "peaks": peaks}
