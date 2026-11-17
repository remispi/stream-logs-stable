import shutil
import subprocess

import numpy as np
import pytest

from stream_log.engine.audio import SAMPLE_RATE, load_audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def test_load_audio_resamples_to_16k_mono(tmp_path):
    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:sample_rate=44100:duration=1", "-ac", "2", str(wav)],
        check=True,
    )
    audio = load_audio(wav)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - SAMPLE_RATE) <= SAMPLE_RATE * 0.01
    assert 0.05 < np.abs(audio).max() <= 1.0


def test_load_audio_missing_file_raises(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        load_audio(tmp_path / "missing.wav")
