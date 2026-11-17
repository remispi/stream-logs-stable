"""WhisperX-style forced alignment: wav2vec2 CTC (ONNX) + Viterbi over the known transcript.

Whisper's word timestamps (and especially CrisperWhisper's under CTranslate2) can be off by a few
hundred ms, which is too coarse for cutting fillers. Each line is re-aligned character by
character; words with nothing alignable (numbers, symbols) are placed in the gap between their
aligned neighbours.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from stream_log.engine.asr.fillers import PROBABLE_FILLER
from stream_log.engine.audio import SAMPLE_RATE, read_wav

LINE_PAD = 0.4
# Mean log-probability per aligned character below this means the audio doesn't match the text
# (music, crosstalk, ASR hallucination): keep the original Whisper times.
MIN_SCORE = -4.0
BLANK = 0
WORD_SEP = "|"

_KEEP = re.compile(r"[^A-Z']")


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def forced_align(emission: np.ndarray, tokens: list[int]) -> list[tuple[int, float]] | None:
    """Viterbi path through a CTC emission (T x V log-probs) for `tokens`.

    Returns (frame, log_prob) for each token, or None when the audio is too short.
    """
    frames, n = emission.shape[0], len(tokens)
    if n == 0 or frames < n:
        return None
    token_ids = np.asarray(tokens)
    trellis = np.full((frames + 1, n + 1), -np.inf, dtype=np.float32)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = np.cumsum(emission[:, BLANK])
    for t in range(frames):
        stay = trellis[t, 1:] + emission[t, BLANK]
        advance = trellis[t, :-1] + emission[t, token_ids]
        trellis[t + 1, 1:] = np.maximum(stay, advance)

    path: list[tuple[int, float]] = []
    j = n
    t = frames
    while j > 0 and t > 0:
        stay = trellis[t - 1, j] + emission[t - 1, BLANK]
        advance = trellis[t - 1, j - 1] + emission[t - 1, token_ids[j - 1]]
        if advance >= stay:
            path.append((t - 1, float(emission[t - 1, token_ids[j - 1]])))
            j -= 1
        t -= 1
    if j > 0:
        return None
    return path[::-1]


class CtcAligner:
    def __init__(self, model_dir: Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        self.vocab: dict[str, int] = vocab
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2) - 2)
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def emission(self, audio: np.ndarray) -> np.ndarray:
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        (logits,) = self.session.run(["logits"], {"input_values": audio[None, :].astype(np.float32)})
        return _log_softmax(logits[0])

    def word_tokens(self, word: str) -> list[int]:
        if word == PROBABLE_FILLER:
            return []
        return [self.vocab[c] for c in _KEEP.sub("", word.upper()) if c in self.vocab]

    def align_line(self, wav_path: Path, line: dict, duration: float) -> bool:
        """Update the line's word times in place. Returns False if alignment was rejected."""
        start = max(0.0, line["start"] - LINE_PAD)
        end = min(duration, line["end"] + LINE_PAD)
        audio = read_wav(wav_path, start, end)
        if len(audio) < SAMPLE_RATE // 10:
            return False
        emission = self.emission(audio)
        frame_seconds = (len(audio) / SAMPLE_RATE) / emission.shape[0]

        words = line["words"]
        per_word = [self.word_tokens(w["w"]) for w in words]
        tokens: list[int] = []
        spans: list[tuple[int, int] | None] = []
        separator = self.vocab[WORD_SEP]
        for word_tokens in per_word:
            if not word_tokens:
                spans.append(None)
                continue
            if tokens:
                tokens.append(separator)
            spans.append((len(tokens), len(tokens) + len(word_tokens)))
            tokens.extend(word_tokens)
        path = forced_align(emission, tokens)
        if path is None:
            return False
        score = float(np.mean([p for _, p in path]))
        if score < MIN_SCORE:
            return False

        for word, span in zip(words, spans, strict=True):
            if span is None:
                continue
            first, last = path[span[0]][0], path[span[1] - 1][0]
            word["start"] = round(start + first * frame_seconds, 3)
            word["end"] = round(start + (last + 1) * frame_seconds, 3)
        _fill_unaligned(words, spans)
        line["start"], line["end"] = words[0]["start"], max(w["end"] for w in words)
        line["align_score"] = round(score, 2)
        return True


def _fill_unaligned(words: list[dict], spans: list) -> None:
    """Place words without alignable characters between their aligned neighbours."""
    for index, span in enumerate(spans):
        if span is not None or words[index]["w"] == PROBABLE_FILLER:
            continue
        before = next((words[i]["end"] for i in range(index - 1, -1, -1) if spans[i]), None)
        after = next((words[i]["start"] for i in range(index + 1, len(words)) if spans[i]), None)
        start = before if before is not None else words[index]["start"]
        end = after if after is not None else words[index]["end"]
        if end > start:
            words[index]["start"], words[index]["end"] = start, end


def align_transcript(
    wav_path: Path,
    lines: list[dict],
    model_dir: Path,
    duration: float,
    progress: Callable[[float, str], None] | None = None,
) -> int:
    """Align every line in place; returns how many lines kept their Whisper times."""
    aligner = CtcAligner(model_dir)
    rejected = 0
    for index, line in enumerate(lines):
        if not aligner.align_line(wav_path, line, duration):
            rejected += 1
            line["align_score"] = None
        if progress and index % 20 == 0:
            progress((index + 1) / len(lines), f"line {index + 1} / {len(lines)}")
    return rejected
