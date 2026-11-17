import numpy as np

from stream_log.engine.asr.align import _fill_unaligned, forced_align
from stream_log.engine.asr.fillers import (
    PROBABLE_FILLER,
    insert_gap_fillers,
    normalize_filler,
    uncovered_speech,
)
from stream_log.engine.asr.transcribe import build_lines


def _w(text, start, end, filler=False):
    return {"w": text, "start": start, "end": end, "p": 1.0, "filler": filler}


def test_normalize_filler():
    assert normalize_filler("[UM]") == "um"
    assert normalize_filler(" Uh,") == "uh"
    assert normalize_filler("hmm...") == "hmm"
    assert normalize_filler("umbrella") is None
    assert normalize_filler("like") is None


def test_build_lines_splits_on_pause_and_sentence_end():
    words = [_w("Hi", 0.0, 0.3), _w("chat.", 0.35, 0.6), _w("So", 0.9, 1.0), _w("yeah", 1.05, 1.3),
             _w("later", 2.5, 2.8)]
    lines = build_lines(words)
    assert [line["text"] for line in lines] == ["Hi chat.", "So yeah", "later"]
    assert [line["id"] for line in lines] == [0, 1, 2]


def test_build_lines_caps_length():
    words = [_w(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(40)]
    lines = build_lines(words)
    assert all(line["end"] - line["start"] <= 12.0 for line in lines)
    assert sum(len(line["words"]) for line in lines) == 40


def test_uncovered_speech():
    speech = [(0.0, 2.0), (3.0, 4.0)]
    words = [(0.1, 0.5), (0.4, 0.9), (1.5, 2.0)]
    gaps = uncovered_speech(speech, words)
    assert [(round(a, 2), round(b, 2)) for a, b in gaps] == [(0.0, 0.06), (0.94, 1.46), (3.0, 4.0)]


def test_gap_fillers_inserted_and_merged():
    lines = [{"id": 0, "start": 0.0, "end": 3.0, "text": "", "words": [
        _w("so", 0.0, 0.3), _w("uh", 0.5, 0.55, filler=True), _w("yeah", 0.9, 1.2),
        _w("well", 2.4, 3.0),
    ]}]
    # 0.6-0.85: right after "uh" -> extends it; 1.5-2.2: a new probable filler; 5-8: unrecognised.
    unrecognised = insert_gap_fillers(lines, [(0.0, 3.0), (5.0, 8.0)])
    words = lines[0]["words"]
    uh = next(w for w in words if w["w"] == "uh")
    assert uh["end"] >= 0.8
    probable = [w for w in words if w["w"] == PROBABLE_FILLER]
    assert len(probable) == 1 and probable[0]["start"] >= 1.2
    assert unrecognised == [{"start": 5.0, "end": 8.0}]


def test_forced_align_follows_emission():
    # Vocab: 0 blank, 1 'A', 2 'B'. Audio says A at frames 2-3 and B at frame 7.
    frames, vocab = 10, 3
    emission = np.full((frames, vocab), -10.0, dtype=np.float32)
    emission[:, 0] = -0.1
    emission[2:4, 1] = -0.01
    emission[2:4, 0] = -5
    emission[7, 2] = -0.01
    emission[7, 0] = -5
    path = forced_align(emission, [1, 2])
    assert path is not None
    assert [frame for frame, _ in path] in ([2, 7], [3, 7])


def test_forced_align_too_short():
    assert forced_align(np.zeros((1, 3), dtype=np.float32), [1, 2]) is None


def test_fill_unaligned_uses_neighbours():
    words = [_w("I", 0.0, 0.2), _w("4", 0.0, 0.0), _w("times", 0.5, 0.8)]
    _fill_unaligned(words, [(0, 1), None, (2, 3)])
    assert (words[1]["start"], words[1]["end"]) == (0.2, 0.5)


def test_gap_fillers_respect_whisper_spans():
    # Aligned spans are short; Whisper's original spans cover the gap between them.
    lines = [{"id": 0, "start": 0.0, "end": 2.0, "text": "", "words": [
        _w("made", 0.0, 0.2), _w("switch", 1.5, 2.0)]}]
    insert_gap_fillers(lines, [(0.0, 2.0)], extra_spans=[(0.0, 0.8), (0.8, 2.0)])
    assert [w["w"] for w in lines[0]["words"]] == ["made", "switch"]
