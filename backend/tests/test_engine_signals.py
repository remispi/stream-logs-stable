import numpy as np

from stream_log.engine.chapters import choose_boundaries, depth_scores, youtube_chapters
from stream_log.engine.ingest.chat import ChatMessage
from stream_log.engine.signals.chat import (
    chat_reliability,
    describe_emotes,
    estimate_delay,
    find_interactions,
    kl_shift,
    shift_back,
)
from stream_log.engine.signals.common import counts_per_second, peaks, robust_z, smooth
from stream_log.engine.signals.fusion import fuse, weights
from stream_log.engine.signals.visual import MODES, frozen_spans, segments_from_path, viterbi

rng = np.random.default_rng(1)


def test_robust_z_flags_spike_against_local_baseline():
    x = np.concatenate([rng.normal(0, 1, 1200), rng.normal(20, 1, 1200)])  # loud second half
    x[600] = 10
    z = robust_z(x)
    assert z[600] > 5  # unusual for the quiet half
    assert abs(np.median(z[1500:2000])) < 1  # loud half is normal for itself


def test_smooth_counts_peaks():
    counts = counts_per_second(np.array([0.2, 0.7, 5.1]), 8)
    assert counts.tolist() == [2, 0, 0, 0, 0, 1, 0, 0]
    assert smooth(np.ones(10), 5).tolist() == [1.0] * 10
    signal = np.zeros(200)
    signal[[20, 25, 120]] = [5, 4, 3]
    assert peaks(signal, min_gap=30, threshold=1) == [20, 120]


def test_describe_emotes():
    text, count = describe_emotes("KEKW that was OMEGALUL")
    assert count == 2 and "laughing" in text


def test_delay_estimation_finds_lag():
    arousal = rng.normal(0, 1, 2000)
    chat = np.r_[np.zeros(7), arousal[:-7]] + rng.normal(0, 0.3, 2000)
    delay, corr = estimate_delay(chat, arousal)
    assert delay == 7 and corr > 0.5
    shifted = shift_back(np.arange(10.0), 3)
    assert shifted[:3].tolist() == [3, 4, 5]


def test_chat_reliability_scales_with_chatters():
    assert chat_reliability(np.zeros(100)) == 0.0
    assert chat_reliability(np.full(100, 30.0)) == 1.0
    assert 0 < chat_reliability(np.full(100, 3.0)) < 1


def test_tiny_chat_has_no_weight_in_fusion():
    w = weights(0.0)
    assert all(v == 0 for k, v in w.items() if k.startswith("chat_"))
    assert abs(sum(w.values()) - 1) < 1e-6
    n = 1200
    signals = {k: rng.normal(0, 1, n).astype(np.float32) for k in w}
    for k in ("laughter", "scream", "cheer"):
        signals[k] = np.zeros(n, dtype=np.float32)
    signals["laughter"][500:505] = 0.9
    fused, _ = fuse(signals, 0.0)
    assert int(np.argmax(fused)) in range(495, 510)


def test_kl_shift_detects_vibe_change():
    vibes = np.tile(np.array([0.9, 0.1]), (1200, 1)).astype(np.float32)
    vibes[800:830] = [0.1, 0.9]
    shift = kl_shift(vibes)
    assert shift[815] > 5 * shift[300]


def test_viterbi_smooths_single_frame_flips():
    probs = np.full((30, len(MODES)), 0.05)
    probs[:, 0] = 0.8
    probs[10, 0], probs[10, 1] = 0.3, 0.5  # one odd frame
    probs[20:, 0], probs[20:, 4] = 0.05, 0.8  # real switch to static
    path = viterbi(probs / probs.sum(axis=1, keepdims=True))
    assert path[10] == 0
    segments = segments_from_path(path, probs, 300)
    assert [s["mode"] for s in segments] == ["gaming", "static"]
    assert segments[1]["start"] == 200 and segments[1]["end"] == 300


def test_frozen_spans():
    motion = np.full(100, 0.05)
    motion[30:50] = 0.0
    assert frozen_spans(motion) == [(30, 50)]


def test_find_interactions():
    lines = [
        {"id": 0, "start": 30.0, "text": "thanks bigfan123 for being here"},
        {"id": 1, "start": 50.0, "text": "someone asks what is my favourite game honestly"},
        {"id": 2, "start": 70.0, "text": "my favourite game is probably elden ring"},
        {"id": 3, "start": 90.0, "text": "unrelated guest line about pizza toppings"},
    ]
    chat = [
        ChatMessage(20.0, "bigfan123", "hello!"),
        ChatMessage(45.0, "curious", "what is your favourite game?"),
        ChatMessage(60.0, "other", "what is your favourite game ever streamer?"),
    ]
    found = find_interactions(lines, {3: "guest"}, chat)
    kinds = {f["line_id"]: f["kind"] for f in found}
    assert kinds[0] == "says_username"
    assert kinds[1] == "reads_chat"
    assert kinds[2] == "answers_question"
    assert 3 not in kinds


def test_chapter_boundaries_follow_topic_shift():
    a, b = rng.normal(size=16), rng.normal(size=16)
    embs = np.array([a + rng.normal(0, 0.1, 16) for _ in range(10)]
                    + [b + rng.normal(0, 0.1, 16) for _ in range(10)])
    depth = depth_scores(embs)
    assert int(np.argmax(depth)) == 10
    bounds = choose_boundaries(depth, forced=[], duration=1200)
    assert bounds == [0.0, 600.0]


def test_youtube_chapters_format():
    text = youtube_chapters([{"start": 0, "end": 600, "title": "Intro"},
                             {"start": 600, "end": 4000, "title": "Boss"},
                             {"start": 4000, "end": 4005, "title": "Too short"}])
    assert text.splitlines() == ["0:00 Intro", "10:00 Boss"]


def test_kl_shift_is_zero_without_chat():
    vibes = np.zeros((1200, 2), dtype=np.float32)
    vibes[600:, 0] = 1.0  # chat starts at 10 min
    shift = kl_shift(vibes)
    assert shift[:560].max() == 0
