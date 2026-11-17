import numpy as np

from stream_log.engine.ingest.chat import ChatMessage
from stream_log.engine.speakers.embed import N_MELS, _mel_banks, kaldi_fbank
from stream_log.engine.speakers.profiles import pick_enrollment_lines
from stream_log.engine.speakers.tagging import cluster, matches_chat, tag_lines

rng = np.random.default_rng(0)


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _voices(n_voices=2, per_voice=20, spread=0.3):
    centers = _unit(rng.normal(size=(n_voices, 256)))
    embs, truth = [], []
    for k, c in enumerate(centers):
        noise = rng.normal(size=(per_voice, 256)) * spread / np.sqrt(256)
        embs.append(_unit(c + noise))
        truth += [k] * per_voice
    return np.concatenate(embs), np.array(truth), centers


def test_fbank_shape_and_mel_banks():
    audio = rng.normal(size=16000).astype(np.float32) * 0.1
    feats = kaldi_fbank(audio)
    assert feats.shape == (98, N_MELS)  # 1 + (16000 - 400) // 160
    assert np.isfinite(feats).all()
    banks = _mel_banks()
    assert banks.shape == (N_MELS, 257)
    assert (banks.max(axis=1) > 0).all()  # no empty filter (the lowest ones are narrower than a bin)


def test_fbank_tone_peaks_in_right_band():
    t = np.arange(16000) / 16000
    low = kaldi_fbank(np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.5).mean(axis=0)
    high = kaldi_fbank(np.sin(2 * np.pi * 4000 * t).astype(np.float32) * 0.5).mean(axis=0)
    assert np.argmax(low) < np.argmax(high)


def test_cluster_separates_voices_and_orders_by_speech():
    embs, truth, _ = _voices()
    weights = np.where(truth == 1, 3.0, 1.0)  # voice 1 talks more
    labels = cluster(embs, weights)
    assert len(set(labels)) == 2
    assert (labels == (1 - truth)).all()  # label 0 = most speech


def _lines(n):
    return [{"id": i, "start": i * 5.0, "end": i * 5.0 + 4.0, "text": f"line number {i} here"}
            for i in range(n)]


def test_tag_lines_with_enrolled_voice():
    embs, _, centers = _voices(per_voice=10)
    lines = _lines(20)
    embeddings = dict(enumerate(embs))
    # The guest talks more, so without enrollment the guest would be assumed to be the streamer.
    for i in range(10):
        lines[i]["end"] = lines[i]["start"] + 1.0
    result = tag_lines(lines, embeddings, voice=centers[0], chat=[])
    tags = [e["tag"] for e in result["lines"]]
    assert tags == ["streamer"] * 10 + ["guest"] * 10
    assert "enrollment_suggestion" not in result


def test_tag_lines_without_voice_suggests_enrollment_and_overrides():
    embs, _, _ = _voices(per_voice=10)
    lines = _lines(20)
    result = tag_lines(lines, dict(enumerate(embs)), voice=None, chat=[],
                       overrides={"1": "media"})
    assert result["enrollment_suggestion"]["cluster"] == 0
    assert {c["tag"] for c in result["clusters"]} == {"streamer", "media"}


def test_short_lines_inherit_neighbour_voice():
    embs, _, centers = _voices(per_voice=5)
    lines = _lines(10)
    embeddings = {i: embs[i] for i in range(10) if i != 2}  # line 2 too short to embed
    result = tag_lines(lines, embeddings, voice=centers[0], chat=[])
    assert result["lines"][2]["tag"] == "streamer"


def test_tts_detection():
    chat = [ChatMessage(10.0, "fan", "thanks for the five dollars you are the best", "donation")]
    assert matches_chat("Thanks for the five dollars, you are the best!", 30.0, chat)
    assert matches_chat("Thanks for the five dollars, you are the best!", 300.0, chat) is None
    assert matches_chat("ok", 12.0, chat) is None

    embs, _, centers = _voices(per_voice=5)
    lines = _lines(10)
    lines[7]["text"] = "thanks for the five dollars you are the best"
    lines[7]["start"] = 20.0
    result = tag_lines(lines, dict(enumerate(embs)), voice=centers[0], chat=chat)
    assert result["lines"][7]["tag"] == "tts"


def test_pick_enrollment_lines_caps_seconds():
    embs, _, centers = _voices(n_voices=1, per_voice=20)
    lines = _lines(20)
    voice, seconds = pick_enrollment_lines(lines, dict(enumerate(embs)), list(range(20)),
                                           max_seconds=12)
    assert 12 <= seconds < 16
    assert float(voice @ centers[0]) > 0.9


def _quiet_guest(guest_starts):
    """2.2 h of streamer speech plus a loose guest voice (lines match each other ~0.4) with under
    2 % of it, so the main clustering folds the guest into the streamer."""
    streamer, _, centers = _voices(n_voices=1, per_voice=200)
    guest_center = _unit(rng.normal(size=256))
    guest = _unit(guest_center + rng.normal(size=(len(guest_starts), 256)) * 1.2 / np.sqrt(256))
    lines = [{"id": i, "start": i * 50.0, "end": i * 50.0 + 40.0, "text": f"line number {i} here"}
             for i in range(200)]
    lines += [{"id": 200 + j, "start": t, "end": t + 4.0, "text": f"guest line {j} here"}
              for j, t in enumerate(guest_starts)]
    embeddings = dict(enumerate(streamer)) | {200 + j: e for j, e in enumerate(guest)}
    return tag_lines(lines, embeddings, voice=centers[0], chat=[])


def test_quiet_guest_in_one_part_of_the_stream_gets_a_voice():
    result = _quiet_guest([1003.0 + 20 * j for j in range(30)])  # 120 s within 10 min
    tags = {e["id"]: e["tag"] for e in result["lines"]}
    assert all(tags[i] == "streamer" for i in range(200))
    assert all(tags[200 + j] == "guest" for j in range(30))
    assert len(result["clusters"]) == 2


def test_sparse_other_voice_is_not_a_guest():
    result = _quiet_guest([330.0 * j + 7 for j in range(30)])  # lyrics: 120 s over 2.7 h
    assert {e["tag"] for e in result["lines"]} == {"streamer"}


def test_repeated_lines_in_another_voice_are_media():
    embs, _, centers = _voices(per_voice=6)
    lines = _lines(12)
    # Lines 6-11 are another voice; 6 and 11 repeat a game voice line 200 s apart.
    for i, line in enumerate(lines):
        line["start"], line["end"] = i * 40.0, i * 40.0 + 3.0
    lines[6]["text"] = lines[11]["text"] = "I seek only the strongest"
    lines[2]["text"] = lines[4]["text"] = "lets go chat lets go"  # streamer catchphrase
    result = tag_lines(lines, dict(enumerate(embs)), voice=centers[0], chat=[])
    tags = {e["id"]: e["tag"] for e in result["lines"]}
    assert tags[6] == tags[11] == "media"
    assert tags[2] == tags[4] == "streamer"
    assert tags[7] == "guest"  # other voice but not repeated
