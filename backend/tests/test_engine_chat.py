import json
from datetime import datetime

import pytest

from stream_log.engine.ingest.chat import (
    ChatMessage,
    classify_message,
    detect_format,
    load_chat,
    read_saved_chat,
    save_chat,
)


@pytest.mark.parametrize(
    ("user", "text", "bits", "expected"),
    [
        ("viewer", "lol that was insane", 0, "message"),
        ("viewer", "Cheer100 lets go", 0, "bits"),
        ("viewer", "nice", 50, "bits"),
        ("subguy", "subguy subscribed at Tier 1. They've subscribed for 3 months!", 0, "sub"),
        ("subguy", "subguy subscribed with Prime.", 0, "sub"),
        ("giver", "giver is gifting 5 Tier 1 Subs to the community!", 0, "gift"),
        ("StreamElements", "bigfan just tipped $5.00!", 0, "donation"),
        ("StreamElements", "SomeStreamer just raided the channel with 34 viewers", 0, "raid"),
        ("Nightbot", "Follow on twitter!", 0, "bot"),
    ],
)
def test_classify_message(user, text, bits, expected):
    assert classify_message(user, text, bits) == expected


def test_twitchdownloader(tmp_path):
    data = {"comments": [
        {"content_offset_seconds": 5.5, "commenter": {"display_name": "Ann"},
         "message": {"body": "hi Kappa", "bits_spent": 0,
                     "fragments": [{"text": "hi "}, {"text": "Kappa", "emoticon": {"emoticon_id": "25"}}]}},
        {"content_offset_seconds": 1.0, "commenter": {"name": "bob"},
         "message": {"body": "cheer100", "bits_spent": 100}},
    ]}
    path = tmp_path / "chat.json"
    path.write_text(json.dumps(data))
    fmt, messages = load_chat(path)
    assert fmt == "twitchdownloader"
    assert [m.t for m in messages] == [1.0, 5.5]  # sorted
    assert messages[0].kind == "bits" and messages[0].bits == 100
    assert messages[1].emotes == ["Kappa"]


def test_youtube_live_chat(tmp_path):
    def action(offset, renderer, body):
        return json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(offset), "actions": [
            {"addChatItemAction": {"item": {renderer: body}}}]}})

    lines = [
        action(2000, "liveChatTextMessageRenderer", {
            "authorName": {"simpleText": "Ann"},
            "message": {"runs": [{"text": "so good "}, {"emoji": {"shortcuts": [":fire:"]}}]}}),
        action(3500, "liveChatPaidMessageRenderer", {
            "authorName": {"simpleText": "Rich"}, "purchaseAmountText": {"simpleText": "$10.00"},
            "message": {"runs": [{"text": "great stream"}]}}),
        action(4000, "liveChatViewerEngagementMessageRenderer", {}),
    ]
    path = tmp_path / "v.live_chat.json"
    path.write_text("\n".join(lines))
    fmt, messages = load_chat(path)
    assert fmt == "youtube"
    assert len(messages) == 2
    assert messages[0].text == "so good :fire:" and messages[0].emotes == [":fire:"]
    assert messages[1].kind == "donation" and messages[1].amount == "$10.00"


def test_chatterino(tmp_path):
    log = tmp_path / "somechannel-2026-10-30.log"
    log.write_text(
        "# Start logging at 2026-10-30 18:59:00\n"
        "[19:00:10] ann: hello\n"
        "[19:01:00] bob (Bob): nice shot\n"
        "[19:02:00] subguy subscribed at Tier 1. They've subscribed for 2 months!\n"
        "[18:00:00] early: before the VOD\n"
    )
    fmt, messages = load_chat(log, vod_start=datetime(2026, 10, 30, 19, 0, 0))
    assert fmt == "chatterino"
    assert [(m.t, m.user, m.kind) for m in messages] == [
        (10.0, "ann", "message"), (60.0, "bob", "message"), (120.0, "subguy", "sub"),
    ]


def test_chatterino_start_with_a_timezone(tmp_path):
    log = tmp_path / "c-2026-10-30.log"
    log.write_text("[19:00:10] ann: hello\n")
    start = datetime(2026, 10, 30, 19, 0, 0).astimezone()  # this machine's zone, as an offset
    _, messages = load_chat(log, vod_start=start)
    assert [m.t for m in messages] == [10.0]


def test_chatterino_needs_start(tmp_path):
    log = tmp_path / "c-2026-10-30.log"
    log.write_text("[19:00:10] ann: hello\n")
    assert detect_format(log) == "chatterino"
    with pytest.raises(ValueError, match="chat-start"):
        load_chat(log)


def test_duration_filter_and_roundtrip(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps({"comments": [
        {"content_offset_seconds": t, "commenter": {"name": "a"}, "message": {"body": "x"}}
        for t in (1, 50, 500)
    ]}))
    _, messages = load_chat(path, duration=100)
    assert [m.t for m in messages] == [1, 50]
    saved = tmp_path / "saved.json"
    save_chat(saved, "twitchdownloader", str(path), messages)
    assert read_saved_chat(saved) == messages
    assert ChatMessage(1, "a", "x").to_dict() == {"t": 1, "user": "a", "text": "x"}
