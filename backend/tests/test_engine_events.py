import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import numpy as np
import pytest

from stream_log.engine import events as ev
from stream_log.engine.ingest.chat import ChatMessage
from stream_log.engine.llm import LlmClient, LlmError, find_llm, parse_json

LINES = [
    {"id": 0, "start": 10.0, "end": 14.0, "text": "welcome back everyone, today we play elden ring"},
    {"id": 1, "start": 70.0, "end": 75.0, "text": "I'll be streaming tomorrow at 6 with my friend"},
    {"id": 2, "start": 130.0, "end": 136.0, "text": "my cat knocked over the plant this morning"},
]


def test_find_quote_grounding():
    assert ev.find_quote("streaming tomorrow at 6", 72, LINES, []) is not None
    assert ev.find_quote("Streaming tomorrow, at 6!", 100, LINES, []) is not None  # within 45 s
    assert ev.find_quote("streaming tomorrow at 6", 400, LINES, []) is None  # far off in time
    assert ev.find_quote("we beat the final boss", 72, LINES, []) is None  # invented
    chat = [ChatMessage(80.0, "fan", "pog the collab is happening")]
    assert ev.find_quote("the collab is happening", 75, LINES, chat) == (80.0, "chat")


def test_grounding_rejects_invented_endings_and_generic_phrases():
    lines = [{"id": i, "start": 100.0 + i * 5, "end": 104.0 + i * 5, "text": text} for i, text in
             enumerate(["I mean the new patch is so fast", "I mean whatever", "I mean yeah",
                        "I mean okay so we go again", "I think Bramble is still playable",
                        "because the horse rules are still very strong"])]
    # Words spread over several lines still ground.
    assert ev.find_quote("Bramble is still playable because the horse rules are still very strong",
                         120, lines, []) is not None
    # A real start with an invented ending does not.
    assert ev.find_quote("I think Bramble is still playable and I will win this lobby",
                         120, lines, []) is None
    # "i mean" is said in every other line, so it proves nothing.
    assert ev.find_quote("I mean", 105, lines, []) is None
    # A generic piece of a stitched quote is skipped; the distinctive piece still counts.
    assert ev.find_quote("I mean... the new patch is so fast", 100, lines, []) == (100.0,
                                                                                  "transcript")


def test_parse_json_tolerates_wrappers():
    assert parse_json('<think>hmm</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! [{"x": 2}]') == {"items": [{"x": 2}]}
    # the template opened the block in the prompt; a block cut short has no answer after it
    assert parse_json('maybe {"no": 1}?</think>{"a": 2}') == {"a": 2}
    with pytest.raises(LlmError):
        parse_json('<think>I could answer {"no": 1} but')


@pytest.mark.parametrize(("text", "value"), [
    ("$25", 25.0), ("$25.00", 25.0), ("€2,50", 2.5), ("$1,000.00", 1000.0),
    ("€1.000,50", 1000.5), ("£1,000", 1000.0), ("$10.5", 10.5)])
def test_donation_amounts(text, value):
    assert ev._amount_value(ev._AMOUNT.search(text).group(1)) == value


def test_community_events_and_bursts():
    chat = [
        ChatMessage(10, "StreamElements", "bigfan just tipped $25.00!", "donation"),
        ChatMessage(20, "g", "g is gifting 10 Tier 1 Subs to the community!", "gift"),
        *[ChatMessage(30 + i, f"u{i}", f"u{i} subscribed at Tier 1.", "sub") for i in range(6)],
    ]
    found = ev.community_events(chat, [])
    donation = next(e for e in found if e["type"] == "donation")
    assert donation["people"] == ["bigfan"] and donation["importance"] == "key"
    assert next(e for e in found if e["type"] == "gift")["importance"] == "key"
    subs = [e for e in found if e["type"] == "sub"]
    assert len(subs) == 1 and subs[0]["count"] == 6


def test_structure_events_need_silence():
    segments = [{"start": 0, "end": 120, "mode": "static", "confidence": 0.9},
                {"start": 120, "end": 600, "mode": "gaming", "confidence": 0.9}]
    quiet = np.zeros(600)
    talking = np.ones(600)
    assert [e["type"] for e in ev.structure_events(segments, [], quiet[:], 600)
            if e["type"] != "silence"] == ["starting_soon"]
    assert ev.structure_events(segments, [], talking, 600) == []


def test_merge_combines_sources():
    a = ev.event(70, 75, "announcement", "announcement", "Announcement", source="transcript_regex",
                 follow_up=True)
    b = ev.event(71, 80, "announcement", "schedule", "Streaming tomorrow at 6", source="llm",
                 description="Next stream tomorrow 6pm", importance="notable")
    merged = ev.merge_events([a, b])
    assert len(merged) == 1
    assert merged[0]["sources"] == ["llm", "transcript_regex"]
    assert merged[0]["title"] == "Streaming tomorrow at 6" and merged[0]["follow_up"]


class _FakeLlm(BaseHTTPRequestHandler):
    reply: ClassVar[dict] = {}

    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send({"choices": [{"message": {"content": json.dumps(self.reply)}}]})


@pytest.fixture
def fake_llm():
    server = HTTPServer(("127.0.0.1", 0), _FakeLlm)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def _llm_event(time, quote, title):
    return {"time": time, "end_time": "", "category": "announcement", "type": "schedule",
            "title": title, "description": "", "quote": quote, "importance": "notable",
            "people": [], "follow_up": True}


def test_llm_events_are_grounded(fake_llm):
    _FakeLlm.reply = {"events": [
        _llm_event("0:01:10", "I'll be streaming tomorrow at 6", "Next stream tomorrow"),
        _llm_event("0:02:10", "we are giving away a PC", "PC giveaway"),  # invented
    ]}
    client = find_llm(fake_llm)
    assert isinstance(client, LlmClient) and client.model == "fake-model"
    chapters = [{"id": "ch_01", "start": 0, "end": 200, "title": "Intro", "mode": "chatting"}]
    kept, dropped = ev.llm_events(client, chapters, LINES, {}, [])
    assert [e["title"] for e in kept] == ["Next stream tomorrow"]
    assert dropped == 1
    assert kept[0]["start"] == 70.0 and kept[0]["follow_up"]


def test_find_llm_returns_none_without_server():
    assert find_llm("http://127.0.0.1:9/v1") is None


def test_gpt_oss_gets_reasoning_hint_and_truncation_is_an_error(fake_llm):
    seen = {}

    def post(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.update(body)
        self._send({"choices": [{"finish_reason": "length", "message": {"content": ""}}]})

    original = _FakeLlm.do_POST
    _FakeLlm.do_POST = post
    try:
        client = LlmClient(fake_llm, "gpt-oss-20b")
        with pytest.raises(LlmError, match="token limit"):
            client.chat_json("sys", "user", {"type": "object"}, max_tokens=100)
    finally:
        _FakeLlm.do_POST = original
    assert seen["messages"][0]["content"].startswith("Reasoning: low")
    assert seen["max_tokens"] > 100


@pytest.mark.parametrize("answer", [None, b"not json", b'{"error": "busy"}'])
def test_a_dropped_or_odd_answer_is_an_llm_error(fake_llm, answer):
    def post(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if answer is None:  # the server died mid-answer (out of memory)
            self.close_connection = True
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)

    original = _FakeLlm.do_POST
    _FakeLlm.do_POST = post
    try:
        with pytest.raises(LlmError):
            LlmClient(fake_llm, "fake-model").chat("sys", "user")
    finally:
        _FakeLlm.do_POST = original


def test_plain_paragraph_strips_markdown():
    from stream_log.engine.chapters import plain_paragraph

    text = "**0:00:00 Stories About Pixel**\nThe streamer **talks** about Pixel.\n# Heading\nMore."
    assert plain_paragraph(text) == "The streamer talks about Pixel. More."


def test_serve_args_use_manifest_overrides(tmp_path):
    from stream_log.engine.llm_server import server_args
    from stream_log.engine.models import load_manifest

    manifest = load_manifest()
    for model_id in ("qwen3.6-35b-a3b", "qwen3.8-27b"):
        folder = tmp_path / model_id
        folder.mkdir()
        (folder / "model.gguf").write_bytes(b"")
        (folder / "mmproj-F16.gguf").write_bytes(b"")
    moe = server_args(manifest.get("qwen3.6-35b-a3b"), tmp_path)
    dense = server_args(manifest.get("qwen3.8-27b"), tmp_path)
    assert "--n-cpu-moe" in moe and "--fit-target" not in moe
    assert "--fit-target" in dense and "--n-cpu-moe" not in dense
    assert moe[moe.index("-m") + 1].endswith("model.gguf")


def test_parse_clock_forms():
    assert ev.parse_clock("0:36:35") == 2195
    assert ev.parse_clock("0:36:35-0:36:57") == 2195
    assert ev.parse_clock("[01:02]") == 62
    assert ev.parse_clock("soon") is None


def test_stitched_quotes_need_every_piece():
    assert ev.find_quote("welcome back everyone... I'll be streaming tomorrow", 40, LINES, [])         == (10.0, "transcript")
    assert ev.find_quote("welcome back everyone... we won the whole tournament", 40, LINES, [])         is None
