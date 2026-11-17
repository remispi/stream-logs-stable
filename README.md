# Stream Log

A Windows app that turns a stream VOD into a short timestamped list of what happened, meant to
be read at about a minute per hour of stream, so you can go and clip or edit from it.

- **Get the VOD**: type a Twitch account and pick one of its past broadcasts. It's downloaded to
  your Raw Streams folder as `TwitchStream<YYYYMMDD>-<TitleInPascalCase>.mp4` (the day the stream
  started). Or use a video you already have, optionally matched to its Twitch VOD for chat and
  Twitch links.
- **The list**: games and topic blocks as headers, then one short line per moment, in order, each
  tagged with why it was flagged: `topic` and `play` (described by a local LLM from the transcript),
  `laugh` and `loud` (heard in the audio), `chat ×N` (chat that many times busier than usual),
  raids and subs. At most 14 lines per hour of stream (Settings); the rest fold into "+N smaller".
- **Games**: TFT games are found from the video alone (loading screen, game in progress, post-game
  lobby), with their length. Longer stretches of TFT on screen with no lobby are shown as
  "TFT on screen" blocks instead. A game being watched (spectated, a tournament broadcast) looks
  like one being played, so the transcript decides: someone watching says "he" and "his" far more
  than "I" and "my". Those show as "Watched game" or "Watching TFT" and aren't counted as games
  played (English streams only).
- **Costreams**: line two VODs up from the audio they share (the same voice chat). Each list then
  shows the other VOD's time next to every moment, and says where either stream dropped.

The list is written next to the video as `<video>.timestamps.txt` and `.timestamps.html`; in the
HTML, a click on a time opens the video 15 s before it, and a second link opens the Twitch VOD there.

## How it works

Everything runs on your PC with local models; only the VOD, chat and first-run downloads need the
internet. The analysis (`backend/src/stream_log/engine/`) transcribes the stream, times every word,
tells voices apart, reads audio, chat and video signals, cuts the stream into chapters and has a
local LLM (through llama.cpp) describe what happens, keeping only events whose quotes are found in
the transcript. On top of that, Stream Log adds the Twitch side (VOD lists, chat download, file
naming), the TFT game detector (`games.py`, a small classifier on SigLIP frame embeddings,
`screens.npz`), the list itself (`listing.py`, `output.py`) and the costream sync (`sync.py`).

The first run downloads the speech, voice, sound, text and vision models, the LLMs of the chosen
mode, llama.cpp and (on NVIDIA GPUs) the CUDA libraries into the data folder,
`%LOCALAPPDATA%\StreamLog` by default (Settings). Plan for several GB.

## Requirements

Windows 10/11 x64. An NVIDIA GPU is strongly recommended (the LLM and speech models are slow on
the CPU); the app picks models for the GPU's memory.

## Layout

- `backend/`: the `stream_log` Python package and its command line (`python -m stream_log`).
  `engine/` is the analysis, `models/manifest.toml` in it lists every model and its licence.
- `app/`: the Electron window (`electron/`, `ui/`) and the installer build (`scripts/package.mjs`).
- `HANDOVER.md`: how everything fits together, for anyone working on the code.

## Development

Needs [uv](https://docs.astral.sh/uv/), Python 3.12 (from python.org under Smart App Control),
FFmpeg on PATH and Node.js 22.

```
cd backend
uv sync --python <python 3.12>
.venv\Scripts\python -m pytest
.venv\Scripts\python -m stream_log setup
.venv\Scripts\python -m stream_log vods <account>
.venv\Scripts\python -m stream_log run --file <video> [--vod <twitch link>]

cd ..pp
npm install
npm start                  # the window, on backend\.venv
npm run installer          # release\Stream Log Setup <version>.exe
```

Set `STREAM_LOG_HOME` to use another data folder. Add `--no-games` to `run` (or turn off "Find TFT
games" in Settings) for streams without TFT. If Twitch changes its web client,
`STREAM_LOG_TWITCH_CLIENT_ID` and `STREAM_LOG_TWITCH_COMMENTS_HASH` override the built-in values.

The game detector is retrained with `backend/tools/train_screens.py` from hand-labelled VODs
(labels and frame embeddings stay outside the repository).

## Contributing

Issues and pull requests are welcome. Run `pytest` and `ruff check` in `backend/` before sending
a change, and see `HANDOVER.md` for the conventions.

## Licence

The code is under the [MIT licence](LICENSE). The models it downloads are not part of this
repository and keep their own licences (listed in `backend/src/stream_log/engine/models/manifest.toml`).
