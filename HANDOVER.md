# Stream Log — Handover Guide

> Audience: a developer taking over Stream Log. This document explains **what the app is**, **how it's built**, **what every part does**, **the core workflows**, **how to get it running**, and **where the work stands**. Read the [Glossary](#2-glossary-read-this-first) first: its words (project, stage, tier, mode, block, moment…) are used everywhere in the code.
>
> User-facing docs: [`README.md`](README.md). This guide adds the developer's view and the hand-off notes (§12).

---

## 1. What is Stream Log?

Stream Log is an **offline Windows desktop app** for streamers. It takes a **stream VOD** (a public Twitch past broadcast it downloads, or a video already on disk) and writes a **short timestamped list of what happened**, meant to be read at about **a minute per hour of stream**, so the streamer can go and clip or edit from it.

The list has game and topic **blocks** as headers, then one short line per **moment**, in order, each tagged with why it was flagged: `topic` / `play` (described by a local LLM from the transcript), `laugh` / `yell` / `loud` (heard in the audio), `chat ×N` (chat busier than usual), raids and subs. It is written next to the video as `<video>.timestamps.txt` and `.timestamps.html`; in the HTML, a click on a time opens the video 15 s before it, and a second link opens the Twitch VOD there.

Key properties:
- **Offline.** Only the Twitch VOD list, VOD and chat downloads and the first-run model download need a connection. Nothing is sent anywhere else; Twitch is used without signing in (public VODs only).
- **Local models only.** Speech recognition (faster-whisper / CrisperWhisper), small ONNX models (word timing, voices, sound events, text embeddings and emotions, SigLIP) and local LLMs through **llama.cpp's `llama-server`**.
- **TFT-aware.** Teamfight Tactics games are found from the video alone, and games watched are told apart from games played by what the streamer says. This can be turned off (Settings, `--no-games`).
- **Costreams.** Two VODs that share audio (the same voice chat) can be lined up; each list then shows the other VOD's time next to every moment.
- **Built for Windows Smart App Control (SAC)**, which shapes several technical choices (§11).

**Tech stack:** Python 3.12 backend (managed with `uv`; argparse CLI that speaks JSON lines; NumPy; FFmpeg as an executable; ONNX Runtime; CTranslate2 through faster-whisper; yt-dlp for the video download), Electron 45 window with plain HTML/CSS/JS (no framework, no build step). Packaged as an NSIS `Setup.exe` (electron-builder). No database: JSON + NumPy files per project.

---

## 2. Glossary (read this first)

| Term | Meaning |
|------|---------|
| **VOD** | The recorded stream being turned into a list. One VOD = one **project**. |
| **Project** | A folder `projects/<id>/` in the data folder holding every cached artefact of the analysis for one video, plus `project.json` (metadata) and `stages.json` (stage cache keys). See `engine/project.py`. The id is the file name slugged plus a short hash of its path. |
| **Data folder** | Where models, projects and caches live: `%LOCALAPPDATA%\StreamLog`, or `STREAM_LOG_HOME` (the window sets it from Settings). Several GB; never inside the repo or the install folder. |
| **Raw Streams folder** | Where Twitch VODs are downloaded (`%USERPROFILE%\Videos\Raw Streams` by default) and where the lists are written, next to each video. |
| **Stage** | One cached step of the analysis (`engine.pipeline.STAGES`): `media → transcribe → align → words → speakers → audio_signals → chat_signals → visual → fusion → chapters → llm → events`. Once a stage re-runs, every stage after it re-runs. Then Stream Log's own steps: `games` and `list`. |
| **Proxy** | The 540p H.264 copy of the video (`proxy.mp4`) the visual stage and the game detector read. |
| **Stream mode** | Per-segment kind of content: `gaming | chatting | irl | reaction | static`, detected every 10 s (SigLIP + audio) and smoothed (`segments.json`). |
| **Signals / fusion** | 1 Hz arrays (audio, chat, visual) z-scored against a rolling baseline and fused (`signals.npz`); the list reads laughter, screams, voice energy ("arousal") and chat rate from them. |
| **Speaker tags** | Each transcript line tagged `streamer | guest | media | tts`. The streamer's voice can be enrolled in a **profile**. |
| **Profile** | `profiles/<name>/` in the data folder: `voice.npy` (the streamer's voice) and `profile.json` (streamer name, description fed to the LLM prompts). Optional; Settings picks one. |
| **Tier** | Hardware class picked by `engine/hardware.py`: `high` (≥ 15 GB VRAM and ≥ 30 GB RAM), `mid` (≥ 10 GB), `low` (≥ 7 GB), `cpu`. Decides which models and modes exist. |
| **Mode** | `fast | balanced | best`: which LLMs extract the events. In Best, three model families check each other's claims. Default per tier in the manifest. |
| **Role** | What an LLM does in a mode. There is one: **events** (extract + cross-check). Defined in the manifest `[tiers.<tier>.roles.<mode>]`, overridable in `<data folder>/models.toml`; roles left over from older model files are ignored. |
| **Panel** | The LLMs a run uses (`engine/panel.py`, `pipeline.Panel`). |
| **Grounding** | Every LLM event must **quote the transcript**; quotes not found near the timestamp are dropped (`engine/events.py`). |
| **Evidence / moment** | Evidence: one reason to flag a time (an LLM event, a laugh, a chat spike…). Moment: evidence within 30 s merged into one line of the list, scored (`listing.cluster`). |
| **Block** | A header of the list: a game (`Game 3 · 34 min`), a watched game, "TFT on screen", or a topic stretch of about 25 min along the chapters. Each keeps at most *per hour* lines for its length; the rest fold into "+N smaller". |
| **Game detector** | `games.py`: SigLIP embeddings of one frame every 10 s → a small classifier (`screens.npz`) → an HMM (Viterbi) over `game / loading / lobby / other` → games. |
| **Sync** | `sync.py`: two VODs lined up by cross-correlating their loudness; pieces of shared audio with a time offset (lag). Saved in both projects. |
| **Manifest / `models.toml`** | `engine/models/manifest.toml` (bundled: every model, its licence, tier presets) overlaid by `<data folder>/models.toml` (user overrides). |
| **Setup** | `stream-log setup` (`setup.py` + `engine/first_run.py`): first-run download of CUDA wheels, llama.cpp and the models the chosen mode needs. |
| **SAC** | Windows **Smart App Control**: blocks unsigned / unknown native binaries. See §11. |

---

## 3. Architecture at a glance

```
 Electron main (app/electron/main.cjs)
   settings.json, one job at a time, kills the whole process tree on Stop
   preload.cjs = the page's only bridge (a dozen actions)
        │  BrowserWindow (app/ui: index.html, app.js, style.css), sandboxed
        ▼
 python -m stream_log <command> ...      one process per job; stdout = one JSON object per line
        │
        ├─ twitch.py      VOD lists and details, chat (Twitch GraphQL), video (yt-dlp)
        ├─ run.py         one job: VOD → video → chat → analysis → games → list
        │     └─ engine.pipeline.process(until="events")   cached stages in the project folder
        │           ├─ FFmpeg (exe)       audio, proxy
        │           ├─ faster-whisper     transcribe;  ONNX Runtime: align, speakers, sounds, SigLIP…
        │           └─ llama-server       LLM steps, one model at a time on the GPU
        ├─ games.py       TFT games from the proxy
        ├─ listing.py     evidence → moments → blocks → StreamList
        ├─ output.py      .timestamps.txt / .timestamps.html next to the video
        └─ sync.py        costream line-up, then both lists written again
```

**Layer responsibilities:**

- **Window** (`app/`) — picks a VOD or a file, starts jobs, shows progress from the JSON lines, opens the result. Holds no logic about streams.
- **Command line** (`cli.py`) — the only entry point. Every command emits JSON lines; library prints go to stderr (`sys.stdout` is swapped while a command runs).
- **Jobs** (`run.py`, `sync.py`, `setup.py`) — orchestration, progress events.
- **Stream Log logic** (`twitch.py`, `naming.py`, `games.py`, `listing.py`, `output.py`) — pure-ish functions, tested in `tests/test_basics.py`.
- **Engine** (`engine/`) — the analysis: stage list and cache (`pipeline.py`, `analysis.py`, `project.py`), domain modules (`asr/`, `speakers/`, `signals/`, `chapters.py`, `events.py`, `wordcheck.py`), LLM layer (`llm.py`, `llm_server.py`, `panel.py`), models and setup (`models/`, `first_run.py`, `hardware.py`, `gpu.py`).

---

## 4. Directory & file reference

### Top-level

| Path | Purpose |
|------|---------|
| `backend/` | Python package `stream_log` + `stream-log` command. `uv` project (`pyproject.toml`, `uv.lock`). |
| `backend/tools/train_screens.py` | Retrains the game detector (`screens.npz`) from hand-labelled VODs. |
| `app/` | Electron window and the installer build. |
| `app/scripts/package.mjs` | Builds `release\Stream Log Setup <version>.exe` (`npm run installer`). |
| `app/scripts/make_icon.py` | Draws `electron/icon.png` / `icon.ico` (`npm run icon`). |
| `local/` | Gitignored: hand labels, frame embeddings, notes about real VODs. Not in the repo. |

### `backend/src/stream_log` — the app's own code

| File | Responsibility |
|------|----------------|
| `cli.py` | Commands `vods`, `vod`, `status`, `setup`, `run`, `relist`, `sync`; JSON lines on stdout. |
| `run.py` | `run()`: one job start to finish, `STEPS` (the progress list), reuse of an earlier analysis of the same Twitch VOD, `relist()` (write a list again). |
| `twitch.py` | `list_vods`, `vod_info` (+ chapter markers), `download_chat` (chat replay in TwitchDownloader's JSON shape), `download_video` (yt-dlp; tags the file with the VOD link), `is_same_vod`. |
| `naming.py` | `TwitchStream<YYYYMMDD>-<TitleInPascalCase>` file names, `bad_stem` (Windows-safe names), `free_path`. |
| `games.py` | Game detector (see Glossary), cached in the project as `stream_log_games.json`; `mark_watched` (pronoun share: "he/his" vs "I/my"). |
| `listing.py` | Evidence (LLM events, reactions, chat spikes), `cluster` into moments, `make_blocks`, `fill` (per-hour cap), `build` → `StreamList`. |
| `output.py` | The `.txt` and `.html` lists (links to the video `#t=` and to Twitch `?t=`). |
| `sync.py` | Costream line-up (loudness envelopes, normalised cross-correlation, pieces), saved in `stream_log_sync.json`; also which videos and list folders a project was used for (`stream_log_videos.json`). |
| `setup.py` | `status` (what's missing, tier, modes, profiles) and `run_setup`, limited to the models this app needs. |
| `screens.npz` | The game detector's classifier weights (versioned; the version is part of the games cache key). |

### `backend/src/stream_log/engine` — the analysis

| File | Responsibility |
|------|----------------|
| `config.py` | `Paths` / `get_paths()`: the data folder and its sub-folders. |
| `project.py` | A project folder: `project.json`, stage cache (`stage_key`, `is_fresh`, `record_stage`), JSON helpers, `clock`. |
| `pipeline.py` | `ingest()` (a local video + optional chat), `Settings`, `STAGES`, `resolve_panel()` (which LLMs), `process()` (runs stale stages). Also the media, transcribe, align, words and speakers stages. |
| `analysis.py` | The audio/chat/visual signals, fusion, chapters, llm and events stages. |
| `hardware.py` | GPU/RAM detection, tier choice. `gpu.py`: puts CUDA DLLs on the path for CTranslate2. |
| `media.py`, `audio.py` | Probe, proxy, sprite, waveform peaks; WAV extraction/reading — all through the FFmpeg executable. |
| `ingest/sources.py` | The video's project id and fingerprint. `ingest/chat.py`: TwitchDownloader JSON, YouTube `live_chat.json`, Chatterino logs → one message list. |
| `asr/transcribe.py`, `align.py`, `fillers.py` | Windowed faster-whisper; wav2vec2 CTC forced alignment (ONNX) + Viterbi; fillers and unworded speech. |
| `wordcheck.py`, `lexicon.py` | Each heard word checked as a normal word, a game term or a name (LLM, cached), against `glossary.txt` and `lexicon/game.txt` / `names.txt` in the data folder. |
| `speakers/` | WeSpeaker embeddings, clustering + streamer/guest/media/tts tags, profiles, voice names. |
| `signals/` | 1 Hz signals (loudness, sound events; chat vibe; motion, frozen video, stream modes), fusion and vibe vectors. |
| `onnx_models.py` | Shared small ONNX models: text embeddings, text emotions, SigLIP. |
| `chapters.py` | Topic shifts + mode changes + game switches, titled by the LLM. |
| `events.py` | Cheap detectors (raids, subs, donations, silences…) → LLM pass per chapter → grounding → merge → importance. |
| `llm.py`, `llm_server.py`, `panel.py` | OpenAI-compatible client (finds a running server: llama-server, Ollama, LM Studio, `STREAM_LOG_LLM_URL`); starts/stops `llama-server` per model; multi-model extraction + cross-checking. |
| `models/manifest.toml`, `manifest.py`, `user.py`, `download.py` | Model catalogue (licences, tiers, roles), user overlay `models.toml`, resumable downloads. |
| `first_run.py` | CUDA wheels (`gpu_wheels.json` in the installed app, else read from `uv.lock`), llama.cpp `LLAMA_BUILD` (b11701, pinned with hashes), the tier's models. |

### `app/`

| Path | Purpose |
|------|---------|
| `electron/main.cjs` | Picks the Python (`STREAM_LOG_PYTHON`, else `backend\.venv`, or the bundled embeddable Python when installed), settings, jobs, the IPC handlers, `--screenshot` mode for development. |
| `electron/preload.cjs` | The only bridge to the page. `contextIsolation` on, `nodeIntegration` off, sandboxed. |
| `ui/index.html`, `app.js`, `style.css` | The single page: Twitch / file tabs, progress, result, Settings dialog. Strict CSP, no `innerHTML`. |

### Tests

- `backend/tests/test_basics.py` — Stream Log's own logic (naming, Twitch links, listing, games, sync, output, CLI flags).
- `backend/tests/test_engine_*.py` — the engine (ASR helpers, chat formats, signals, speakers, events and grounding, LLM client and server args, manifest, user models, downloads, first run, projects, word check).
- Run with `.venv\Scripts\python -m pytest` and `.venv\Scripts\python -m ruff check src tests` in `backend/`. There are no front-end tests; `npm run screenshot` captures the window for a look.

---

## 5. Core workflows

### 5.1 Making a list (the central workflow)

```
stream-log run --vod <Twitch link> [--out DIR] [--name STEM]      # download it
stream-log run --file <video> [--vod <Twitch link>]               # a video you have
   [--mode fast|balanced|best] [--profile P] [--per-hour 14] [--no-games] [--no-reuse]
```

1. **VOD** (`--vod`): `twitch.vod_info` reads title, channel, start time, length and chapter markers.
2. **Video**: with `--file`, that file. Otherwise `<out>/<stem>.mp4`; if that name is taken by another VOD (checked with the link tag or the length), `<stem>-2.mp4`, `-3`… The name is checked by `naming.bad_stem`.
3. **Chat**: downloaded once to `<data folder>/cache/twitch-chat/<vod id>.json`. Optional: a failure is logged and the run goes on.
4. **Analysis**: if an earlier project for the same Twitch VOD already has its event log, it is reused (unless `--no-reuse`). Else `pipeline.ingest` + `pipeline.process(until="events")`. For each stage, `stage_key(name, version, config, upstream_keys)`; if the key and outputs match `stages.json` it is *cached*, otherwise it runs and **every later stage runs too**.
5. **LLM panel** (`resolve_panel`): the mode's events models from the manifest / `models.toml` → `llama-server` per model, one at a time. Fallbacks: managed models → a server already running → skip LLM steps.
6. **Games** (unless `--no-games`), then **the list**: `listing.build` → `output.write`.

The window does the same through `run` with the Settings values. `relist <project>` writes a list again without re-running the analysis (after changing lines per hour, for example), into the folder the list was last written to.

**When you change a stage's logic in a way that affects its output, bump its `version` in `STAGES`** — that is what invalidates old caches. For the game detector, retrain/replace `screens.npz` with a new `version`.

### 5.2 Costream sync

`stream-log sync <video A> <video B>` (or "Line up with another VOD…" after a list is made). Both must have been turned into lists first and have their stream start time (run with their Twitch link). The start times give a first guess of the offset; 4-minute windows every 2 minutes are matched around it; confident windows that agree become pieces. Both projects get `stream_log_sync.json` and both lists are written again.

### 5.3 First-run setup

`stream-log status` reports the tier, modes, what's missing and the download size; `stream-log setup [--mode M]` downloads the CUDA wheels (NVIDIA only), llama.cpp b11701 and the models the mode needs into the data folder. The window shows a "First-time setup" card while anything is missing.

### 5.4 Retraining the game detector

`backend/tools/train_screens.py` (see its docstring): embed hand-labelled VODs (`--embed`), run the leave-one-VOD-out check, write `screens.npz`. Labels and embeddings stay in `local/`, outside git.

---

## 6. Runtime & security

- **One process per job:** the window runs `python -s -m stream_log <command>` with `windowsHide`, reads JSON lines from stdout, appends stderr to `%APPDATA%\Stream Log\backend.log`. One job at a time; Stop / quitting kills the whole tree (`taskkill /T /F`, which also ends `llama-server`) and is reported as "Stopped.", not as an error.
- **Electron hardening:** `contextIsolation` on, `nodeIntegration` off, sandboxed, navigation and new windows denied, CSP `default-src 'self'`. The page can only open or reveal the lists a job reported writing and only ask whether video files exist. Keep it that way — add a preload action rather than loosening isolation.
- **Settings:** `%APPDATA%\Stream Log\settings.json` — `account`, `rawFolder`, `dataHome`, `mode`, `perHour`, `profile`, `findGames`, `theme`.
- **Packaged:** the backend runs on python.org's **embeddable Python 3.12.10** (PSF-signed; the build refuses an unsigned one), with FFmpeg copied from the build machine's PATH into `resources\backend\ffmpeg`.
- **Network:** Twitch GraphQL (`gql.twitch.tv`, Twitch's public web client ID), Twitch video servers through yt-dlp, Hugging Face / GitHub / PyPI for first-run downloads. No telemetry, no auto-updater.

---

## 7. CLI & JSON events

| Command | Emits |
|---------|-------|
| `vods <account> [--limit 30]` | `{"type": "vods", "vods": [...]}` with each VOD's file `stem` |
| `vod <url>` | `{"type": "vod", "vod": {...}}` |
| `status [--mode]` | `{"type": "status", "ready", "missing", "download_gb", "tier", "modes", "profiles", ...}` |
| `setup [--mode]` | `step` / `log` …, then `done` |
| `run …` | `steps` (the list of steps), `step` (`id`, `fraction`, `text`), `log`, then `done` (`video`, `text`, `html`, `project`, `games`, `lines`) |
| `relist <project> [--video] [--per-hour] [--no-games]` | `log`, `done` |
| `sync A B [--label-a] [--label-b] [--per-hour] [--no-games]` | `step`, `log`, `done` per list, then `sync` (`pieces`, `note_a`, `note_b`) |

Any failure ends with `{"type": "error", "text": "..."}` and exit code 1 (the traceback goes to stderr). `python -m stream_log <command> --help` is authoritative.

---

## 8. Local setup & running

**Prerequisites:** Windows x64, [uv](https://docs.astral.sh/uv/), **python.org** Python 3.12 (needed under SAC), FFmpeg (with ffprobe) on PATH, NVIDIA driver for GPU work, Node.js ≥ 22.

```bash
# 1. Backend
cd backend
uv sync --python <python.org 3.12>\python.exe
.venv\Scripts\python -m pytest
.venv\Scripts\python -m stream_log status       # tier + what's missing
.venv\Scripts\python -m stream_log setup        # CUDA, llama.cpp, models (several GB)

# 2. A first list (a 20-30 min file is enough to try)
.venv\Scripts\python -m stream_log run --file path\to\vod.mp4

# 3. Window
cd ..\app
npm install
npm start                                       # runs the backend from backend\.venv
npm run installer                               # release\Stream Log Setup <version>.exe
```

Put test data on a big drive with `STREAM_LOG_HOME` (or Settings > Data folder) — videos, proxies and models get large fast.

---

## 9. Configuration & environment

| Var / file | Purpose |
|------------|---------|
| `STREAM_LOG_HOME` | Data folder (default `%LOCALAPPDATA%\StreamLog`). The window sets it from Settings. |
| `STREAM_LOG_PYTHON` | Python the window uses to run the backend (development). |
| `STREAM_LOG_LLAMA_SERVER` | Path to `llama-server.exe` (else PATH, else the newest ≥ 2-week-old build in `runtimes/llama.cpp/`). |
| `STREAM_LOG_LLM_URL`, `STREAM_LOG_LLM_MODEL` | Use an already-running OpenAI-compatible server. |
| `STREAM_LOG_TWITCH_CLIENT_ID`, `STREAM_LOG_TWITCH_COMMENTS_HASH` | Override Twitch's web client ID / chat query hash when Twitch changes them. |
| `<data folder>/models.toml` | User models and per-mode choices (same format as the manifest). |
| `<data folder>/glossary.txt` | Channel names / game terms; `heard -> right` fixes. |
| `<data folder>/lexicon/game.txt`, `names.txt` | Game terms and player names for the word check. |
| `<data folder>/profiles/<name>/` | `voice.npy`, `profile.json`. |
| `backend/pyproject.toml` `[tool.uv] exclude-newer` | Dependency cut-off date (SAC two-week rule, §11). |
| `engine/first_run.py` `LLAMA_BUILD` | Pinned llama.cpp release (b11701) + hashes. |

**Files in a project folder:** `project.json`, `stages.json`, `audio.wav`, `peaks.json`, `proxy.mp4`, `sprite.jpg/.json`, `transcript.raw.json`, `transcript.json`, `words.json` (+ `words_runs/`), `speakers.json`, `speaker_embeddings.npz`, `audio_signals.npz`, `chat.json`, `chat_signals.npz`, `chat_interactions.json`, `visual.npz`, `segments.json`, `signals.npz`, `vibe.json`, `chapters.json`, `llm.json` (+ `llm_runs/`), `events.json`; Stream Log's own `stream_log_games.json`, `stream_log_sync.json`, `stream_log_videos.json`.

---

## 10. Distribution

- **Installer** (`npm run installer`, `scripts/package.mjs`): stages the window, embeddable Python with the `uv.lock` dependencies minus the `nvidia-*` wheels (their url/hash/size go to `engine/gpu_wheels.json` for the first-run download), the `stream_log` package and FFmpeg; electron-builder NSIS, per-user install, `asar` off, **unsigned**.
- **First run:** the window checks `status` and offers the setup download while anything is missing. Settings and the data folder survive updates and uninstall.
- No CI. Builds are made by hand; verify on a clean Windows PC (ideally with SAC on) before distributing.

---

## 11. Conventions, patterns & gotchas

- **Smart App Control drives the stack.** Before adding *any* native dependency:
  - Dependencies must be **≥ 2 weeks old** (`exclude-newer` in `pyproject.toml`; same rule for llama.cpp builds). Move the date forward, keeping the gap, when upgrading.
  - Prefer signed runtimes (python.org Python, ONNX Runtime, NVIDIA DLLs). Avoid new unsigned native modules when a little code does the job — that's why FFmpeg is used as an exe and there is no PyInstaller.
- **Stage cache:** change a stage's output-affecting logic → bump its `version` in `engine/pipeline.STAGES`. Forgetting this serves stale results silently. Projects made by older versions stay readable as long as file names and shapes don't change.
- **Grounding is non-negotiable:** any LLM output that claims something about the stream must quote the transcript and go through the same check.
- **Every prompt fits a 16k-token context** (so the models stay on the GPU).
- **stdout is for JSON lines only.** Use `emit` / the `log` callback; anything printed goes to stderr.
- **The window has no logic about streams.** New behaviour goes into the backend and a CLI flag; the window passes Settings through (`commonArgs` in `app.js`) and shows events.
- **Writing style:** docs, UI text and commit messages are plain sentences that state only what the code does.
- **Privacy:** real VOD content, labels and notes go in `local/` (gitignored), never in tracked files.
- **Generated, don't hand-edit:** `app/release/`, `electron/icon.ico` / `icon.png` (`npm run icon`), `gpu_wheels.json` (written at package time), `screens.npz` (`train_screens.py`).

### Known rough edges

1. **No CI** and no code signing: tests and packaging are run by hand; the installer may be blocked on SAC machines.
2. **Windows only:** paths, `taskkill`, the installer. macOS/Linux are untested.
3. **The game detector knows TFT only**, and the watched/played check works on English streams only.
4. **Twitch's GraphQL is unofficial:** the client ID and chat query hash can change (overridable, §9).
5. The media stage still writes `peaks.json` and `sprite.jpg`, which the list doesn't use (removing them would invalidate existing caches).
6. Voice profiles have no command to create them; they are folders made by hand (§9).

---

## 12. Current state & hand-off notes

**What works:** Twitch VOD lists and downloads (named, collision-safe), chat download, the full analysis up to the event log on local models, TFT games (played / watched / on screen), the list (txt + html, per-hour cap, Twitch links), costream sync, first-run setup, the installer.

**Suggested next steps:**
1. A CI job running `pytest` and `ruff` on Windows.
2. Code signing for the installer.
3. A command (and Settings button) to enrol a streamer voice profile from a few minutes of a VOD.
4. Game detectors for other games (the classifier + HMM in `games.py` is game-agnostic; it needs labels).
5. Front-end tests (the screenshot mode is the only check today).

---

## 13. Where to start

1. Read `README.md`, then §2 and §5 here.
2. Get the backend running (§8), run `pytest`, then make a list from a short VOD with `stream-log run --file`.
3. Open the `.timestamps.html` it writes, and the project folder in the data folder (§9) to see what each stage left.
4. Read `run.py` (one job end to end), then `listing.py` (how moments and blocks are chosen) and `games.py`.
5. Read `engine/pipeline.py` (stages + cache), then follow one stage, e.g. `engine/analysis.py: events_run` → `engine/events.py`.
6. Read `app/electron/main.cjs` and `app/ui/app.js` together: the JSON events are the contract between them.
7. Skim `tests/test_basics.py` and the `test_engine_*.py` files for behavioural expectations.

---

*Handover aid written from the repository state on 2026-11-17. Usage: `README.md`.*
