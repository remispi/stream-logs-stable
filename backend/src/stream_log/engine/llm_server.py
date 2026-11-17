"""Start and stop llama-server for models from the manifest.

The executable comes from $STREAM_LOG_LLAMA_SERVER, PATH, or the newest llama.cpp release in
`<data home>/runtimes/llama.cpp/<build>/` that is at least two weeks old, else the newest there
(Smart App Control only runs builds it already trusts).
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from stream_log.engine.models.manifest import ModelEntry

EXE_ENV = "STREAM_LOG_LLAMA_SERVER"
DEFAULT_PORT = 8080
# Keep context within the high-tier memory budget. Chapter prompts are capped at about 4k
# tokens, so the 16k default leaves room for model weights and inference overhead.
DEFAULT_CONTEXT = 16384
FIT_MARGIN_MB = 512
CONTEXT_CHECKPOINTS = 4


MIN_RUNTIME_AGE_DAYS = 14
START_TIMEOUT = 900  # big models from a cold disk take a few minutes


def runtime_builds(home: Path) -> list[Path]:
    """llama-server.exe of each `runtimes/llama.cpp/b<build>` folder, newest build first."""
    root = home / "runtimes" / "llama.cpp"
    found = []
    for folder in root.glob("b*"):
        exe = folder / "llama-server.exe"
        if exe.is_file() and re.fullmatch(r"b\d+", folder.name):
            found.append((int(folder.name[1:]), exe))
    return [exe for _, exe in sorted(found, reverse=True)]


def find_exe(explicit: Path | None = None, home: Path | None = None) -> Path:
    for candidate in (explicit, os.environ.get(EXE_ENV), shutil.which("llama-server")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    if home is not None:
        builds = runtime_builds(home)
        cutoff = time.time() - MIN_RUNTIME_AGE_DAYS * 86400
        aged = [exe for exe in builds if exe.stat().st_mtime <= cutoff]
        if aged or builds:
            return (aged or builds)[0]
    raise FileNotFoundError(
        f"llama-server not found: run stream-log setup, set {EXE_ENV}, put llama-server on "
        "PATH, or unpack a llama.cpp release into <data home>/runtimes/llama.cpp/b<build>/"
    )


def free_port(preferred: int = DEFAULT_PORT) -> int:
    """`preferred` if nothing listens on it (e.g. a server the user started), else any free port."""
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


@contextmanager
def managed_server(entry: ModelEntry, models_root: Path, exe: Path, log_path: Path,
                   *, context: int | None = None) -> Iterator[str]:
    """Run llama-server for `entry` while the block runs; yields its OpenAI-compatible base URL.

    One model at a time: the GPU holds one of these models, never two.
    """
    port = free_port()
    args = server_args(entry, models_root, port=port, context=context)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        # llama-server loads its ggml/CUDA DLLs from its own folder.
        process = subprocess.Popen([str(exe), *args], cwd=exe.parent, stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + START_TIMEOUT
            while not _healthy(port):
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited while loading {entry.id} "
                                       f"(code {process.returncode}); see {log_path}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{entry.id} did not load within {START_TIMEOUT} s")
                time.sleep(1)
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


ADAPTER_FILE = "adapter.gguf"


def model_file(entry: ModelEntry, models_root: Path) -> Path:
    """The language model GGUF (not the mmproj vision projector); an adapter's base model's."""
    if entry.base_entry is not None:
        return model_file(entry.base_entry, models_root)
    if entry.local_file is not None:
        if not entry.local_file.is_file():
            raise FileNotFoundError(f"{entry.id}: {entry.local_file} is gone")
        return entry.local_file
    files = [f for f in sorted(entry.local_dir(models_root).glob("*.gguf"))
             if not f.name.lower().startswith("mmproj")]
    if not files:
        raise FileNotFoundError(f"{entry.id}: no .gguf in {entry.local_dir(models_root)}")
    return files[0]


def adapter_file(entry: ModelEntry, models_root: Path) -> Path | None:
    """A LoRA adapter's GGUF (trained in the app), or None for a plain model."""
    if entry.base_entry is None:
        return None
    path = entry.local_dir(models_root) / ADAPTER_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{entry.id}: no {ADAPTER_FILE} in {entry.local_dir(models_root)}")
    return path


def serve_settings(entry: ModelEntry) -> dict:
    """An adapter runs with its base model's settings, then its own."""
    return {**(entry.base_entry.serve if entry.base_entry else {}), **entry.serve}


def model_identity(entry: ModelEntry, models_root: Path) -> str:
    """Changes when the weights change (the LLM stages' cache keys)."""
    identity = model_file(entry, models_root).name
    adapter = adapter_file(entry, models_root)
    if adapter is not None:
        identity += f"+{entry.id}:{adapter.stat().st_size}"
    return identity


def server_args(entry: ModelEntry, models_root: Path, *, port: int = DEFAULT_PORT,
                context: int | None = None, reasoning: bool = False,
                fit_margin: int | None = None) -> list[str]:
    # No -ngl / --n-cpu-moe: llama.cpp then fits layers and MoE experts to free VRAM itself,
    # leaving `fit_margin` MiB free. Hybrid models (Qwen3.8) keep context checkpoints of ~150 MB
    # each, 32 by default; uncapped they filled VRAM and Windows spilled into shared memory
    # (1 tok/s). We send one prompt at a time, so a few checkpoints are enough.
    serve = serve_settings(entry)
    context = context or int(serve.get("context") or DEFAULT_CONTEXT)
    if fit_margin is None:
        fit_margin = int(serve.get("fit_margin", FIT_MARGIN_MB))
    if "n_cpu_moe" in serve:
        placement = ["-ngl", "99", "--n-cpu-moe", str(serve["n_cpu_moe"])]
    else:
        placement = ["--fit-target", str(fit_margin)]
    adapter = adapter_file(entry, models_root)
    return [
        "-m", str(model_file(entry, models_root)),
        *(["--lora", str(adapter)] if adapter else []),
        *placement,
        "--ctx-checkpoints", str(CONTEXT_CHECKPOINTS),
        # No prompt cache: every chapter prompt is different, and on a 3.7 h VOD the cache
        # (~750 MB per entry for Qwen3.8, 8 GB by default) filled system RAM after ~20 requests
        # and generation fell from 40 to 1 tok/s.
        "--cache-ram", "0",
        "--alias", entry.id,
        "-c", str(context),
        "-fa", "on",
        "--reasoning", "on" if reasoning else "off",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--parallel", "1",
        *[str(a) for a in serve.get("extra_args", [])],
    ]
