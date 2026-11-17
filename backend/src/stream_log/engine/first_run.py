"""First-run setup for the installed app: download what the installer
leaves out because of its size, into the data home.

- **CUDA libraries** (cuBLAS, cuDNN, NVRTC: ~2 GB unpacked) for CTranslate2 on NVIDIA GPUs: the
  `nvidia-*` wheels pinned in uv.lock, checked against their hashes and unpacked into
  `<data home>/runtimes/cuda`, which `gpu.register_cuda_dlls` adds to `sys.path`.
- **llama.cpp** (the official ggml-org release this app was tested with): CUDA 13 build when the
  driver supports it, else CUDA 12.4, else the CPU build, into `runtimes/llama.cpp/b<build>`.
- **Models** of the machine's tier (`sync.tier_models`), from Hugging Face.

Every part is skipped once present and resumes from what's there, so `stream-log setup` can be run
again after an interrupted download.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from stream_log.engine.config import Paths
from stream_log.engine.hardware import HardwareInfo, Tier

CUDA_DIR = Path("runtimes") / "cuda"
CUDA_MARKER = "installed.json"
GPU_WHEELS_FILE = Path(__file__).with_name("gpu_wheels.json")  # written by app/scripts/package.mjs
# In a checkout, the wheels come from the lock file itself.
UV_LOCK = Path(__file__).resolve().parents[3] / "uv.lock"  # backend/uv.lock in a checkout

LLAMA_BUILD = "b11701"  # Pinned runtime build for reproducible model execution.
_LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/"
# name -> (sha256, bytes), from the release's asset digests
LLAMA_ASSETS = {
    "cuda-13.4": [
        (f"llama-{LLAMA_BUILD}-bin-win-cuda-13.4-x64.zip",
         "8cd15ea4ebf8da8467dcd625437e5862424239a125081b201893ce79b73f889f", 154748314),
        ("cudart-llama-bin-win-cuda-13.4-x64.zip",
         "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668", 423535356),
    ],
    "cuda-12.4": [
        (f"llama-{LLAMA_BUILD}-bin-win-cuda-12.4-x64.zip",
         "ed3e783fc3859d7bcb92dd44613635ea4b0c28130e4201a6332b51faef454810", 266832960),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip",
         "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6", 391443627),
    ],
    "cpu": [
        (f"llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip",
         "2cf9ad0c3bb449386ccae1b31ed091821a9aa773def9dbb94569deaed51d6dfe", 19611225),
    ],
}
# CUDA 13 needs an R580 driver or newer.
CUDA13_DRIVER = 580

Progress = Callable[[str, int, int], None]  # label, bytes done, bytes total


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    sha256: str
    size: int


@dataclass
class SetupPlan:
    tier: Tier
    cuda: list[Download] = field(default_factory=list)
    llama_variant: str | None = None
    llama: list[Download] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    models_gb: float = 0.0

    @property
    def download_gb(self) -> float:
        files = sum(d.size for d in self.cuda + self.llama)
        return round(files / 1e9 + self.models_gb, 1)

    @property
    def empty(self) -> bool:
        return not (self.cuda or self.llama or self.models)

    def to_json(self) -> dict:
        return {"tier": str(self.tier), "cuda": bool(self.cuda), "llama": self.llama_variant
                if self.llama else None, "models": self.models, "download_gb": self.download_gb}


# ---------------------------------------------------------------------- what is needed


def gpu_wheels() -> list[Download]:
    """The `nvidia-*` wheels of the backend's GPU extra, for Windows x64."""
    if GPU_WHEELS_FILE.exists():
        return [Download(**w) for w in json.loads(GPU_WHEELS_FILE.read_text(encoding="utf-8"))]
    if UV_LOCK.exists():
        return wheels_from_lock(UV_LOCK.read_text(encoding="utf-8"))
    return []


def wheels_from_lock(text: str) -> list[Download]:
    found = []
    for package in tomllib.loads(text).get("package", []):
        if not package["name"].startswith("nvidia-"):
            continue
        for wheel in package.get("wheels", []):
            if wheel["url"].endswith("win_amd64.whl"):
                found.append(Download(wheel["url"].rsplit("/", 1)[-1], wheel["url"],
                                      wheel["hash"].removeprefix("sha256:"), wheel.get("size", 0)))
    return found


def cuda_dir(paths: Paths) -> Path:
    return paths.home / CUDA_DIR


def cuda_ready(paths: Paths) -> bool:
    """cuBLAS/cuDNN importable: installed with the backend (development) or by setup."""
    if (cuda_dir(paths) / CUDA_MARKER).exists():
        return True
    try:
        import nvidia  # noqa: F401
    except ImportError:
        return False
    return True


def llama_variant(info: HardwareInfo, tier: Tier) -> str:
    gpu = info.best_gpu
    if tier == Tier.CPU or gpu is None:
        return "cpu"
    major = re.match(r"\d+", gpu.driver or "")
    return "cuda-13.4" if major and int(major.group()) >= CUDA13_DRIVER else "cuda-12.4"


def llama_ready(paths: Paths) -> bool:
    import os

    from stream_log.engine.llm_server import EXE_ENV, runtime_builds

    return bool(runtime_builds(paths.home) or os.environ.get(EXE_ENV)
                or shutil.which("llama-server"))


def tier_models(tier: str) -> list[str]:
    """Every model the analysis uses on `tier`: speech, sound, text, vision and each mode's
    "events" LLMs."""
    from stream_log.engine.analysis import (
        SOUND_EVENT_MODEL,
        TEXT_EMBED_MODEL,
        TEXT_EMOTION_MODEL,
        VISION_MODEL,
    )
    from stream_log.engine.models import load_manifest
    from stream_log.engine.pipeline import ALIGN_MODEL, SPEAKER_MODEL

    preset = load_manifest().tiers[tier]
    ids = [preset.asr, preset.verbatim_asr, ALIGN_MODEL, SPEAKER_MODEL, SOUND_EVENT_MODEL,
           TEXT_EMBED_MODEL, TEXT_EMOTION_MODEL, VISION_MODEL]
    for mode in preset.modes:
        ids.extend(preset.mode_roles(mode).get("events", []))
    return list(dict.fromkeys(i for i in ids if i))


def plan(paths: Paths, info: HardwareInfo, tier: Tier) -> SetupPlan:
    from stream_log.engine.models import load_manifest

    result = SetupPlan(tier)
    if tier != Tier.CPU and not cuda_ready(paths):
        result.cuda = gpu_wheels()
    if not llama_ready(paths):
        result.llama_variant = llama_variant(info, tier)
        result.llama = [Download(name, _LLAMA_URL + name, sha, size)
                        for name, sha, size in LLAMA_ASSETS[result.llama_variant]]
    manifest = load_manifest()
    for model_id in tier_models(tier):
        entry = manifest.get(model_id)
        if entry.source.startswith("hf:") and not entry.is_downloaded(paths.models):
            result.models.append(model_id)
            result.models_gb += entry.size_gb
    result.models_gb = round(result.models_gb, 1)
    return result


# ---------------------------------------------------------------------- doing it


def fetch(item: Download, folder: Path, progress: Progress) -> Path:
    """Download `item` into `folder` (resuming a `.part` file) and check its hash."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / item.name
    if target.exists() and _sha256(target) == item.sha256:
        return target
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(item.url, headers={"User-Agent": "StreamLog-setup"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if have and response.status != 206:  # the server ignored the range: start over
            have = 0
        total = have + int(response.headers.get("Content-Length") or 0) or item.size
        with part.open("ab" if have else "wb") as out:
            done = have
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                progress(item.name, done, total)
    digest = _sha256(part)
    if digest != item.sha256:
        part.unlink()
        raise RuntimeError(f"{item.name}: checksum mismatch (got {digest}); run setup again")
    part.replace(target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def install_cuda(paths: Paths, wheels: list[Download], progress: Progress) -> Path:
    """Unpack the CUDA wheels into runtimes/cuda (a wheel is a zip of its site-packages files)."""
    target = cuda_dir(paths)
    downloads = paths.cache / "downloads"
    for wheel in wheels:
        archive = fetch(wheel, downloads, progress)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(target)
    (target / CUDA_MARKER).write_text(json.dumps([w.name for w in wheels], indent=1),
                                      encoding="utf-8")
    for wheel in wheels:
        (downloads / wheel.name).unlink(missing_ok=True)
    return target


def install_llama(paths: Paths, assets: list[Download], progress: Progress) -> Path:
    target = paths.home / "runtimes" / "llama.cpp" / LLAMA_BUILD
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    downloads = paths.cache / "downloads"
    for asset in assets:
        archive = fetch(asset, downloads, progress)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(partial)
    # Some releases put the files in a subfolder; llm_server expects llama-server.exe on top.
    exe = next(partial.rglob("llama-server.exe"))
    if exe.parent != partial:
        for item in exe.parent.iterdir():
            item.replace(partial / item.name)
    shutil.rmtree(target, ignore_errors=True)
    partial.replace(target)
    for asset in assets:
        (downloads / asset.name).unlink(missing_ok=True)
    return target


def run(paths: Paths, todo: SetupPlan, progress: Progress,
        say: Callable[[str], None] = print) -> None:
    from stream_log.engine.models import download, load_manifest

    if todo.cuda:
        say(f"CUDA libraries -> {install_cuda(paths, todo.cuda, progress)}")
    if todo.llama:
        say(f"llama.cpp {LLAMA_BUILD} ({todo.llama_variant}) -> "
            f"{install_llama(paths, todo.llama, progress)}")
    manifest = load_manifest()
    for model_id in todo.models:
        entry = manifest.get(model_id)
        say(f"model {model_id} ({entry.size_gb:g} GB)...")
        download(entry, paths.models, say=say, progress=progress)
    say("setup done")


def add_cuda_to_path(home: Path) -> None:
    """Make the CUDA wheels setup unpacked importable (the `nvidia` namespace package)."""
    folder = home / CUDA_DIR
    if (folder / CUDA_MARKER).exists() and str(folder) not in sys.path:
        sys.path.append(str(folder))
