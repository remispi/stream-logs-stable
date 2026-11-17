"""CUDA runtime setup for libraries that load cuBLAS/cuDNN at runtime (CTranslate2)."""

from __future__ import annotations

import os
from pathlib import Path

_registered: list[Path] | None = None


def register_cuda_dlls() -> list[Path]:
    """Expose DLLs from pip-installed `nvidia-*` wheels to CTranslate2 on Windows.

    Must run before `import ctranslate2` / `faster_whisper`. Safe to call repeatedly.
    """
    global _registered
    if _registered is not None:
        return _registered
    _registered = []
    # The installed app downloads them on first run (first_run.py) instead of bundling them.
    from stream_log.engine.config import get_paths
    from stream_log.engine.first_run import add_cuda_to_path

    add_cuda_to_path(get_paths().home)
    try:
        import nvidia  # namespace package provided by nvidia-cublas-cu12 / nvidia-cudnn-cu12
    except ImportError:
        return _registered
    for base in nvidia.__path__:
        for bin_dir in sorted(Path(base).glob("*/bin")):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(bin_dir))
            # CTranslate2 resolves some DLLs through PATH rather than the DLL search list.
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            _registered.append(bin_dir)
    return _registered
