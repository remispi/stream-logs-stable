"""Hardware detection and tier selection."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import psutil


class Tier(StrEnum):
    HIGH = "high"
    MID = "mid"
    LOW = "low"
    CPU = "cpu"


@dataclass(frozen=True)
class Gpu:
    name: str
    vram_mb: int
    driver: str
    compute_capability: str | None = None


@dataclass(frozen=True)
class HardwareInfo:
    cpu: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    free_disk_gb: float
    gpus: list[Gpu] = field(default_factory=list)

    @property
    def best_gpu(self) -> Gpu | None:
        return max(self.gpus, key=lambda gpu: gpu.vram_mb, default=None)


# Thresholds sit a little under the marketing size: a "16 GB" card reports ~16300 MiB.
_HIGH_VRAM_MB = 15 * 1024
_HIGH_RAM_GB = 30
_MID_VRAM_MB = 10 * 1024
_LOW_VRAM_MB = 7 * 1024


def choose_tier(info: HardwareInfo) -> Tier:
    gpu = info.best_gpu
    if gpu is None or gpu.vram_mb < _LOW_VRAM_MB:
        return Tier.CPU
    if gpu.vram_mb >= _HIGH_VRAM_MB and info.ram_gb >= _HIGH_RAM_GB:
        return Tier.HIGH
    if gpu.vram_mb >= _MID_VRAM_MB:
        return Tier.MID
    return Tier.LOW


def parse_nvidia_smi(output: str) -> list[Gpu]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,driver_version[,compute_cap]` CSV."""
    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            vram_mb = int(float(parts[1]))
        except ValueError:
            continue
        compute = parts[3] if len(parts) > 3 and parts[3] not in ("", "[N/A]") else None
        gpus.append(Gpu(name=parts[0], vram_mb=vram_mb, driver=parts[2], compute_capability=compute))
    return gpus


def query_nvidia_gpus() -> list[Gpu]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    # Older drivers reject the compute_cap field, so retry without it.
    for fields in ("name,memory.total,driver_version,compute_cap", "name,memory.total,driver_version"):
        try:
            result = subprocess.run(
                [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        return parse_nvidia_smi(result.stdout)
    return []


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown CPU"


def _free_disk_gb(path: Path) -> float:
    # The data home may not exist yet; measure the nearest existing parent.
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def detect(data_home: Path) -> HardwareInfo:
    return HardwareInfo(
        cpu=_cpu_name(),
        cpu_cores=psutil.cpu_count(logical=False) or os.cpu_count() or 1,
        cpu_threads=psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        ram_gb=psutil.virtual_memory().total / 1024**3,
        free_disk_gb=_free_disk_gb(data_home),
        gpus=query_nvidia_gpus(),
    )
