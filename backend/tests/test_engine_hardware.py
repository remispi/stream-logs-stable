import pytest

from stream_log.engine.hardware import Gpu, HardwareInfo, Tier, choose_tier, parse_nvidia_smi


def _info(vram_mb: int | None, ram_gb: float = 32.0) -> HardwareInfo:
    gpus = [Gpu("Test GPU", vram_mb, "1.0")] if vram_mb is not None else []
    return HardwareInfo(
        cpu="Test CPU", cpu_cores=8, cpu_threads=16, ram_gb=ram_gb, free_disk_gb=500, gpus=gpus
    )


def test_parse_nvidia_smi_with_compute_capability():
    gpus = parse_nvidia_smi("NVIDIA Test GPU, 16303, 555.55, 8.0\n")
    assert gpus == [Gpu("NVIDIA Test GPU", 16303, "555.55", "8.0")]


def test_parse_nvidia_smi_without_compute_capability():
    gpus = parse_nvidia_smi("NVIDIA Test GPU, 12288, 555.55\n")
    assert gpus == [Gpu("NVIDIA Test GPU", 12288, "555.55", None)]


def test_parse_nvidia_smi_skips_malformed_lines():
    output = "garbage\nNVIDIA A, not-a-number, 1.0\nNVIDIA B, 8188, 1.0, [N/A]\n"
    assert parse_nvidia_smi(output) == [Gpu("NVIDIA B", 8188, "1.0", None)]


@pytest.mark.parametrize(
    ("vram_mb", "ram_gb", "expected"),
    [
        (16303, 31.1, Tier.HIGH),  # 16 GB GPU + 32 GB RAM, of which Windows reports ~31 GB
        (16303, 16.0, Tier.MID),  # big GPU, not enough RAM for MoE expert offload
        (12282, 32.0, Tier.MID),
        (8188, 32.0, Tier.LOW),
        (6144, 32.0, Tier.CPU),
        (None, 32.0, Tier.CPU),
    ],
)
def test_choose_tier(vram_mb, ram_gb, expected):
    assert choose_tier(_info(vram_mb, ram_gb)) == expected


def test_best_gpu_picks_most_vram():
    info = HardwareInfo(
        cpu="cpu",
        cpu_cores=8,
        cpu_threads=16,
        ram_gb=32,
        free_disk_gb=100,
        gpus=[Gpu("small", 4096, "1"), Gpu("big", 16303, "1")],
    )
    assert info.best_gpu.name == "big"
