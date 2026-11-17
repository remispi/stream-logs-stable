import hashlib
import http.server
import threading
import zipfile

import pytest

from stream_log.engine import first_run
from stream_log.engine.config import Paths
from stream_log.engine.hardware import Gpu, HardwareInfo, Tier


def _info(driver: str | None = "591.44", vram_mb: int = 16303) -> HardwareInfo:
    gpus = [Gpu("RTX", vram_mb, driver)] if driver else []
    return HardwareInfo("cpu", 12, 24, 31.6, 500.0, gpus)


def test_gpu_wheels_from_the_lock_file():
    wheels = first_run.wheels_from_lock(first_run.UV_LOCK.read_text(encoding="utf-8"))
    names = {w.name.split("-")[0] for w in wheels}
    assert names == {"nvidia_cublas_cu12", "nvidia_cuda_nvrtc_cu12", "nvidia_cudnn_cu12"}
    assert all(w.url.startswith("https://") and len(w.sha256) == 64 and w.size > 0
               for w in wheels)


def test_llama_variant_follows_driver_and_tier():
    assert first_run.llama_variant(_info("591.44"), Tier.HIGH) == "cuda-13.4"
    assert first_run.llama_variant(_info("566.36"), Tier.MID) == "cuda-12.4"
    assert first_run.llama_variant(_info(None), Tier.CPU) == "cpu"


def test_plan_on_an_empty_home(tmp_path, monkeypatch):
    monkeypatch.delenv("STREAM_LOG_LLAMA_SERVER", raising=False)
    monkeypatch.setattr(first_run.shutil, "which", lambda _: None)
    monkeypatch.setattr(first_run, "cuda_ready", lambda _: False)
    paths = Paths(tmp_path)
    todo = first_run.plan(paths, _info(), Tier.HIGH)
    assert len(todo.cuda) == 3 and todo.llama_variant == "cuda-13.4"
    assert todo.models and todo.models_gb > 1 and todo.download_gb > todo.models_gb
    cpu = first_run.plan(paths, _info(None), Tier.CPU)
    assert not cpu.cuda and cpu.llama_variant == "cpu"


def test_plan_skips_what_is_there(tmp_path, monkeypatch):
    exe = tmp_path / "runtimes" / "llama.cpp" / "b100" / "llama-server.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    (tmp_path / first_run.CUDA_DIR).mkdir(parents=True)
    (tmp_path / first_run.CUDA_DIR / first_run.CUDA_MARKER).write_text("[]")
    todo = first_run.plan(Paths(tmp_path), _info(), Tier.HIGH)
    assert not todo.cuda and not todo.llama


@pytest.fixture
def served(tmp_path):
    """An HTTP server for tmp_path/www that honours Range requests."""
    root = tmp_path / "www"
    root.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *_):
            pass

        def do_GET(self):
            data = (root / self.path.lstrip("/")).read_bytes()
            start = 0
            if "Range" in self.headers:
                start = int(self.headers["Range"].split("=")[1].rstrip("-"))
                self.send_response(206)
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def test_fetch_resumes_and_checks_the_hash(served, tmp_path):
    root, url = served
    data = bytes(range(256)) * 5000
    (root / "pack.zip").write_bytes(data)
    item = first_run.Download("pack.zip", url + "pack.zip", hashlib.sha256(data).hexdigest(),
                              len(data))
    out = tmp_path / "dl"
    out.mkdir()
    (out / "pack.zip.part").write_bytes(data[:1000])  # an interrupted download
    seen = []
    target = first_run.fetch(item, out, lambda _l, done, total: seen.append((done, total)))
    assert target.read_bytes() == data and seen[-1] == (len(data), len(data))
    bad = first_run.Download("pack.zip", url + "pack.zip", "0" * 64, len(data))
    with pytest.raises(RuntimeError, match="checksum"):
        first_run.fetch(bad, tmp_path / "dl2", lambda *_: None)


def test_install_cuda_makes_nvidia_importable(served, tmp_path, monkeypatch):
    root, url = served
    wheel = root / "nvidia_fake-1-py3-none-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w") as zipped:
        zipped.writestr("nvidia/fakepkg/bin/fake.dll", b"dll")
    data = wheel.read_bytes()
    item = first_run.Download(wheel.name, url + wheel.name, hashlib.sha256(data).hexdigest(),
                              len(data))
    paths = Paths(tmp_path / "home")
    target = first_run.install_cuda(paths, [item], lambda *_: None)
    assert (target / "nvidia" / "fakepkg" / "bin" / "fake.dll").exists()
    assert first_run.cuda_ready(paths)
    monkeypatch.setattr(first_run.sys, "path", list(first_run.sys.path))
    first_run.add_cuda_to_path(paths.home)
    assert str(target) in first_run.sys.path
