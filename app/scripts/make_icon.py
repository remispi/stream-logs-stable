"""Draw the app icon (an amber tile with three list lines) into electron/icon.ico.

Plain Python, no imaging library: PNG images written by hand, packed into one .ico.
    python app/scripts/make_icon.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

AMBER = (240, 166, 58, 255)
INK = (26, 20, 8, 255)
CLEAR = (0, 0, 0, 0)


def draw(size: int) -> list[list[tuple[int, int, int, int]]]:
    s = size / 256
    radius = 48 * s
    px = [[CLEAR] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            cx = min(max(x + 0.5, radius), size - radius)
            cy = min(max(y + 0.5, radius), size - radius)
            if (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= radius ** 2:
                px[y][x] = AMBER
    # Three rows: a short "time" bar, then a longer "text" bar.
    for row, length in ((0, 120), (1, 88), (2, 104)):
        top = int((62 + row * 50) * s)
        height = max(1, int(26 * s))
        for y in range(top, top + height):
            for x in range(int(48 * s), int(84 * s)):
                px[y][x] = INK
            for x in range(int(100 * s), int((100 + length) * s)):
                px[y][x] = INK
    return px


def png(px: list[list[tuple[int, int, int, int]]]) -> bytes:
    size = len(px)
    raw = b"".join(b"\x00" + b"".join(bytes(p) for p in row) for row in px)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def main() -> None:
    sizes = [256, 64, 48, 32, 16]
    images = [png(draw(size)) for size in sizes]
    out = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    for size, data in zip(sizes, images, strict=True):
        out += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    out += b"".join(images)
    target = Path(__file__).resolve().parents[1] / "electron" / "icon.ico"
    target.write_bytes(out)
    (target.with_name("icon.png")).write_bytes(images[0])
    print(f"wrote {target} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
