from __future__ import annotations

import binascii
import json
from pathlib import Path
import struct
import sys
from typing import Any
import zlib


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)


def render(replay_path: Path, output_path: Path, scale: int = 20) -> None:
    replay: dict[str, Any] = json.loads(replay_path.read_bytes())
    frames = replay.get("frames", [])
    if not frames:
        raise ValueError("replay contains no frames")
    frame = frames[-1]
    width, height = int(frame["width"]), int(frame["height"])
    colors = frame["canvas"]
    if len(colors) != width * height:
        raise ValueError("canvas length does not match its dimensions")

    rows = bytearray()
    for y in range(height):
        expanded_row = bytearray()
        for x in range(width):
            color = colors[y * width + x]
            rgb = bytes.fromhex(color.removeprefix("#"))
            if len(rgb) != 3:
                raise ValueError(f"invalid color: {color}")
            expanded_row.extend(rgb * scale)
        scanline = b"\x00" + bytes(expanded_row)
        rows.extend(scanline * scale)

    output_width, output_height = width * scale, height * scale
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", output_width, output_height, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
    png += png_chunk(b"IEND", b"")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: render_replay.py REPLAY OUTPUT_PNG [SCALE]")
    render(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else 20)
