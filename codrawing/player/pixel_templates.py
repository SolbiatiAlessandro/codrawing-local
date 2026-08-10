from __future__ import annotations

from collections.abc import Iterable


Pixel = tuple[int, int, str]


def _ring(cx: float, cy: float, r: float, color: str, points: int = 64) -> Iterable[Pixel]:
    import math

    for step in range(points):
        angle = 2 * math.pi * step / points
        yield round(cx + r * math.cos(angle)), round(cy + r * math.sin(angle)), color


def _ellipse(cx: float, cy: float, rx: float, ry: float, color: str) -> Iterable[Pixel]:
    left, right = int(cx - rx), int(cx + rx)
    top, bottom = int(cy - ry), int(cy + ry)
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                yield x, y, color


def _line(x1: int, y1: int, x2: int, y2: int, color: str) -> Iterable[Pixel]:
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    for step in range(steps + 1):
        yield round(x1 + (x2 - x1) * step / steps), round(y1 + (y2 - y1) * step / steps), color


def _put(canvas: dict[tuple[int, int], str], pixels: Iterable[Pixel], width: int, height: int) -> None:
    for x, y, color in pixels:
        if 0 <= x < width and 0 <= y < height:
            canvas[(x, y)] = color


def make_template(target: str, width: int, height: int) -> list[Pixel]:
    """Return a small demonstrator image; later writes intentionally add detail."""
    sx, sy = width / 24, height / 24
    canvas: dict[tuple[int, int], str] = {}

    def scale(pixels: Iterable[Pixel]) -> Iterable[Pixel]:
        for x, y, color in pixels:
            yield min(width - 1, int(x * sx)), min(height - 1, int(y * sy)), color

    if target.lower() == "light bulb":
        # Quick, Draw! prototypes are stroke sketches: an outline bulb scores
        # far above a filled one, and sun-like rays flip the class to "sun".
        _put(canvas, scale(_ring(12, 9, 6, "#F59E0B")), width, height)
        _put(canvas, scale(_line(9, 14, 9, 17, "#94A3B8")), width, height)
        _put(canvas, scale(_line(15, 14, 15, 17, "#94A3B8")), width, height)
        _put(canvas, scale(_line(9, 17, 15, 17, "#94A3B8")), width, height)
        _put(canvas, scale(_line(10, 19, 14, 19, "#475569")), width, height)
    elif target.lower() == "elephant":
        _put(canvas, scale(_ellipse(12, 12, 8, 6, "#8B95A5")), width, height)
        _put(canvas, scale(_ellipse(5, 10, 3, 4, "#758091")), width, height)
        _put(canvas, scale(_ellipse(19, 10, 3, 4, "#758091")), width, height)
        _put(canvas, scale(_line(12, 13, 12, 21, "#8B95A5")), width, height)
        _put(canvas, scale(_line(12, 21, 15, 21, "#8B95A5")), width, height)
        _put(canvas, scale([(9, 10, "#111827"), (15, 10, "#111827")]), width, height)
        _put(canvas, scale([(11, 9, "#FFFFFF"), (17, 9, "#FFFFFF")]), width, height)
    elif target.lower() == "dog":
        _put(canvas, scale(_ellipse(12, 13, 7, 6, "#C68642")), width, height)
        _put(canvas, scale(_ellipse(5, 10, 3, 6, "#6B3F24")), width, height)
        _put(canvas, scale(_ellipse(19, 10, 3, 6, "#6B3F24")), width, height)
        _put(canvas, scale(_ellipse(12, 16, 3, 2, "#E7B77A")), width, height)
        _put(canvas, scale([(9, 11, "#111827"), (15, 11, "#111827"), (12, 15, "#111827")]), width, height)
        _put(canvas, scale(_line(10, 18, 14, 18, "#8B1E3F")), width, height)
    else:
        _put(canvas, scale(_ellipse(12, 13, 7, 6, "#AAB4C4")), width, height)
        _put(canvas, scale(_line(6, 9, 7, 3, "#AAB4C4")), width, height)
        _put(canvas, scale(_line(7, 3, 11, 7, "#AAB4C4")), width, height)
        _put(canvas, scale(_line(18, 9, 17, 3, "#AAB4C4")), width, height)
        _put(canvas, scale(_line(17, 3, 13, 7, "#AAB4C4")), width, height)
        _put(canvas, scale([(9, 12, "#22C55E"), (15, 12, "#22C55E"), (12, 15, "#F472B6")]), width, height)
        _put(canvas, scale(_line(12, 16, 10, 18, "#111827")), width, height)
        _put(canvas, scale(_line(12, 16, 14, 18, "#111827")), width, height)
        for y in (15, 17):
            _put(canvas, scale(_line(3, y, 9, y + 1, "#475569")), width, height)
            _put(canvas, scale(_line(15, y + 1, 21, y, "#475569")), width, height)

    return [(x, y, color) for (x, y), color in sorted(canvas.items(), key=lambda item: (item[0][1], item[0][0]))]
