"""Render a run into a standalone game-viewer page with its episode inlined.

    python scripts/render_versus_page.py runs/versus-<slug> [out.html]

This is the *match* view — two squads facing the canvas, a scoreboard over
each scored half, and a scrubbable timeline — as opposed to report.html,
which is the analysis view (charts, full traces, interviews).

The page is the same `codrawing/game/client/viewer.html` the live game
serves, with the replay embedded as JSON instead of fetched, and with the
document wrapper stripped so it can be published directly as a Claude
artifact. It needs no server and makes no network requests.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER = REPO_ROOT / "codrawing" / "game" / "client" / "viewer.html"


def slice_between(source: str, start_tag: str, end_tag: str) -> str:
    start = source.index(start_tag) + len(start_tag)
    return source[start : source.index(end_tag, start)]


def render(run_dir: Path, output_path: Path | None = None) -> Path:
    replay = json.loads((run_dir / "replay.json").read_text())
    frames = replay["frames"]
    if not frames:
        raise SystemExit(f"{run_dir} has no frames")
    teams = frames[0].get("teams") or []
    if output_path is None:
        output_path = run_dir / "game.html"

    viewer = VIEWER.read_text()
    style = slice_between(viewer, "<style>", "</style>")
    body = slice_between(viewer, "<body>", "</body>")
    # Distinct from report.html's title: the two pages cover the same match and
    # sit side by side in an artifact gallery.
    title = (
        " vs ".join(team["target"].title() for team in teams) + " Arena"
        if teams
        else frames[0]["target"].title() + " Arena"
    )

    # Only the frames are needed to draw the match; traces stay in report.html.
    payload = json.dumps({"frames": frames}, separators=(",", ":")).replace("</", "<\\/")
    data_tag = f'<script type="application/json" id="replay-data">{payload}</script>'
    # The data has to be parsed before the viewer script runs. Plain string
    # replace, not re.sub: the payload is full of backslash escapes.
    body = body.replace("<script>", data_tag + "\n<script>", 1)
    # A sandboxed page cannot hand its viewer a file, so the PNG export would
    # be a dead control here: drop the button and its handler, which is the
    # last statement in the viewer script.
    body = body.replace('<button id="download">PNG</button>', "")
    marker = "  // The standalone build drops this button"
    start = body.index(marker)
    body = body[:start] + body[body.index("</script>", start) :]

    # Served from a plain file server the page has no other charset hint, and
    # the viewer uses middle dots and arrows; without this Chrome sniffs GBK.
    page = f'<meta charset="utf-8">\n<title>{title}</title>\n<style>{style}</style>\n{body}'
    output_path.write_text(page)
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: render_versus_page.py RUN_DIR [OUTPUT_HTML]")
    render(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) == 3 else None)
