"""Serve a local home page listing every run, linking to each run's report.

    .venv/bin/python scripts/serve_runs.py [--port 8600]

Runs are plain directories under runs/ (config.json, replay.json,
results.json, trace-<slot>.jsonl, report.html); this server scans them at
request time — no database, refresh the page to pick up new runs. Missing
report.html files are rendered on demand from replay.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_replay_page  # noqa: E402

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>codrawing runs</title>
<style>
:root {
  --paper: #FAF7F0; --card: #FFFFFF; --grid: #E9E3D6; --line: #D8D2C4;
  --ink: #26221A; --muted: #7A7466; --accent: #B45309;
  --good: #15803D; --bad: #B91C1C;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #161512; --card: #201E19; --grid: #2E2B24; --line: #3A362D;
    --ink: #EAE6DC; --muted: #948D7D; --accent: #F59E0B;
    --good: #4ADE80; --bad: #F87171;
  }
}
body {
  background: var(--paper); color: var(--ink);
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  margin: 0; padding: 24px 16px 64px; font-variant-numeric: tabular-nums;
}
.wrap { max-width: 1160px; margin: 0 auto; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.14em; font-size: 12px; color: var(--muted); }
h1 { font-size: 22px; font-weight: 700; margin: 4px 0 20px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
a.card {
  display: grid; grid-template-columns: 104px 1fr; gap: 14px;
  background: var(--card); border: 1px solid var(--line); border-radius: 6px;
  padding: 14px; color: inherit; text-decoration: none;
}
a.card:hover { border-color: var(--muted); }
.thumb { width: 100px; height: 100px; background: #FFFFFF; border: 1px solid var(--grid); border-radius: 4px; }
.info { display: flex; flex-direction: column; gap: 4px; min-width: 0; font-size: 12.5px; }
.info .target { font-size: 15px; font-weight: 700; }
.info .score { font-size: 14px; }
.info .score strong { color: var(--accent); }
.info .pass { color: var(--good); font-weight: 700; }
.info .fail { color: var(--bad); font-weight: 700; }
.info .meta { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.empty { color: var(--muted); font-size: 14px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">codrawing</div>
  <h1>Runs</h1>
  <div class="cards">__CARDS__</div>
  __EMPTY__
</div>
</body>
</html>
"""

CARD = """<a class="card" href="/runs/{dirname}/report.html">
  <svg class="thumb" viewBox="0 0 {w} {h}" shape-rendering="crispEdges" aria-hidden="true">{pixels}</svg>
  <span class="info">
    <span class="target">{target}</span>
    <span class="score"><strong>{score}</strong> best &middot; <span class="{cls}">{verdict}</span></span>
    <span class="meta">{seats} agents &middot; {turns} turns{traces}</span>
    <span class="meta">{when}</span>
    <span class="meta">{dirname}</span>
  </span>
</a>"""


def thumb_pixels(canvas: list[str], width: int) -> str:
    parts = []
    for index, color in enumerate(canvas):
        if color == "#FFFFFF":
            continue
        parts.append(
            f'<rect x="{index % width}" y="{index // width}" width="1" height="1" fill="{color}"/>'
        )
    return "".join(parts)


def run_card(run_dir: Path) -> str | None:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None
    try:
        results = json.loads(results_path.read_text())
    except json.JSONDecodeError:
        return None
    canvas = results.get("final_canvas") or []
    width = 0
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            width = int(json.loads(config_path.read_text()).get("width", 0))
        except json.JSONDecodeError:
            pass
    if not width and canvas:
        width = int(math.isqrt(len(canvas)))
    height = len(canvas) // width if width else 0
    best = results.get("best_target_score")
    if best is None:
        best = max(results.get("scores") or [0])
    passed = bool(results.get("evaluation_passed"))
    trace_count = len(list(run_dir.glob("trace-*.jsonl")))
    when = datetime.fromtimestamp(results_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return CARD.format(
        dirname=run_dir.name,
        w=width or 1,
        h=height or 1,
        pixels=thumb_pixels(canvas, width) if width else "",
        target=results.get("target", "?"),
        score=f"{best * 100:.1f}%",
        cls="pass" if passed else "fail",
        verdict="PASS" if passed else "not passing",
        seats=len(results.get("accepted_pixels", [])),
        turns=results.get("turns", "?"),
        traces=" &middot; agent traces" if trace_count else "",
        when=when,
    )


def index_page() -> str:
    run_dirs = sorted(
        (d for d in RUNS_DIR.iterdir() if d.is_dir()) if RUNS_DIR.exists() else [],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    cards = [card for card in (run_card(d) for d in run_dirs) if card]
    empty = "" if cards else '<p class="empty">No runs yet. Start one with scripts/run_agent_episode.py.</p>'
    return PAGE.replace("__CARDS__", "".join(cards)).replace("__EMPTY__", empty)


class RunsHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            body = index_page().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # Render a missing report on demand so old runs get pages too.
        if self.path.endswith("/report.html"):
            target = REPO_ROOT / self.path.lstrip("/")
            replay = target.parent / "replay.json"
            if not target.exists() and replay.exists():
                try:
                    render_replay_page.render(target.parent)
                except Exception as error:
                    print(f"could not render {target}: {error!r}", flush=True)
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()
    handler = partial(RunsHandler, directory=str(REPO_ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"serving runs at http://127.0.0.1:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
