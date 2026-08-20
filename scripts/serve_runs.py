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
import render_versus_page  # noqa: E402

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
h2 { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); border-bottom: 1px solid var(--line); padding-bottom: 8px; margin: 26px 0 14px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
a.card {
  display: grid; grid-template-columns: 104px 1fr; gap: 14px;
  background: var(--card); border: 1px solid var(--line); border-radius: 6px;
  padding: 14px; color: inherit; text-decoration: none;
}
a.card:hover { border-color: var(--muted); }
/* A cell holds the run card plus, for versus runs, a link to the match view.
   The card is an anchor, so the second link has to be its sibling. */
.cell { display: flex; flex-direction: column; }
.cell > a.card { flex: 1; }
a.match {
  align-self: stretch; text-align: center; font-size: 11.5px; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--accent); text-decoration: none;
  background: var(--card); border: 1px solid var(--line); border-top: none;
  border-radius: 0 0 6px 6px; margin-top: -1px; padding: 8px;
}
a.match:hover { text-decoration: underline; }
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
  __SECTIONS__
  __EMPTY__
</div>
</body>
</html>
"""

CARD = """<div class="cell"><a class="card" href="/runs/{dirname}/report.html">
  <svg class="thumb" viewBox="0 0 {w} {h}" shape-rendering="crispEdges" aria-hidden="true">{pixels}</svg>
  <span class="info">
    <span class="target">{target}</span>
    <span class="score">{scoreline}</span>
    <span class="meta">{seats} agent{plural} &middot; {turns} turns{model}{traces}</span>
    <span class="meta">{when}</span>
    <span class="meta">{dirname}</span>
  </span>
</a>{extra}</div>"""


def thumb_pixels(canvas: list[str], width: int) -> str:
    parts = []
    for index, color in enumerate(canvas):
        if color == "#FFFFFF":
            continue
        parts.append(
            f'<rect x="{index % width}" y="{index // width}" width="1" height="1" fill="{color}"/>'
        )
    return "".join(parts)


def run_card(run_dir: Path) -> tuple[str, float, str] | None:
    """Return (environment key, mtime, card html) for a run directory."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None
    try:
        results = json.loads(results_path.read_text())
    except json.JSONDecodeError:
        return None
    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass
    canvas = results.get("final_canvas") or []
    width = int(config.get("width", 0))
    if not width and canvas:
        width = int(math.isqrt(len(canvas)))
    height = len(canvas) // width if width else 0
    teams = results.get("teams") or []
    if teams:
        # Versus run: two final scores and a winner, not one pass/fail score.
        finals = " vs ".join(
            f"<strong>{team['name']} {team.get('final_score', 0) * 100:.1f}%</strong>"
            for team in teams
        )
        verdict = (
            f'<span class="pass">{results.get("winner_name", "?")} wins</span>'
            if results.get("winner") is not None
            else '<span class="meta">tie</span>'
        )
        scoreline = f"{finals} &middot; {verdict}"
    else:
        best = results.get("best_target_score")
        if best is None:
            best = max(results.get("scores") or [0])
        passed = bool(results.get("evaluation_passed"))
        scoreline = (
            f"<strong>{best * 100:.1f}%</strong> best &middot; "
            f'<span class="{"pass" if passed else "fail"}">'
            f'{"PASS" if passed else "not passing"}</span>'
        )
    seats = len(results.get("accepted_pixels", []))
    turns = results.get("turns", "?")
    model = config.get("model")
    trace_count = len(list(run_dir.glob("trace-*.jsonl")))
    mtime = results_path.stat().st_mtime
    when = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    environment = config.get("environment") or f"{results.get('target', '?')} &middot; {turns} turns"
    card = CARD.format(
        dirname=run_dir.name,
        w=width or 1,
        h=height or 1,
        pixels=thumb_pixels(canvas, width) if width else "",
        target=results.get("target", "?"),
        scoreline=scoreline,
        seats=seats,
        plural="" if seats == 1 else "s",
        turns=turns,
        model=f" &middot; {model}" if model else "",
        traces=" &middot; agent traces" if trace_count else "",
        when=when,
        extra=(
            f'<a class="match" href="/runs/{run_dir.name}/game.html">watch the match &rarr;</a>'
            if teams
            else ""
        ),
    )
    return environment, mtime, card


def index_page() -> str:
    run_dirs = (d for d in RUNS_DIR.iterdir() if d.is_dir()) if RUNS_DIR.exists() else []
    groups: dict[str, list[tuple[float, str]]] = {}
    for run_dir in run_dirs:
        entry = run_card(run_dir)
        if entry:
            environment, mtime, card = entry
            groups.setdefault(environment, []).append((mtime, card))
    sections = []
    for environment, entries in sorted(
        groups.items(), key=lambda item: max(m for m, _ in item[1]), reverse=True
    ):
        cards = "".join(card for _, card in sorted(entries, reverse=True))
        sections.append(f'<h2>{environment}</h2><div class="cards">{cards}</div>')
    empty = "" if sections else '<p class="empty">No runs yet. Start one with codrawing-run.</p>'
    return PAGE.replace("__SECTIONS__", "".join(sections)).replace("__EMPTY__", empty)


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
        # Render either page on demand, and re-render it whenever the replay
        # has moved on: a running episode checkpoints every few turns, so a
        # page built from an earlier checkpoint is stale, not missing.
        for name, renderer in (
            ("/report.html", render_replay_page.render),
            ("/game.html", render_versus_page.render),
        ):
            if not self.path.endswith(name):
                continue
            target = REPO_ROOT / self.path.lstrip("/")
            replay = target.parent / "replay.json"
            if not replay.exists():
                break
            # The template can change too (a redesigned viewer), so compare
            # against the newest of the replay and the renderer's sources.
            sources = [replay.stat().st_mtime, Path(renderer.__code__.co_filename).stat().st_mtime]
            if hasattr(render_versus_page, 'VIEWER') and renderer is render_versus_page.render:
                sources.append(render_versus_page.VIEWER.stat().st_mtime)
            fresh = target.exists() and target.stat().st_mtime >= max(sources)
            if not fresh:
                try:
                    renderer(target.parent)
                except Exception as error:
                    print(f"could not render {target}: {error!r}", flush=True)
            break
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
