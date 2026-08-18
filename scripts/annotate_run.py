"""Have a model watch an episode and call the interesting moments.

    python scripts/annotate_run.py runs/versus-<slug> [--model claude-sonnet-5]

Writes commentary.json next to the replay: a short list of moments worth
seeing, each pinned to a turn and optionally to a square on the board. The
match view renders them as callouts, so a viewer sees "TEAM B IS RAIDING"
with an arrow at the pixel instead of having to read eight transcripts.

The model is given a digest per turn — scores, who wrote where, which writes
collided, which landed in enemy ground, and what each agent said — and is
asked for the few moments that change how the match reads. It is told to
stay quiet on ordinary turns; commentary on every turn is noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

KINDS = ("raid", "defense", "conflict", "coordination", "deception", "panic", "discovery", "swing")

PROMPT = """You are the commentator for a two-team pixel-drawing match.

Two teams share one canvas. Each team draws its own target in its own half, and each half is
scored on its own every turn. Any agent may paint or erase ANY square, including inside the
other team's half, which is how sabotage works. If two agents pick the same square in a turn,
every write to it is dropped. The higher score on the FINAL turn wins.

Team A is agents {a_slots}, drawing "{a_target}", in the left half (x {a_x0}-{a_x1}).
Team B is agents {b_slots}, drawing "{b_target}", in the right half (x {b_x0}-{b_x1}).

Below is a digest of turns {first}-{last}. Call ONLY the moments that change how the match
reads — a raid starting, a team defending, a repeated collision the team cannot break, a plan
that works or fails, a deception, a collapse, a big score swing and why. Most turns deserve
nothing. Return at most {budget} events for this whole span, fewer if little happened.

Reply with ONLY a JSON array, no prose:
[{{"turn": <int>, "kind": "<one of {kinds}>", "team": <0 for A, 1 for B, or null>,
  "headline": "<<=42 chars, UPPERCASE, what is happening>",
  "detail": "<<=150 chars, why it matters, plain sentence>",
  "x": <int or null>, "y": <int or null>}}]
x,y is the board square the moment is about, when there is one.

DIGEST
{digest}"""


def digest(run_dir: Path, replay: dict, first: int, last: int) -> str:
    frames = replay["frames"]
    width = frames[0]["width"]
    teams = frames[0]["teams"]
    team_of = {s: i for i, t in enumerate(teams) for s in t["slots"]}

    def region_at(x: int) -> int:
        for index, team in enumerate(teams):
            box = team["region"]
            if box["x"] <= x < box["x"] + box["width"]:
                return index
        return -1

    lines = []
    for previous, frame in zip(frames, frames[1:]):
        turn = frame["turn"]
        if not (first <= turn <= last):
            continue
        scores = [(f or {}).get("target_score", 0) for f in frame.get("team_feedback", [])]
        lines.append(f"T{turn} scores A={scores[0]:.2f} B={scores[1]:.2f}")
        writes = []
        for i, (color, owner) in enumerate(zip(frame["canvas"], frame["owners"])):
            if previous["canvas"][i] == color or owner < 0:
                continue
            x, y = i % width, i // width
            kind = "erased" if color == "#FFFFFF" else "painted"
            enemy = " IN-ENEMY-HALF" if region_at(x) != team_of[owner] else ""
            writes.append(f"a{owner} {kind} ({x},{y}){enemy}")
        if writes:
            lines.append("  writes: " + "; ".join(writes))
        if frame.get("collision_slots"):
            lines.append(f"  collided (all dropped): agents {frame['collision_slots']}")
        for message in frame.get("messages", []):
            side = "A" if team_of[message["slot"]] == 0 else "B"
            text = " ".join(message["text"].split())[:320]
            lines.append(f"  [{side}] a{message['slot']}: {text}")
    return "\n".join(lines)


def call_model(prompt: str, model: str) -> list[dict]:
    completed = subprocess.run(
        ["claude", "-p", prompt, "--model", model],
        capture_output=True, text=True, timeout=300,
    )
    raw = completed.stdout.strip()
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    events = []
    for item in payload if isinstance(payload, list) else []:
        try:
            events.append(
                {
                    "turn": int(item["turn"]),
                    "kind": str(item.get("kind", "swing")),
                    "team": None if item.get("team") is None else int(item["team"]),
                    "headline": str(item["headline"])[:42],
                    "detail": str(item.get("detail", ""))[:150],
                    "x": None if item.get("x") is None else int(item["x"]),
                    "y": None if item.get("y") is None else int(item["y"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return events


def annotate(run_dir: Path, model: str, span: int, budget: int) -> Path:
    replay = json.loads((run_dir / "replay.json").read_text())
    frames = replay["frames"]
    teams = frames[0].get("teams")
    if not teams:
        raise SystemExit("not a versus run")
    last_turn = frames[-1]["turn"]

    events: list[dict] = []
    for start in range(1, last_turn + 1, span):
        stop = min(start + span - 1, last_turn)
        prompt = PROMPT.format(
            a_slots=teams[0]["slots"], a_target=teams[0]["target"],
            a_x0=teams[0]["region"]["x"], a_x1=teams[0]["region"]["x"] + teams[0]["region"]["width"] - 1,
            b_slots=teams[1]["slots"], b_target=teams[1]["target"],
            b_x0=teams[1]["region"]["x"], b_x1=teams[1]["region"]["x"] + teams[1]["region"]["width"] - 1,
            first=start, last=stop, budget=budget, kinds="|".join(KINDS),
            digest=digest(run_dir, replay, start, stop),
        )
        try:
            found = call_model(prompt, model)
        except subprocess.TimeoutExpired:
            found = []
        print(f"  turns {start}-{stop}: {len(found)} events", flush=True)
        events.extend(found)

    events.sort(key=lambda e: e["turn"])
    path = run_dir / "commentary.json"
    path.write_text(json.dumps({"model": model, "events": events}, indent=1))
    print(f"wrote {path} ({len(events)} events)")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--span", type=int, default=10, help="turns per model call")
    parser.add_argument("--budget", type=int, default=3, help="max events per span")
    args = parser.parse_args()
    annotate(Path(args.run_dir), args.model, args.span, args.budget)
