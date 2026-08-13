"""Summarize an adversarial episode: who attacked, who talked, who collided.

    python scripts/versus_summary.py runs/versus-<slug>

The score alone says little in this environment — the research signal is in
the behavior. This prints the things worth reading a transcript for, with
the turn numbers to jump to, and works on a checkpointed (still running)
run as well as a finished one.

Turn numbering: frames are stamped with the turn *after* resolution, so a
frame's writes were chosen on the agents' turn N (1-indexed as the agents
see it) and the frame is stamped N. Both are the same number here.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import sys


def load(run_dir: Path) -> tuple[dict, dict]:
    replay = json.loads((run_dir / "replay.json").read_text())
    results = replay.get("results") or json.loads((run_dir / "results.json").read_text())
    return replay, results


def attempted_pixels(run_dir: Path, seats: int) -> dict[tuple[int, int], tuple[int, int]]:
    """(turn, slot) -> the pixel that seat submitted, read from its trace."""
    attempts: dict[tuple[int, int], tuple[int, int]] = {}
    for slot in range(seats):
        trace = run_dir / f"trace-{slot}.jsonl"
        if not trace.exists():
            continue
        for line in trace.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("phase") != "turn":
                continue
            for event in record.get("events", []):
                if event.get("type") == "tool_use" and event.get("name", "").endswith("paint_pixel"):
                    try:
                        payload = json.loads(event["input"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    attempts[(record["turn"], slot)] = (payload["x"], payload["y"])
    return attempts


def summarize(run_dir: Path) -> None:
    replay, results = load(run_dir)
    frames = replay["frames"]
    if not frames:
        raise SystemExit("no frames yet")
    width = frames[0]["width"]
    teams = frames[0].get("teams") or []
    if not teams:
        raise SystemExit("not a versus run")
    team_of = {slot: index for index, team in enumerate(teams) for slot in team["slots"]}

    def region_at(x: int) -> int:
        for index, team in enumerate(teams):
            box = team["region"]
            if box["x"] <= x < box["x"] + box["width"]:
                return index
        return -1

    print(f"# {run_dir.name}")
    print(f"{frames[0]['width']}x{frames[0]['height']} canvas, {len(frames)} frames "
          f"of {frames[0]['max_turns']} turns, {len(frames[0]['player_names'])} agents")
    for index, team in enumerate(results.get("teams", teams)):
        final = team.get("final_score")
        best = team.get("best_score")
        line = f"  {team['name']:8s} {team['target']:12s}"
        if final is not None:
            line += f" final {final:.3f}  best {best:.3f}"
        print(line + f"  slots {team['slots']}")
    if results.get("winner_name"):
        print(f"  WINNER: {results['winner_name']}")

    # Attacks: writes that landed inside the other team's scored region.
    attacks: list[tuple[int, int, int, int, str]] = []
    for previous, frame in zip(frames, frames[1:]):
        for i, (color, owner) in enumerate(zip(frame["canvas"], frame["owners"])):
            if previous["canvas"][i] == color or owner < 0:
                continue
            region = region_at(i % width)
            if region >= 0 and team_of[owner] != region:
                attacks.append((frame["turn"], owner, i % width, i // width,
                                "erase" if color == "#FFFFFF" else "paint"))
    made = collections.Counter(team_of[a[1]] for a in attacks)
    total_writes = sum(len(f.get("accepted_slots", [])) for f in frames)
    print(f"\n## Attacks: {len(attacks)} of {total_writes} accepted writes")
    for index, team in enumerate(teams):
        print(f"  {team['name']} made {made.get(index, 0)}")
    for turn, owner, x, y, kind in attacks:
        print(f"    T{turn} agent{owner} ({teams[team_of[owner]]['name']}) {kind} ({x},{y})")

    # Collisions: agents picking the same pixel, every write to it dropped.
    # A frame records only the turn's collided slots, so two separate
    # same-team collisions look identical to one cross-team collision. The
    # seat traces carry the coordinates each agent actually submitted, so
    # group by pixel and only call a collision cross-team when it truly is.
    attempts = attempted_pixels(run_dir, len(frames[0]["player_names"]))
    collisions = [(f["turn"], f["collision_slots"]) for f in frames if f.get("collision_slots")]
    dropped = sum(len(slots) for _, slots in collisions)
    print(f"\n## Collisions: {dropped} writes dropped over {len(collisions)} turns")
    for turn, slots in collisions:
        groups: dict[tuple[int, int] | None, list[int]] = collections.defaultdict(list)
        for slot in slots:
            groups[attempts.get((turn - 1, slot))].append(slot)
        for pixel, members in groups.items():
            sides = {team_of[s] for s in members}
            kind = "cross-team" if len(sides) > 1 else f"{teams[team_of[members[0]]]['name']}"
            where = f"({pixel[0]},{pixel[1]})" if pixel else "(pixel unknown: no trace)"
            print(f"    T{turn} {where} agents {members} [{kind}]")

    # Talk: the public channel is the one agents keep ignoring.
    messages = [m for f in frames for m in f.get("messages", [])]
    public = [m for m in messages if m.get("public")]
    print(f"\n## Messages: {len(messages)} total, {len(public)} public broadcasts")
    for index, team in enumerate(teams):
        mine = [m for m in messages if team_of[m["slot"]] == index]
        turns_spoken = {m["turn"] for m in mine}
        silent = [f["turn"] for f in frames if f["turn"] not in turns_spoken]
        print(f"  {team['name']}: {len(mine)} messages, silent on {len(silent)} turns"
              + (f" (T{silent[0]}-T{silent[-1]})" if silent else ""))
    for m in public:
        print(f"    T{m['turn']} agent{m['slot']} PUBLIC: {m['text'][:200]}")

    completed = results.get("completed_slots") or frames[-1].get("completed_slots") or []
    if completed:
        print(f"\n## complete() votes: agents {completed}")

    # Where the lead changed hands.
    print("\n## Lead changes")
    previous_leader = None
    for frame in frames:
        scores = [(f or {}).get("target_score", 0) for f in frame.get("team_feedback", [])]
        if len(scores) != 2:
            continue
        leader = 0 if scores[0] > scores[1] else 1 if scores[1] > scores[0] else None
        if leader is not None and leader != previous_leader:
            print(f"    T{frame['turn']}: {teams[leader]['name']} takes the lead "
                  f"({scores[0]:.2f} vs {scores[1]:.2f})")
            previous_leader = leader


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: versus_summary.py RUN_DIR")
    summarize(Path(sys.argv[1]))


