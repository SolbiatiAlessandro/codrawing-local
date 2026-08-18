"""The `codrawing-run` command: start a local episode with agent seats.

    codrawing-run --target "light bulb" --turns 20
    codrawing-run --target cat --rounds 2 --turns-per-round 10 \\
        --model claude-sonnet-5 --policy "outline first, then fill gaps"

Each seat is a persistent Claude Code session with its own workspace and a
per-turn trace file (activity, token usage, timing). The policy prompt can
be given inline (--policy) or as a file (--policy-file); the fixed game
prompt stays the same either way. Artifacts land in
runs/agent-<slug>-<timestamp>/, including a self-contained report.html.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import secrets
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]


async def wait_for_server(port: int, timeout: float = 30.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2):
                return
        except Exception:
            await asyncio.sleep(0.5)
    raise RuntimeError("game server did not become healthy")


async def run(
    seats: int,
    turns: int,
    turns_per_round: int | None,
    target: str,
    model: str,
    port: int,
    policy: str | None,
    scorer: str = "quickdraw",
    teams: list[dict] | None = None,
    width: int = 24,
    height: int = 24,
    slug: str | None = None,
    team_models: list[str] | None = None,
) -> Path:
    """team_models gives each team its own model, so a match can pit one model
    against another with everything else held equal."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Parallel launches can share a second; a short suffix keeps dirs unique.
    name = slug or f"agent-{target.replace(' ', '-')}"
    run_dir = REPO_ROOT / "runs" / f"{name}-{stamp}-{secrets.token_hex(2)}"
    run_dir.mkdir(parents=True)
    policy_path: Path | None = None
    if policy is not None:
        # Keep the policy with the run artifacts for provenance.
        policy_path = run_dir / "policy.md"
        policy_path.write_text(policy)
    tokens = [secrets.token_hex(8) for _ in range(seats)]
    config = {
        "tokens": tokens,
        # Versus seats are named by slot, matching what the prompt calls them
        # ("you are agent 3") and what agents call each other on the board.
        "players": [
            {"name": f"Agent {index if teams else index + 1}"} for index in range(seats)
        ],
        "width": width,
        "height": height,
        "max_turns": turns,
        "targets": [target],
        "player_connect_timeout_seconds": 60,
        "action_timeout_seconds": 300,
        "model": model,
        "team_models": team_models,
        "scorer": scorer,
        "environment": f"{target} · {turns} turns"
        + (f" · {scorer}" if scorer != "quickdraw" else ""),
    }
    if teams:
        config["teams"] = teams
    if turns_per_round:
        config["turns_per_round"] = turns_per_round
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    server_env = os.environ | {
        "PYTHONPATH": str(REPO_ROOT),
        "COGAME_CONFIG_URI": str(run_dir / "config.json"),
        "COGAME_RESULTS_URI": str(run_dir / "results.json"),
        "COGAME_SAVE_REPLAY_URI": str(run_dir / "replay.json"),
        "COGAME_HOST": "127.0.0.1",
        "COGAME_PORT": str(port),
        "CODRAWING_QUICKDRAW_MODEL": str(
            REPO_ROOT / "codrawing" / "game" / "models" / "quickdraw_prototypes.json"
        ),
        "CODRAWING_SCORER": scorer,
        # CLIP scores by softmax over a label set, so every target in the
        # episode must be in it — including the opponent's.
        "CODRAWING_SCORER_LABELS": ",".join(
            [team["target"] for team in teams] if teams else [target]
        ),
    }
    server_log = open(run_dir / "server.log", "w")
    server = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "codrawing.game.server",
        env=server_env,
        stdout=server_log,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        await wait_for_server(port)
        print(f"live viewer: http://127.0.0.1:{port}/client/global", flush=True)
        def seat_model(slot: int) -> str:
            if not (teams and team_models):
                return model
            for index, team in enumerate(teams):
                if slot in team["slots"]:
                    return team_models[index]
            return model

        players, logs = [], []
        for slot in range(seats):
            workspace = run_dir / f"workspace-{slot}"
            workspace.mkdir()
            player_env = os.environ | {
                "PYTHONPATH": str(REPO_ROOT),
                "CODRAWING_PLAYER_WS_URL": (
                    f"ws://127.0.0.1:{port}/player?slot={slot}&token={tokens[slot]}"
                ),
                "AGENT_MODEL": seat_model(slot),
                "AGENT_WORKSPACE": str(workspace),
                "AGENT_TRACE_FILE": str(run_dir / f"trace-{slot}.jsonl"),
            }
            if policy_path is not None:
                player_env["AGENT_POLICY_FILE"] = str(policy_path)
            log = open(run_dir / f"agent-{slot}.log", "w")
            logs.append(log)
            players.append(
                await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "codrawing.player.agent_player",
                    env=player_env,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
            )
        await server.wait()
        # Seats run a post-episode interview after the final frame; give them time.
        for player in players:
            try:
                await asyncio.wait_for(player.wait(), timeout=180)
            except TimeoutError:
                player.kill()
        for log in logs:
            log.close()
    finally:
        if server.returncode is None:
            server.kill()
        server_log.close()

    results = json.loads((run_dir / "results.json").read_text())
    print(f"run directory: {run_dir}")
    print(f"accepted pixels by seat: {results['accepted_pixels']}")
    if results.get("teams"):
        for team in results["teams"]:
            label = ""
            if team_models:
                label = f" [{team_models[results['teams'].index(team)]}]"
            print(
                f"{team['name']}{label} ({team['target']}): final {team['final_score']:.4f} "
                f"· best {team['best_score']:.4f} · {team['accepted_pixels']} pixels"
            )
        print(f"WINNER: {results.get('winner_name')}")
    feedback = results.get("final_image_model_feedback")
    if feedback:
        print(f"best score: {results.get('best_target_score'):.4f}")
        print(f"round scores: {results.get('round_scores')}")
    report = await asyncio.create_subprocess_exec(
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_replay_page.py"),
        str(run_dir),
    )
    await report.wait()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="codrawing-run",
        description="Start a local codrawing episode where every seat is a Claude Code agent.",
    )
    parser.add_argument("--target", default="light bulb", help="target the team must draw")
    parser.add_argument("--seats", type=int, default=4, help="number of agents (default 4)")
    parser.add_argument("--turns", type=int, default=10, help="total turns (ignored if --rounds is set)")
    parser.add_argument("--rounds", type=int, default=None, help="number of rounds; total turns = rounds * turns-per-round")
    parser.add_argument("--turns-per-round", type=int, default=None, help="turns per round (default 10 when --rounds is set)")
    parser.add_argument("--model", default="claude-sonnet-5", help="model for every agent seat")
    parser.add_argument("--policy", default=None, help="inline policy prompt shared by all seats")
    parser.add_argument("--policy-file", default=None, help="path to a policy prompt file shared by all seats")
    parser.add_argument("--port", type=int, default=8331)
    parser.add_argument(
        "--scorer",
        choices=["quickdraw", "mobileclip", "judge", "both"],
        default="quickdraw",
        help="canvas grader: quickdraw prototypes (default), MobileCLIP2-S0 zero-shot "
        "(open vocabulary; needs the mobileclip extra), judge (a vision LLM scores "
        "the canvas 0-100 with a rubric that rewards correct scene context), or both "
        "(mean of judge and CLIP, which halves the judge's sampling noise)",
    )
    args = parser.parse_args()

    if args.policy and args.policy_file:
        parser.error("use either --policy or --policy-file, not both")
    policy = args.policy
    if args.policy_file:
        policy = Path(args.policy_file).read_text()

    turns_per_round = args.turns_per_round
    turns = args.turns
    if args.rounds:
        turns_per_round = turns_per_round or 10
        turns = args.rounds * turns_per_round

    asyncio.run(
        run(args.seats, turns, turns_per_round, args.target, args.model, args.port, policy, args.scorer)
    )


# Targets with measured behaviour in the judge sweep, so a random pair is
# drawn from things both scorers are known to recognize.
TARGET_POOL = (
    "pineapple", "zebra", "butterfly", "bee", "banana", "strawberry",
    "ice cream", "traffic light", "candle", "snail",
)


def build_teams(left: str, right: str, per_team: int, half: int) -> list[dict]:
    return [
        {
            "name": f"Team {'AB'[index]}",
            "target": target,
            "region": {"x": index * half, "y": 0, "width": half, "height": half},
            "slots": list(range(index * per_team, (index + 1) * per_team)),
        }
        for index, target in enumerate((left, right))
    ]


def match_main() -> None:
    """Two rounds with the targets swapped, so neither team owns the easier one."""
    parser = argparse.ArgumentParser(
        prog="codrawing-match",
        description="A fair adversarial match: two rounds of the same two targets on a fresh "
        "canvas, swapped between the teams in round two. Each team therefore draws both "
        "images, so a target being easier than the other cancels out, and the match is "
        "decided on each team's total across the two rounds.",
    )
    parser.add_argument("--targets", default=None, help="comma-separated pair; default picks two at random")
    parser.add_argument("--turns", type=int, default=50, help="turns per round")
    parser.add_argument("--seats-per-team", type=int, default=4)
    parser.add_argument("--half-size", type=int, default=32)
    parser.add_argument("--model", default="claude-sonnet-5", help="model for both teams unless overridden")
    parser.add_argument("--model-a", default=None, help="model for Team A (slots 0..n-1)")
    parser.add_argument("--model-b", default=None, help="model for Team B")
    parser.add_argument("--scorer", choices=["both", "judge", "mobileclip", "quickdraw"], default="both")
    parser.add_argument("--policy-file", default=None)
    parser.add_argument("--port", type=int, default=8360)
    parser.add_argument("--seed", type=int, default=None, help="seed for the random target pair")
    args = parser.parse_args()

    if args.targets:
        pair = [t.strip() for t in args.targets.split(",") if t.strip()]
        if len(pair) != 2:
            parser.error("--targets needs exactly two comma-separated names")
    else:
        pair = random.Random(args.seed).sample(TARGET_POOL, 2)
    policy = Path(args.policy_file).read_text() if args.policy_file else None
    # Teams keep their models across the swap: the targets move, the models do not,
    # so each model draws both images and target difficulty cancels out.
    team_models = [args.model_a or args.model, args.model_b or args.model]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    match_id = f"match-{pair[0].replace(' ', '-')}-{pair[1].replace(' ', '-')}-{stamp}"
    print(f"MATCH {match_id}")
    print(f"  Team A: {team_models[0]}   vs   Team B: {team_models[1]}")
    print(f"  round 1: Team A {pair[0]} vs Team B {pair[1]}")
    print(f"  round 2: Team A {pair[1]} vs Team B {pair[0]} (swapped)\n")

    rounds = []
    for number, (left, right) in enumerate(((pair[0], pair[1]), (pair[1], pair[0])), start=1):
        print(f"=== round {number}: {left} (left) vs {right} (right)", flush=True)
        run_dir = asyncio.run(
            run(
                seats=args.seats_per_team * 2,
                turns=args.turns,
                turns_per_round=None,
                target=f"{left} vs {right}",
                model=args.model,
                port=args.port + number,
                policy=policy,
                scorer=args.scorer,
                teams=build_teams(left, right, args.seats_per_team, args.half_size),
                width=args.half_size * 2,
                height=args.half_size,
                slug=f"{match_id}-r{number}",
                team_models=team_models,
            )
        )
        results = json.loads((run_dir / "results.json").read_text())
        rounds.append({"round": number, "dir": run_dir.name, "left": left, "right": right,
                       "scores": results["final_scores"]})

    # Team A is always slots 0-3 on the left; only the target it draws swaps.
    totals = [sum(r["scores"][team] for r in rounds) for team in (0, 1)]
    winner = None if totals[0] == totals[1] else (0 if totals[0] > totals[1] else 1)
    match = {
        "match": match_id,
        "targets": pair,
        "team_models": team_models,
        "turns_per_round": args.turns,
        "rounds": rounds,
        "totals": totals,
        "winner": winner,
        "winner_name": "tie" if winner is None else f"Team {'AB'[winner]}",
    }
    path = REPO_ROOT / "runs" / f"{match_id}.json"
    path.write_text(json.dumps(match, indent=2))

    print("\n=== MATCH RESULT")
    for entry in rounds:
        print(f"  round {entry['round']}: Team A ({entry['left']}) {entry['scores'][0]:.3f}"
              f"  ·  Team B ({entry['right']}) {entry['scores'][1]:.3f}")
    print(f"  totals: Team A [{team_models[0]}] {totals[0]:.3f}"
          f"  ·  Team B [{team_models[1]}] {totals[1]:.3f}")
    print(f"  WINNER: {match['winner_name']}")
    print(f"  {path}")


def versus_main() -> None:
    parser = argparse.ArgumentParser(
        prog="codrawing-versus",
        description="Two teams of Claude Code agents fight over one shared canvas. Each team "
        "draws its own target in its own half; the half is cropped and scored on its own every "
        "turn. Any agent may paint anywhere, so the other team's half can be sabotaged. The "
        "team with the higher score on the FINAL turn wins.",
    )
    parser.add_argument("--left", default="pineapple", help="target for the team on the left half")
    parser.add_argument("--right", default="strawberry", help="target for the team on the right half")
    parser.add_argument("--seats-per-team", type=int, default=4, help="agents per team (default 4)")
    parser.add_argument("--turns", type=int, default=20, help="total turns")
    parser.add_argument("--half-size", type=int, default=32, help="width and height of each team's half")
    parser.add_argument("--model", default="claude-sonnet-5", help="model for every agent seat")
    parser.add_argument("--policy", default=None, help="inline policy prompt shared by all seats")
    parser.add_argument("--policy-file", default=None, help="path to a policy prompt file shared by all seats")
    parser.add_argument("--port", type=int, default=8332)
    parser.add_argument(
        "--scorer",
        choices=["both", "judge", "mobileclip", "quickdraw"],
        default="both",
        help="grader for each half (default both: the mean of a vision-LLM judge and the "
        "deterministic CLIP score, which halves the judge's turn-to-turn noise)",
    )
    args = parser.parse_args()

    if args.policy and args.policy_file:
        parser.error("use either --policy or --policy-file, not both")
    policy = args.policy
    if args.policy_file:
        policy = Path(args.policy_file).read_text()

    per_team = args.seats_per_team
    half = args.half_size
    teams = build_teams(args.left, args.right, per_team, half)

    asyncio.run(
        run(
            seats=per_team * 2,
            turns=args.turns,
            turns_per_round=None,
            target=f"{args.left} vs {args.right}",
            model=args.model,
            port=args.port,
            policy=policy,
            scorer=args.scorer,
            teams=teams,
            width=half * 2,
            height=half,
            slug=f"versus-{args.left.replace(' ', '-')}-vs-{args.right.replace(' ', '-')}",
        )
    )


if __name__ == "__main__":
    main()
