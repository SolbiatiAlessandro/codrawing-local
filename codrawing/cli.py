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
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO_ROOT / "runs" / f"agent-{target.replace(' ', '-')}-{stamp}"
    run_dir.mkdir(parents=True)
    policy_path: Path | None = None
    if policy is not None:
        # Keep the policy with the run artifacts for provenance.
        policy_path = run_dir / "policy.md"
        policy_path.write_text(policy)
    tokens = [secrets.token_hex(8) for _ in range(seats)]
    config = {
        "tokens": tokens,
        "players": [{"name": f"Agent {index + 1}"} for index in range(seats)],
        "width": 24,
        "height": 24,
        "max_turns": turns,
        "targets": [target],
        "player_connect_timeout_seconds": 60,
        "action_timeout_seconds": 300,
        "model": model,
        "scorer": scorer,
        "environment": f"{target} · {turns} turns"
        + (f" · {scorer}" if scorer != "quickdraw" else ""),
    }
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
        players, logs = [], []
        for slot in range(seats):
            workspace = run_dir / f"workspace-{slot}"
            workspace.mkdir()
            player_env = os.environ | {
                "PYTHONPATH": str(REPO_ROOT),
                "CODRAWING_PLAYER_WS_URL": (
                    f"ws://127.0.0.1:{port}/player?slot={slot}&token={tokens[slot]}"
                ),
                "AGENT_MODEL": model,
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
        choices=["quickdraw", "mobileclip"],
        default="quickdraw",
        help="canvas grader: quickdraw prototypes (default) or MobileCLIP2-S0 zero-shot "
        "(open vocabulary; needs the mobileclip extra installed)",
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


if __name__ == "__main__":
    main()
