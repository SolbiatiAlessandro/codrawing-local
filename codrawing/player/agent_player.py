"""Seat driven by a Claude Code agent session with real game tools.

Each seat is one persistent Claude Agent SDK session. Its system prompt has
two parts:

- The fixed GAME prompt: the rules of the game and the three game APIs
  (paint_pixel, message_board_send, message_board_read). Identical for every
  seat and every policy.
- The POLICY prompt: how this seat chooses to play. Overridable per seat via
  the AGENT_POLICY_FILE env var; this is the part policy authors submit.

The game APIs are exposed to the agent as in-process MCP tools:

- paint_pixel(x, y, color): submit this turn's single pixel. The game turn
  resolves only when every agent has painted.
- message_board_send(text): post to the shared public message board, visible
  to all agents immediately (live, mid-turn).
- message_board_read(): read the latest board messages, including posts made
  by other agents during the current turn.

Each turn the seat appends one JSON line to AGENT_TRACE_FILE (if set) with
its full trace: text, thinking, tool calls, token usage, and timing.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, cast

import websockets
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)

from codrawing.player.llm_player import SEAT_COLORS

COLOR_HINT = "#RRGGBB"

GAME_PROMPT = """You are agent {slot}, one of {seat_count} agents in codrawing, a collaborative pixel-art game.
All agents share one {width}x{height} canvas (x right, y down) and must draw the target together.
Target: {target}. Episode length: {max_turns} turns. {round_line}All agents act in the same turn.

You have access to three game APIs, exposed as tools:
- paint_pixel(x, y, color): submit your single pixel for this turn. Each agent paints exactly one
  pixel per turn. If two agents paint the same pixel in the same turn, both writes are dropped.
  Your paint color is {color}; #FFFFFF erases.
- message_board_send(text): post a message (max 240 chars) to the shared public message board.
  All agents see it immediately, even mid-turn.
- message_board_read(): read the latest board messages, including posts made by other agents
  during the current turn.

When ALL agents have called paint_pixel, the turn resolves and the game moves on. You have a
strict time budget per turn; an agent that misses the paint window loses its pixel for the turn.
A black-box classifier scores the canvas after every turn; the team's recorded score is the BEST
score ever reached. You also have a private workspace (files, bash, python) that persists across
turns."""

DEFAULT_POLICY = """How to play each turn:
1. Use message_board_read to see what the other agents are saying right now.
2. Talk with message_board_send: on your FIRST turn, introduce yourself ("I am agent {slot}...")
   and state your strategy. In later turns, coordinate: claim coordinates, divide work, react to
   what others post this turn.
3. Plan in your private workspace: keep a PLAN file, use python to compute exact coordinates.
4. Derive your share of the work from your seat number so agents do not collide.
5. When coordination is clear, call paint_pixel EXACTLY ONCE, then end your reply. Do not stall:
   read the board once, post at most TWO short messages, then paint. Post exact coordinates
   ("agent N takes (x,y)"), not vague zones. Long file work is only worth it once, early, to
   compute the full point plan. Infer the classifier's behavior from score deltas."""

DEFAULT_SOLO_POLICY = """How to play each turn:
1. Plan in your private workspace: keep a PLAN file with the full point plan, use python to
   compute exact coordinates. Long file work is only worth it once, early.
2. Call paint_pixel EXACTLY ONCE with the most score-improving pixel, then end your reply.
3. Track the classifier's score delta after every pixel and adapt the plan; infer its behavior
   from the deltas."""


def claude_environment() -> None:
    """Set env for the claude subprocess spawned by the agent SDK."""
    env = os.environ
    env.setdefault("DISABLE_TELEMETRY", "1")
    env.setdefault("DISABLE_AUTOUPDATER", "1")
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")


def build_system_prompt(observation: dict[str, Any], slot: int) -> str:
    seat_count = len(observation.get("player_names", [])) or 1
    rounds = int(observation.get("rounds", 1) or 1)
    round_line = (
        f"The episode has {rounds} rounds of {observation.get('turns_per_round')} turns. "
        "At each round's end the score is logged and compared against other teams. "
        if rounds > 1
        else ""
    )
    game = GAME_PROMPT.format(
        slot=slot,
        seat_count=seat_count,
        width=observation["width"],
        height=observation["height"],
        target=observation["target"],
        max_turns=observation["max_turns"],
        round_line=round_line,
        color=SEAT_COLORS[slot],
    )
    if seat_count == 1:
        game += (
            "\nYou are the only agent in this episode: there is no one to coordinate with, "
            "collisions cannot happen, and the message board is your private log."
        )
    policy_file = os.environ.get("AGENT_POLICY_FILE")
    if policy_file:
        policy = Path(policy_file).read_text()
    else:
        policy = DEFAULT_SOLO_POLICY if seat_count == 1 else DEFAULT_POLICY
    # Plain token substitution: policy files may contain braces (JSON, code).
    policy = policy.replace("{slot}", str(slot)).replace("{color}", SEAT_COLORS[slot])
    return f"{game}\n\n# Your policy\n{policy}"


class TraceWriter:
    """Appends one JSON line per turn with the seat's full activity."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


class Seat:
    def __init__(self) -> None:
        self.websocket: Any = None
        self.slot: int | None = None
        self.turn: int = 0
        self.painted: bool = False
        self.game_over: bool = False
        self.board: list[dict[str, Any]] = []
        self.observations: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    @property
    def color(self) -> str:
        return SEAT_COLORS[self.slot or 0]

    async def reader(self) -> None:
        try:
            async for raw in self.websocket:
                payload = cast(dict[str, Any], json.loads(raw))
                kind = payload.get("type")
                if kind == "welcome":
                    self.slot = int(payload["slot"])
                elif kind == "board_update":
                    self.board.append(payload["message"])
                elif kind == "observation":
                    for message in payload.get("messages", []):
                        if message not in self.board:
                            self.board.append(message)
                    await self.observations.put(payload)
                elif kind == "final":
                    self.game_over = True
                    await self.observations.put(None)
                    return
        except websockets.ConnectionClosed:
            self.game_over = True
            await self.observations.put(None)


seat = Seat()


@tool(
    "message_board_send",
    "Post a message (max 240 chars) to the shared public message board. All agents see it immediately.",
    {"text": str},
)
async def message_board_send(args: dict[str, Any]) -> dict[str, Any]:
    if seat.game_over:
        return {"content": [{"type": "text", "text": "the episode is over"}]}
    text = str(args.get("text", ""))[:240]
    await seat.websocket.send(json.dumps({"type": "message", "turn": seat.turn, "text": text}))
    return {"content": [{"type": "text", "text": "posted"}]}


@tool(
    "message_board_read",
    "Read the latest message-board posts, including ones other agents made during this turn.",
    {},
)
async def message_board_read(args: dict[str, Any]) -> dict[str, Any]:
    tail = seat.board[-100:]
    text = "\n".join(f"T{m['turn']} agent{m['slot']}: {m['text']}" for m in tail) or "(board is empty)"
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "paint_pixel",
    "Submit your single pixel for this turn. This ends your turn; the game turn resolves when all "
    "agents have painted. Use your seat color, or #FFFFFF to erase.",
    {"x": int, "y": int, "color": str},
)
async def paint_pixel(args: dict[str, Any]) -> dict[str, Any]:
    if seat.game_over:
        return {"content": [{"type": "text", "text": "the episode is over"}]}
    if seat.painted:
        return {"content": [{"type": "text", "text": "you already painted this turn"}]}
    try:
        x, y = int(args["x"]), int(args["y"])
    except (KeyError, TypeError, ValueError):
        return {"content": [{"type": "text", "text": "x and y must be integers"}], "is_error": True}
    color = str(args.get("color", seat.color)).upper()
    if color != "#FFFFFF":
        color = seat.color
    await seat.websocket.send(
        json.dumps({"turn": seat.turn, "message": "", "paint": {"x": x, "y": y, "color": color}})
    )
    seat.painted = True
    return {
        "content": [
            {"type": "text", "text": f"submitted ({x},{y}) {color}; the turn resolves when all agents have painted"}
        ]
    }


def observation_text(observation: dict[str, Any]) -> str:
    width = observation["width"]
    painted = [
        f"{index % width},{index // width}:{color}"
        for index, color in enumerate(observation["canvas"])
        if color != "#FFFFFF"
    ]
    feedback = observation.get("image_model_feedback")
    if feedback:
        score_line = (
            f"score {feedback['target_score']:.6f} (delta {feedback['score_delta']:+.6f}), "
            f"rank {feedback['target_rank']}/{feedback.get('label_count', '?')}, "
            f"top: {', '.join(p['label'] + ' ' + format(p['probability'], '.1%') for p in feedback['top_predictions'][:3])}"
        )
    else:
        score_line = "unavailable"
    return f"""Turn {observation['turn']} of {observation['max_turns']} (round {observation.get('round', 1)}/{observation.get('rounds', 1)}).
Classifier: {score_line}
Last turn accepted agents: {observation.get('previous_accepted_slots', [])}; collided: {observation.get('previous_collision_slots', [])}.
Painted pixels: {'; '.join(painted) if painted else '(blank canvas)'}

Your turn: read the message board, coordinate, then call paint_pixel once."""


async def main() -> None:
    claude_environment()
    url = os.environ["CODRAWING_PLAYER_WS_URL"]
    model = os.environ.get("AGENT_MODEL", "claude-sonnet-5")
    workspace = Path(os.environ.get("AGENT_WORKSPACE", "/tmp/agent-workspace")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    turn_timeout = float(os.environ.get("AGENT_TIMEOUT_SECONDS", "240"))
    trace_path = os.environ.get("AGENT_TRACE_FILE")
    trace = TraceWriter(Path(trace_path) if trace_path else None)

    async with websockets.connect(url, max_size=None) as websocket:
        seat.websocket = websocket
        reader = asyncio.create_task(seat.reader())
        try:
            first = await seat.observations.get()
            if first is None or seat.slot is None:
                return
            game_server = create_sdk_mcp_server(
                name="game", tools=[message_board_send, message_board_read, paint_pixel]
            )
            options = ClaudeAgentOptions(
                system_prompt=build_system_prompt(first, seat.slot),
                mcp_servers={"game": game_server},
                allowed_tools=[
                    "Bash",
                    "Read",
                    "Write",
                    "Edit",
                    "Glob",
                    "Grep",
                    "mcp__game__message_board_send",
                    "mcp__game__message_board_read",
                    "mcp__game__paint_pixel",
                ],
                permission_mode="bypassPermissions",
                cwd=str(workspace),
                model=model,
                max_turns=30,
                # Without a display setting the CLI emits thinking blocks with
                # empty content; summarized keeps the reasoning in the trace.
                thinking={"type": "adaptive", "display": "summarized"},
            )
            async with ClaudeSDKClient(options=options) as client:
                session_cost = 0.0

                def cost_delta(usage: dict[str, Any]) -> None:
                    nonlocal session_cost
                    cumulative = usage.pop("session_cost_usd", None)
                    if cumulative is not None:
                        usage["cost_usd"] = round(max(0.0, cumulative - session_cost), 6)
                        session_cost = cumulative

                observation: dict[str, Any] | None = first
                while observation is not None:
                    seat.turn = int(observation["turn"])
                    seat.painted = False
                    prompt = observation_text(observation)
                    turn_started = time.monotonic()
                    events: list[dict[str, Any]] = []
                    usage: dict[str, Any] = {}
                    for nudge in range(3):
                        budget = turn_timeout if nudge == 0 else 25.0
                        try:
                            reply = await asyncio.wait_for(
                                _run_query(client, prompt, events, usage), timeout=budget
                            )
                        except (TimeoutError, asyncio.TimeoutError):
                            print(f"turn {seat.turn}: agent query timed out", flush=True)
                            events.append({"type": "timeout", "budget_seconds": budget})
                            if nudge == 2 or seat.painted:
                                break
                            reply = ""
                        except Exception as exc:
                            print(f"turn {seat.turn}: agent query failed: {exc!r}", flush=True)
                            events.append({"type": "error", "error": repr(exc)})
                            break
                        if seat.painted:
                            break
                        if reply:
                            print(f"turn {seat.turn}: no paint in reply: {reply[:200]}", flush=True)
                        prompt = "You have not painted yet. Call paint_pixel immediately."
                    cost_delta(usage)
                    trace.write(
                        {
                            "slot": seat.slot,
                            "phase": "turn",
                            "turn": seat.turn,
                            "painted": seat.painted,
                            "wall_seconds": round(time.monotonic() - turn_started, 2),
                            "usage": usage,
                            "events": events,
                        }
                    )
                    if seat.painted:
                        print(
                            json.dumps(
                                {
                                    "event": "llm_action",
                                    "slot": seat.slot,
                                    "turn": seat.turn,
                                    "harness": "claude-agent-sdk",
                                }
                            ),
                            flush=True,
                        )
                    else:
                        print(f"turn {seat.turn}: no paint submitted", flush=True)
                    observation = await seat.observations.get()

                # Post-episode interview: one final reflection, recorded in the trace.
                interview_started = time.monotonic()
                events = []
                usage = {}
                try:
                    await asyncio.wait_for(
                        _run_query(
                            client,
                            "The episode is over. Write a short debrief for the team notebook: "
                            "(1) How did it go? (2) What did you learn about the classifier and "
                            "about coordinating with the other agents? (3) What would you change "
                            "next time? Do not call any game tools; just answer.",
                            events,
                            usage,
                        ),
                        timeout=120,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    events.append({"type": "timeout", "budget_seconds": 120})
                except Exception as exc:
                    events.append({"type": "error", "error": repr(exc)})
                cost_delta(usage)
                trace.write(
                    {
                        "slot": seat.slot,
                        "phase": "interview",
                        "wall_seconds": round(time.monotonic() - interview_started, 2),
                        "usage": usage,
                        "events": events,
                    }
                )
        finally:
            reader.cancel()


def _clip(value: Any, limit: int = 2000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} chars clipped]"


async def _run_query(
    client: ClaudeSDKClient,
    prompt: str,
    events: list[dict[str, Any]],
    usage: dict[str, Any],
) -> str:
    await client.query(prompt)
    parts: list[str] = []
    async for message in client.receive_response():
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
                events.append({"type": "text", "text": str(text)})
                continue
            thinking = getattr(block, "thinking", None)
            if thinking is not None:
                if str(thinking).strip():
                    events.append({"type": "thinking", "text": _clip(str(thinking))})
                continue
            name = getattr(block, "name", None)
            if name is not None:
                events.append(
                    {"type": "tool_use", "name": str(name), "input": _clip(getattr(block, "input", {}))}
                )
                continue
            tool_content = getattr(block, "content", None)
            if tool_content is not None:
                events.append({"type": "tool_result", "content": _clip(tool_content)})
        message_usage = getattr(message, "usage", None)
        if isinstance(message_usage, dict):
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                if message_usage.get(key):
                    usage[key] = usage.get(key, 0) + int(message_usage[key])
        cost = getattr(message, "total_cost_usd", None)
        if cost is not None:
            # Cumulative for the whole session; the caller converts to a delta.
            usage["session_cost_usd"] = round(float(cost), 6)
        duration = getattr(message, "duration_ms", None)
        if duration is not None:
            usage["duration_ms"] = usage.get("duration_ms", 0) + int(duration)
        result = getattr(message, "result", None)
        if result:
            parts.append(str(result))
        if getattr(message, "is_error", False):
            parts.append(f"[error message: {message}]")
    return " ".join(parts)


if __name__ == "__main__":
    asyncio.run(main())
