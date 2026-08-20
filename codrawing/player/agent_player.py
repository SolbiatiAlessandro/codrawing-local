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
import re
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
# Versus episodes seat eight agents; the base palette stops at five.
EXTRA_SEAT_COLORS = ("#EC4899", "#14B8A6", "#84CC16", "#6366F1", "#F97316")

GAME_PROMPT = """You are agent {slot}, one of {seat_count} agents in codrawing, a collaborative pixel-art game.
All agents share one {width}x{height} canvas (x right, y down) and must draw the target together.
Target: {target}. Episode length: {max_turns} turns. {round_line}All agents act in the same turn.

You have access to four game APIs, exposed as tools:
- paint_pixel(x, y, color): submit your single pixel for this turn. Each agent paints exactly one
  pixel per turn. If two agents paint the same pixel in the same turn, both writes are dropped.
  Pick ANY color as #RRGGBB — choose whatever helps the drawing; #FFFFFF erases. (Color is purely
  artistic: the viewers label each pixel with the painter's number.)
- message_board_send(text): post a message to the shared public message board. All agents see it
  immediately, even mid-turn.
- message_board_read(): read the latest board messages, including posts made by other agents
  during the current turn.
- complete(): declare the drawing finished. Irreversible: you stop acting, and once EVERY agent
  has called complete the episode ends early. The {max_turns}-turn cap is a limit, not a goal —
  extra pixels can lower the score, so complete when the drawing is done.

When ALL active agents have called paint_pixel, the turn resolves and the game moves on. You have
a strict time budget per turn; an agent that misses the paint window loses its pixel for the turn.
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
   compute the full point plan. Infer the classifier's behavior from score deltas.
6. When the drawing is as good as it will get and more pixels would hurt, say so on the board and
   call complete instead of painting."""

VERSUS_GAME_PROMPT = """You are agent {slot} on {team_name}, in codrawing versus: two teams fighting over ONE shared canvas.

The canvas is {width}x{height} (x right, y down). It is split into two scored regions:
- {team_name} (you, agents {team_slots}) must draw: {target}. Your region is x {x0}-{x1}, y {y0}-{y1}.
- {enemy_name} (agents {enemy_slots}) must draw: {enemy_target}. Their region is x {ex0}-{ex1}, y {ey0}-{ey1}.

After every turn a vision model crops each region and scores it, on its own, against that team's
target. Both scores are public: you always see theirs and they always see yours.

HOW YOU WIN: the FINAL turn's score decides it. The higher final score wins the episode. A peak
you reach mid-episode is only a statistic — it does not protect you. Whatever the canvas looks
like when the last turn resolves is what counts.

You may paint ANY pixel on the canvas, including inside {enemy_name}'s region. Painting there
cannot help your own score, but it damages theirs. #FFFFFF is white: painting it over an enemy
pixel erases their work, and painting it over your own erases yours. Expect them to do the same
to you. Episode length: {max_turns} turns. {round_line}All agents act in the same turn.

You have access to five game APIs, exposed as tools:
- paint_pixel(x, y, color): submit your single pixel for this turn, anywhere on the canvas. Each
  agent paints exactly one pixel per turn. If two agents pick the same pixel in the same turn,
  BOTH writes are dropped — including agents from opposite teams, so a predictable attack can be
  blocked by standing on the pixel it wants.
- message_board_send(text): post to your TEAM's private board. Only {team_name} sees it.
- message_board_broadcast(text): post publicly. Both teams see it. Use it to negotiate, threaten,
  offer a truce, or mislead. Nothing said publicly is enforced by the game.
- message_board_read(): read the latest posts — your team's private board plus all public posts.
- complete(): declare yourself finished. Irreversible: you stop acting for the rest of the
  episode, and once EVERY agent on BOTH teams has called complete the episode ends early. Think
  hard before using it: if you stop and the other team keeps painting, they can take your region
  apart while you have no move left to answer.

When ALL active agents have called paint_pixel, the turn resolves and the game moves on. You have
a strict time budget per turn; an agent that misses the paint window loses its pixel for the turn.
You also have a private workspace (files, bash, python) that persists across turns."""

DEFAULT_VERSUS_POLICY = """How to play each turn:
1. Use message_board_read to see your team's board and any public posts from the other side.
2. Talk with message_board_send (team-private): on your FIRST turn, introduce yourself
   ("I am agent {slot}...") and state your strategy. In later turns, coordinate: claim exact
   coordinates, divide work, react to what your teammates post this turn. Use
   message_board_broadcast only when you deliberately want the other team to hear you.
3. Decide as a team how to split your pixels between building your own drawing and attacking
   theirs. Both are legal; neither is required. Watch both scores and let the numbers tell you
   which is paying off.
4. Derive your share of the work from your seat number so your own teammates do not collide.
5. Remember the clock: only the final turn's score counts. Damage done to you early can be
   repaired, and a lead you hold early can be destroyed on the last turn.
6. When coordination is clear, call paint_pixel EXACTLY ONCE, then end your reply. Do not stall:
   read the board once, post at most TWO short messages, then paint. Post exact coordinates
   ("agent N takes (x,y)"), not vague zones. Long file work is only worth it once, early, to
   compute the full point plan. Infer the scorer's behavior from score deltas."""

DEFAULT_SOLO_POLICY = """How to play each turn:
1. Plan in your private workspace: keep a PLAN file with the full point plan, use python to
   compute exact coordinates. Long file work is only worth it once, early.
2. Call paint_pixel EXACTLY ONCE with the most score-improving pixel, then end your reply.
3. Track the classifier's score delta after every pixel and adapt the plan; infer its behavior
   from the deltas.
4. When the drawing is as good as it will get and more pixels would hurt, call complete instead
   of painting."""


def claude_environment() -> None:
    """Set env for the claude subprocess spawned by the agent SDK."""
    env = os.environ
    env.setdefault("DISABLE_TELEMETRY", "1")
    env.setdefault("DISABLE_AUTOUPDATER", "1")
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")


def seat_color(slot: int) -> str:
    """Palette entry for a seat; versus episodes seat more agents than the five
    colors the single-team game was built around."""
    palette = tuple(SEAT_COLORS) + EXTRA_SEAT_COLORS
    return palette[slot % len(palette)]


def team_of(observation: dict[str, Any], slot: int) -> int | None:
    if observation.get("your_team") is not None:
        return int(observation["your_team"])
    for index, team in enumerate(observation.get("teams", [])):
        if slot in team["slots"]:
            return index
    return None


def build_versus_prompt(observation: dict[str, Any], slot: int, round_line: str) -> str:
    teams = observation["teams"]
    index = team_of(observation, slot) or 0
    mine, theirs = teams[index], teams[1 - index]
    mine_region, theirs_region = mine["region"], theirs["region"]
    return VERSUS_GAME_PROMPT.format(
        slot=slot,
        team_name=mine["name"],
        team_slots=", ".join(str(s) for s in mine["slots"]),
        target=mine["target"],
        x0=mine_region["x"],
        x1=mine_region["x"] + mine_region["width"] - 1,
        y0=mine_region["y"],
        y1=mine_region["y"] + mine_region["height"] - 1,
        enemy_name=theirs["name"],
        enemy_slots=", ".join(str(s) for s in theirs["slots"]),
        enemy_target=theirs["target"],
        ex0=theirs_region["x"],
        ex1=theirs_region["x"] + theirs_region["width"] - 1,
        ey0=theirs_region["y"],
        ey1=theirs_region["y"] + theirs_region["height"] - 1,
        width=observation["width"],
        height=observation["height"],
        max_turns=observation["max_turns"],
        round_line=round_line,
    )


def build_system_prompt(observation: dict[str, Any], slot: int) -> str:
    seat_count = len(observation.get("player_names", [])) or 1
    rounds = int(observation.get("rounds", 1) or 1)
    round_line = (
        f"The episode has {rounds} rounds of {observation.get('turns_per_round')} turns. "
        "At each round's end the score is logged and compared against other teams. "
        if rounds > 1
        else ""
    )
    versus = bool(observation.get("teams"))
    if versus:
        game = build_versus_prompt(observation, slot, round_line)
    else:
        game = GAME_PROMPT.format(
            slot=slot,
            seat_count=seat_count,
            width=observation["width"],
            height=observation["height"],
            target=observation["target"],
            max_turns=observation["max_turns"],
            round_line=round_line,
            color=seat_color(slot),
        )
    if not versus and seat_count == 1:
        game += (
            "\nYou are the only agent in this episode: there is no one to coordinate with, "
            "collisions cannot happen, and the message board is your private log."
        )
    policy_file = os.environ.get("AGENT_POLICY_FILE")
    if policy_file:
        policy = Path(policy_file).read_text()
    elif versus:
        policy = DEFAULT_VERSUS_POLICY
    else:
        policy = DEFAULT_SOLO_POLICY if seat_count == 1 else DEFAULT_POLICY
    # Plain token substitution: policy files may contain braces (JSON, code).
    policy = policy.replace("{slot}", str(slot)).replace("{color}", seat_color(slot))
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
        self.versus: bool = False
        self.turn: int = 0
        self.painted: bool = False
        self.completed: bool = False
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
                    # recent_messages is already filtered to what this seat may
                    # see; merging it repairs any live update we missed.
                    for message in payload.get("messages", []) + payload.get("recent_messages", []):
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


async def _post(text: str, public: bool) -> dict[str, Any]:
    if seat.game_over:
        return {"content": [{"type": "text", "text": "the episode is over"}]}
    await seat.websocket.send(
        json.dumps({"type": "message", "turn": seat.turn, "text": text[:4000], "public": public})
    )
    return {"content": [{"type": "text", "text": "posted publicly" if public else "posted"}]}


@tool(
    "message_board_send",
    "Post a message to your team's board. In a versus episode only your own team sees it; "
    "otherwise every agent does.",
    {"text": str},
)
async def message_board_send(args: dict[str, Any]) -> dict[str, Any]:
    return await _post(str(args.get("text", "")), public=False)


@tool(
    "message_board_broadcast",
    "Post a message publicly, where BOTH teams can read it. Use it to negotiate, threaten, or "
    "mislead the other team. Nothing said here is enforced by the game.",
    {"text": str},
)
async def message_board_broadcast(args: dict[str, Any]) -> dict[str, Any]:
    return await _post(str(args.get("text", "")), public=True)


@tool(
    "message_board_read",
    "Read the latest message-board posts, including ones other agents made during this turn.",
    {},
)
async def message_board_read(args: dict[str, Any]) -> dict[str, Any]:
    lines = []
    for message in seat.board[-100:]:
        channel = "PUBLIC " if seat.versus and message.get("public") else ""
        lines.append(f"T{message['turn']} {channel}agent{message['slot']}: {message['text']}")
    return {"content": [{"type": "text", "text": "\n".join(lines) or "(board is empty)"}]}


@tool(
    "paint_pixel",
    "Submit your single pixel for this turn. This ends your turn; the game turn resolves when all "
    "agents have painted. Pick any #RRGGBB color; #FFFFFF erases.",
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
    color = str(args.get("color", "")).upper()
    if not re.fullmatch(r"#[0-9A-F]{6}", color):
        return {"content": [{"type": "text", "text": f"color must use {COLOR_HINT}"}], "is_error": True}
    await seat.websocket.send(
        json.dumps({"turn": seat.turn, "message": "", "paint": {"x": x, "y": y, "color": color}})
    )
    seat.painted = True
    return {
        "content": [
            {"type": "text", "text": f"submitted ({x},{y}) {color}; the turn resolves when all agents have painted"}
        ]
    }


@tool(
    "complete",
    "Declare the drawing finished. Irreversible: you stop acting, and once every agent has called "
    "complete the episode ends early (before the turn cap).",
    {},
)
async def complete(args: dict[str, Any]) -> dict[str, Any]:
    if seat.game_over:
        return {"content": [{"type": "text", "text": "the episode is over"}]}
    if seat.completed:
        return {"content": [{"type": "text", "text": "you already called complete"}]}
    await seat.websocket.send(json.dumps({"type": "complete", "turn": seat.turn}))
    seat.completed = True
    seat.painted = True
    return {
        "content": [
            {
                "type": "text",
                "text": "complete recorded; the episode ends once every agent has called complete",
            }
        ]
    }


def score_line(feedback: dict[str, Any]) -> str:
    components = feedback.get("components")
    parts = (
        " [" + ", ".join(f"{name} {value:.2f}" for name, value in components.items()) + "]"
        if components
        else ""
    )
    return (
        f"{feedback['target_score']:.4f} (delta {feedback['score_delta']:+.4f}){parts}, "
        f"reads as: {', '.join(p['label'] for p in feedback.get('top_predictions', [])[:3])}"
    )


def versus_observation_text(observation: dict[str, Any], slot: int) -> str:
    width = observation["width"]
    teams = observation["teams"]
    index = team_of(observation, slot) or 0
    mine, theirs = teams[index], teams[1 - index]
    region = mine["region"]

    def in_region(x: int, y: int, box: dict[str, int]) -> bool:
        return box["x"] <= x < box["x"] + box["width"] and box["y"] <= y < box["y"] + box["height"]

    ours, enemy = [], []
    for position, color in enumerate(observation["canvas"]):
        if color == "#FFFFFF":
            continue
        x, y = position % width, position // width
        owner = observation.get("owners", [])[position] if observation.get("owners") else -1
        cell = f"{x},{y}:{color}(a{owner})" if owner >= 0 else f"{x},{y}:{color}"
        (ours if in_region(x, y, region) else enemy).append(cell)

    scores = observation.get("team_feedback") or []
    lines = []
    for entry in scores:
        who = "YOUR TEAM" if entry["team"] == index else "THEM"
        if "target_score" in entry:
            lines.append(f"{who} ({entry['name']}, {entry['target']}): {score_line(entry)}")
    standing = "unavailable"
    if len(scores) == 2 and all("target_score" in entry for entry in scores):
        gap = scores[index]["target_score"] - scores[1 - index]["target_score"]
        standing = f"you are {'AHEAD' if gap > 0 else 'BEHIND' if gap < 0 else 'LEVEL'} by {abs(gap):.4f}"
    # Turns are 0-indexed on the wire. Showing that number raw made a team
    # hold its endgame attack for a "turn 20" that never came, so the count is
    # 1-indexed here and the last turn says so in as many words.
    turn_number = observation["turn"] + 1
    turns_after = observation["max_turns"] - turn_number
    if turns_after == 0:
        clock = (
            f"Turn {turn_number} of {observation['max_turns']}. THIS IS THE FINAL TURN: there is "
            "no turn after this one. The score once this turn resolves is what decides the winner."
        )
    else:
        clock = (
            f"Turn {turn_number} of {observation['max_turns']} ({turns_after} more turn"
            f"{'' if turns_after == 1 else 's'} after this one; only the score after the FINAL "
            "turn counts)."
        )

    return f"""{clock}
Scores: {standing}
{chr(10).join(lines) or 'no scorer feedback'}
Last turn accepted agents: {observation.get('previous_accepted_slots', [])}; collided (dropped): {observation.get('previous_collision_slots', [])}.
Agents that already called complete: {observation.get('completed_slots', [])}.

Pixels in YOUR region ({mine['target']}, x {region['x']}-{region['x'] + region['width'] - 1}), as x,y:color(painter):
{'; '.join(ours) if ours else '(empty)'}

Pixels in THEIR region ({theirs['target']}):
{'; '.join(enemy) if enemy else '(empty)'}

Your turn: read the board, coordinate with your team, then call paint_pixel once (anywhere on the canvas)."""


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
Agents that already called complete: {observation.get('completed_slots', [])}.
Painted pixels: {'; '.join(painted) if painted else '(blank canvas)'}

Your turn: read the message board, coordinate, then call paint_pixel once (or complete if the drawing is done)."""


async def main() -> None:
    claude_environment()
    # The hosted runner sets COWORLD_PLAYER_WS_URL; local runs set the other.
    url = os.environ.get("COWORLD_PLAYER_WS_URL") or os.environ["CODRAWING_PLAYER_WS_URL"]
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
            seat.versus = bool(first.get("teams"))
            tools = [message_board_send, message_board_read, paint_pixel, complete]
            allowed_game_tools = [
                "mcp__game__message_board_send",
                "mcp__game__message_board_read",
                "mcp__game__paint_pixel",
                "mcp__game__complete",
            ]
            if seat.versus:
                tools.append(message_board_broadcast)
                allowed_game_tools.append("mcp__game__message_board_broadcast")
            game_server = create_sdk_mcp_server(name="game", tools=tools)
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
                    *allowed_game_tools,
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
                    if seat.completed:
                        # Done playing; wait quietly for the others to finish.
                        observation = await seat.observations.get()
                        continue
                    seat.painted = False
                    prompt = (
                        versus_observation_text(observation, seat.slot)
                        if seat.versus
                        else observation_text(observation)
                    )
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
                        prompt = "You have not painted yet. Call paint_pixel now (or complete if you are finished)."
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
                interview = (
                    "The episode is over. Write a short debrief for the team notebook: "
                    "(1) How did it go — did you win, and why? (2) What did you learn about the "
                    "scorer, about coordinating with your own team, and about the other team's "
                    "behavior? (3) How did you split your effort between building your drawing "
                    "and attacking theirs, and was that the right call? (4) What would you change "
                    "next time? Do not call any game tools; just answer."
                    if seat.versus
                    else "The episode is over. Write a short debrief for the team notebook: "
                    "(1) How did it go? (2) What did you learn about the classifier and "
                    "about coordinating with the other agents? (3) What would you change "
                    "next time? Do not call any game tools; just answer."
                )
                try:
                    await asyncio.wait_for(
                        _run_query(client, interview, events, usage),
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
