"""Team seat driven by a real LLM, one model call per turn.

This is the adversarial two-team counterpart to `llm_player`. Each seat sees
its own team's target, its own team's region and score, and only the board
posts its team can read. It then asks a model for exactly one pixel and one
short message.

Model access goes through the hosted Bedrock sidecar. The platform injects
`AWS_ENDPOINT_URL_BEDROCK_RUNTIME` into a pod whose policy was uploaded with
`--use-bedrock`, and the sidecar signs the call with the real runner identity,
so no model credentials ship in this image. The endpoint and the model id are
read from the environment and never hardcoded; calls use InvokeModel, because
the runner identity is not granted `bedrock:Converse`.

A seat that cannot reach a model still plays: it falls back to its slice of the
deterministic template, so a throttled or failing episode degrades instead of
stalling the turn barrier.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import struct
import time
import zlib
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import websockets

from codrawing.player.pixel_templates import make_template

# One per seat: eight seats play the versus game, four to a side.
SEAT_COLORS = (
    "#EF4444",
    "#3B82F6",
    "#22C55E",
    "#F59E0B",
    "#A855F7",
    "#EC4899",
    "#14B8A6",
    "#F97316",
)

# How many past turns of this seat's own transcript to carry.
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "16"))

# A shape plan does not fit in a tweet. In the local runs that draw a
# recognisable pineapple, the median board post is 324 characters and the
# longest is 2648; a 240-character cap truncated 72% of them mid-sentence.
MESSAGE_LIMIT = int(os.environ.get("MESSAGE_LIMIT", "2000"))

# How many tool steps a seat may take within one turn: it can read the board,
# post, read again to see what a teammate just claimed, then paint.
TURN_STEPS = int(os.environ.get("TURN_STEPS", "4"))

PAINT_TOOL = {
    "name": "paint_pixel",
    "description": "Post one short message to your team's board and paint one canvas pixel.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["message", "paint"],
        "properties": {
            "message": {"type": "string", "maxLength": MESSAGE_LIMIT},
            "leaving_my_zone": {
                "type": "boolean",
                "description": (
                    "Set true ONLY to deliberately paint outside your own row band, "
                    "for example to erase a harmful pixel or to attack the other team. "
                    "Leave it out otherwise: your pixel is snapped back into your band."
                ),
            },
            "paint": {
                "type": "object",
                "additionalProperties": False,
                "required": ["x", "y", "color"],
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "color": {"type": "string"},
                },
            },
        },
    },
}


def ws_url() -> str:
    """The hosted runner sets COWORLD_PLAYER_WS_URL; local runs set the other."""
    url = os.environ.get("COWORLD_PLAYER_WS_URL") or os.environ.get("CODRAWING_PLAYER_WS_URL")
    if not url:
        raise RuntimeError("no player websocket URL in COWORLD_PLAYER_WS_URL/CODRAWING_PLAYER_WS_URL")
    return url


def team_of(observation: dict[str, Any], slot: int) -> dict[str, Any] | None:
    index = observation.get("your_team")
    teams = observation.get("teams") or []
    if isinstance(index, int) and 0 <= index < len(teams):
        return cast(dict[str, Any], teams[index])
    for team in teams:
        if slot in team["slots"]:
            return cast(dict[str, Any], team)
    return None


def team_index_of(observation: dict[str, Any], slot: int) -> int:
    index = observation.get("your_team")
    if isinstance(index, int):
        return index
    for position, team in enumerate(observation.get("teams") or []):
        if slot in team["slots"]:
            return position
    return 0


def fallback_pixel(observation: dict[str, Any], slot: int) -> dict[str, Any]:
    """This seat's slice of the deterministic template, as a last resort."""
    team = team_of(observation, slot)
    if team is None:
        return {"x": 0, "y": 0, "color": SEAT_COLORS[slot % len(SEAT_COLORS)]}
    region = team["region"]
    pixels = make_template(team["target"], region["width"], region["height"])
    mates = sorted(team["slots"])
    rank = mates.index(slot) if slot in mates else 0
    plan = pixels[rank :: max(len(mates), 1)]
    if not plan:
        return {"x": region["x"], "y": region["y"], "color": SEAT_COLORS[slot % len(SEAT_COLORS)]}
    x, y, color = plan[min(int(observation["turn"]), len(plan) - 1)]
    return {"x": x + region["x"], "y": y + region["y"], "color": color}


def seat_zone(team: dict[str, Any], slot: int) -> tuple[int, int, int, int, str]:
    """The band of rows this seat owns, and what usually belongs there.

    Every seat used to receive the same board and the same instructions, so
    every seat named the same pixel and the whole team's writes collided. Bands
    make that impossible: rows are split between teammates, and a pineapple
    decomposes naturally down the same axis - crown at the top, body below.
    """
    region = team["region"]
    mates = sorted(team["slots"])
    rank = mates.index(slot) if slot in mates else 0
    count = max(len(mates), 1)
    height = region["height"]
    top = region["y"] + (height * rank) // count
    bottom = region["y"] + (height * (rank + 1)) // count - 1
    where = ("the very top of the drawing", "the upper middle", "the lower middle", "the bottom")
    return region["x"], region["x"] + region["width"] - 1, top, bottom, where[min(rank, 3)]


def render_png(observation: dict[str, Any], block: int = 12) -> bytes:
    """The canvas as a PNG, built with the standard library only.

    The player image carries no imaging dependency, so the encoder is written
    out here: one uncompressed-filter scanline per row, upscaled so each canvas
    pixel becomes a block a vision model can actually resolve.
    """
    width, height = int(observation["width"]), int(observation["height"])
    canvas = observation["canvas"]

    def rgb(value: str) -> bytes:
        text = value.lstrip("#")
        if len(text) != 6:
            return b"\xff\xff\xff"
        try:
            return bytes.fromhex(text)
        except ValueError:
            return b"\xff\xff\xff"

    raw = bytearray()
    for y in range(height):
        row = b"".join(rgb(canvas[y * width + x]) * block for x in range(width))
        for _ in range(block):
            raw += b"\x00" + row

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width * block, height * block, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def render_canvas(observation: dict[str, Any]) -> tuple[str, str]:
    """The board as a labelled character grid.

    A coordinate list forces the model to rebuild the picture in its head every
    turn, and it grows with every pixel painted. A grid shows the shape, and at
    a full canvas it costs about half as many tokens.
    """
    width, height = int(observation["width"]), int(observation["height"])
    canvas = observation["canvas"]
    palette = sorted({value for value in canvas if value != "#FFFFFF"})
    symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    key = {value: symbols[index % len(symbols)] for index, value in enumerate(palette)}

    rows = [
        "    " + "".join(str(x // 10) if x >= 10 else " " for x in range(width)),
        "    " + "".join(str(x % 10) for x in range(width)),
    ]
    for y in range(height):
        rows.append(f"{y:3d} " + "".join(key.get(canvas[y * width + x], ".") for x in range(width)))
    legend = (
        "colour key: " + ", ".join(f"{symbol}={value}" for value, symbol in key.items())
        if key
        else "colour key: (nothing painted yet)"
    )
    return "\n".join(rows), legend


def build_prompt(observation: dict[str, Any], slot: int) -> str:
    width, height = observation["width"], observation["height"]
    team = team_of(observation, slot)
    assert team is not None
    index = team_index_of(observation, slot)
    region = team["region"]
    mates = sorted(team["slots"])
    color = SEAT_COLORS[slot % len(SEAT_COLORS)]

    canvas_art, palette_legend = render_canvas(observation)
    board = "\n".join(
        f"T{item['turn']} {item['player']}: {item['text']}"
        for item in observation.get("recent_messages") or []
    ) or "(nothing posted yet)"

    feedback_list = observation.get("team_feedback") or []
    mine = feedback_list[index] if index < len(feedback_list) else None
    theirs = feedback_list[1 - index] if len(feedback_list) == 2 else None
    if mine:
        guesses = ", ".join(
            f"{item['label']} {item['probability']:.1%}" for item in mine["top_predictions"][:5]
        )
        score_block = f"""Classifier read of YOUR region after turn {mine['turn']}:
- your target score: {mine['target_score']:.6f} ({mine['score_delta']:+.6f} this turn)
- your target ranks {mine['target_rank']} of {mine.get('label_count', 12)} labels
- what your region currently looks like: {guesses}"""
        if theirs:
            score_block += f"\n- the other team's score: {theirs['target_score']:.6f}"
    else:
        score_block = "Classifier read: not available yet."

    opponent = next((t for t in observation.get("teams") or [] if t is not team), None)
    opponent_line = (
        f"The other team draws '{opponent['target']}' in x={opponent['region']['x']}.."
        f"{opponent['region']['x'] + opponent['region']['width'] - 1}."
        if opponent
        else "There is no other team."
    )

    x0, x1, y0, y1, where = seat_zone(team, slot)
    turn = int(observation["turn"])
    max_turns = int(observation["max_turns"])
    planning = turn < 3

    if planning:
        phase = f"""PHASE: PLANNING (turns 0-2 of {max_turns}). Before anyone can draw a
{team['target']}, the team needs ONE agreed picture. In your message, state the plan in
concrete coordinates: where the outline of the {team['target']} sits inside x={x0}..{x1},
y={region['y']}..{region['y'] + region['height'] - 1}, roughly how wide and tall it is, and
which part of it lands in each seat's rows. Read what your teammates already proposed and
AGREE with it rather than inventing a rival plan - a shared mediocre plan beats four good
private ones. Then paint the first pixel of YOUR part."""
    else:
        phase = f"""PHASE: EXECUTING (turn {turn} of {max_turns}). The plan is settled. Do not
renegotiate it. Paint the next pixel your part needs and say which part you advanced."""

    return f"""You are seat {slot} on {team['name']}. Four of you are drawing ONE picture
together, at the same time, one pixel each per turn. You are not drawing alone.

THE GOAL: together, make the region below look like a recognizable "{team['target']}".

YOUR TEAM'S REGION IS x={x0} TO {x1}. A classifier crops exactly that rectangle and scores
it against "{team['target']}". A pixel you paint outside x={x0}..{x1} does NOTHING for your
score. Check the x of every pixel before you name it.

YOUR OWN ROWS ARE y={y0} TO {y1} - {where} of the picture.
Your three teammates own the other rows and are painting THIS SAME TURN. If two of you name
the same pixel, BOTH writes are thrown away and the turn is wasted. That is why the rows are
split: stay inside y={y0}..{y1} and you can never collide with a teammate. If you truly need
a pixel outside your rows, you must set leaving_my_zone=true, otherwise your pixel is snapped
back into your band.

{phase}

HOW TO DRAW SOMETHING RECOGNIZABLE:
- Make it BIG. Fill most of the {x1 - x0 + 1} columns and most of the rows. A small shape in
  the middle of a white field reads as nothing.
- Outline first, then fill. A closed outline of the right silhouette scores far better than a
  scattered cloud of correct-coloured pixels.
- Colour is yours to choose per pixel: use the colours the real object has, and use the same
  ones your teammates announced.
- Build on what is already painted. Extend an existing stroke by one pixel rather than
  starting a new mark somewhere empty.

Canvas: {width} wide, {height} tall. x grows right, y grows down.
Turn {turn} of {max_turns}.

The canvas, one character per pixel ('.' is unpainted). Column numbers read down the top two
rows, row numbers on the left, so the character at column x and row y IS pixel (x, y):
{canvas_art}
{palette_legend}

Your team's board (only your team sees this):
{board}

{score_block}

{opponent_line}
Regions are scoring boundaries, not painting limits: you MAY paint in the other team's
rectangle to spoil their picture, but it costs you a turn you could have spent on your own,
and it needs leaving_my_zone=true. Only the FINAL turn decides the winner, so a lead can be
taken away, and must be defended.

Call the paint_pixel tool exactly once. Your message must name the pixel you took and the
next one you intend, so your teammates route around you. No prose outside the tool call.
"""


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return cast(dict[str, Any], json.loads(response.read()))


BOARD_READ_TOOL = {
    "name": "message_board_read",
    "description": (
        "Read your team's board RIGHT NOW, including posts your teammates made "
        "during this same turn. Call this before painting to see which pixels "
        "they have already claimed this turn."
    ),
    "input_schema": {"type": "object", "additionalProperties": False, "properties": {}},
}

BOARD_SEND_TOOL = {
    "name": "message_board_send",
    "description": (
        "Post to your team's board immediately, before you paint. Teammates see "
        "it this turn. Use it to propose the shape plan and to claim the pixel "
        "you are about to take. Set public=true to speak to the other team too."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "maxLength": MESSAGE_LIMIT},
            "public": {"type": "boolean"},
        },
    },
}

TURN_TOOLS = [PAINT_TOOL, BOARD_READ_TOOL, BOARD_SEND_TOOL]

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
    }
    for t in TURN_TOOLS
]


def call_model(messages: list[dict[str, Any]], timeout: float, image: bytes | None = None) -> tuple[str, dict[str, Any]]:
    """One model call. Returns (wire format, response payload).

    Routing is chosen by which credential the pod was given, cheapest-intent
    first: OpenRouter when a key is present, else the hosted Bedrock sidecar,
    else a direct Anthropic key for local runs.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        model = os.environ.get("OPENROUTER_MODEL")
        if not model:
            raise RuntimeError("OPENROUTER_API_KEY is set but OPENROUTER_MODEL is not")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openrouter_key}",
            "X-Title": "coplace-versus",
        }
        turn_messages = [dict(m) for m in messages]
        content: Any = turn_messages[-1]["content"]
        if image is not None:
            encoded = base64.b64encode(image).decode()
            content = [
                {"type": "text", "text": content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
        payload = _post_json(
            f"{base}/chat/completions",
            {
                "model": model,
                # Reasoning models spend this budget thinking before they emit
                # the tool call, so a tight cap silently costs the turn.
                "max_tokens": int(os.environ.get("MODEL_MAX_TOKENS", "1600")),
                "temperature": 0.7,
                **(
                    {"reasoning": {"effort": os.environ["REASONING_EFFORT"]}}
                    if os.environ.get("REASONING_EFFORT")
                    else {}
                ),
                "messages": turn_messages[:-1] + [{"role": "user", "content": content}],
                "tools": OPENAI_TOOLS,
                "tool_choice": "required",
            },
            headers,
            timeout,
        )
        return "openai", payload

    endpoint = os.environ.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
    body: dict[str, Any] = {
        "max_tokens": 640,
        "temperature": 0.7,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "tools": TURN_TOOLS,
        "tool_choice": {"type": "any"},
    }
    if endpoint:
        # Hosted: the sidecar holds the identity and re-signs. Send no auth.
        model = os.environ.get("BEDROCK_MODEL")
        if not model:
            raise RuntimeError("BEDROCK_MODEL is unset but a Bedrock endpoint was provided")
        body["anthropic_version"] = "bedrock-2023-05-31"
        url = f"{endpoint.rstrip('/')}/model/{model}/invoke"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        return "anthropic", _post_json(url, body, headers, timeout)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("no OPENROUTER_API_KEY, no Bedrock endpoint, and no ANTHROPIC_API_KEY")
    body["model"] = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    return "anthropic", _post_json("https://api.anthropic.com/v1/messages", body, headers, timeout)


def extract_call(wire: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if wire == "openai":
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError(f"no choices in response: {json.dumps(payload)[:300]}")
        message = choices[0].get("message") or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            if name in {t["name"] for t in TURN_TOOLS}:
                arguments = function.get("arguments")
                value = json.loads(arguments) if isinstance(arguments, str) else arguments
                if isinstance(value, dict):
                    return name, cast(dict[str, Any], value)
        text = message.get("content") or ""
        match = re.search(r"\{", text)
        if match is None:
            raise ValueError("model returned no tool call")
        value, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        if not isinstance(value, dict):
            raise ValueError("model returned a non-object decision")
        return "paint_pixel", cast(dict[str, Any], value)

    for block in payload.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") in {t["name"] for t in TURN_TOOLS}:
            value = block.get("input")
            if isinstance(value, dict):
                return str(block["name"]), cast(dict[str, Any], value)
    text = "".join(
        block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
    )
    match = re.search(r"\{", text)
    if match is None:
        raise ValueError("model returned no tool call")
    value, _ = json.JSONDecoder().raw_decode(text[match.start() :])
    if not isinstance(value, dict):
        raise ValueError("model returned a non-object decision")
    return "paint_pixel", cast(dict[str, Any], value)


def sanitize(decision: dict[str, Any], observation: dict[str, Any], slot: int) -> dict[str, Any]:
    """Clamp the model's pixel onto the canvas and force this seat's color."""
    paint = decision.get("paint") or {}
    width, height = int(observation["width"]), int(observation["height"])
    try:
        x = max(0, min(width - 1, int(paint["x"])))
        y = max(0, min(height - 1, int(paint["y"])))
    except (KeyError, TypeError, ValueError):
        return fallback_pixel(observation, slot)
    color = str(paint.get("color", "")).upper()
    if not re.fullmatch(r"#[0-9A-F]{6}", color):
        color = SEAT_COLORS[slot % len(SEAT_COLORS)]

    # Snap strays home. A pixel a teammate is also reaching for is a wasted turn,
    # and a pixel outside the team's rectangle does nothing for the score, so both
    # are corrected unless the seat said it meant to leave.
    if os.environ.get("SEAT_ZONES") == "1" and not decision.get("leaving_my_zone"):
        team = team_of(observation, slot)
        if team is not None:
            x0, x1, y0, y1, _ = seat_zone(team, slot)
            x = max(x0, min(x1, x))
            y = max(y0, min(y1, y))
    return {"x": x, "y": y, "color": color}


def turn_recap(decision: dict[str, Any], observation: dict[str, Any], slot: int) -> str:
    """What this seat did last turn, in its own transcript, so it remembers."""
    paint = decision["paint"]
    note = decision.get("message") or ""
    return f"I painted ({paint['x']}, {paint['y']}) with {paint['color']}. I told the team: {note}"


async def run_turn(
    websocket: Any,
    observation: dict[str, Any],
    slot: int,
    history: list[dict[str, Any]],
    board: list[dict[str, Any]],
    attempts: int,
    timeout: float,
) -> dict[str, Any]:
    """Play one turn as a short tool loop, ending in exactly one painted pixel.

    A seat used to get a single blind call: it decided while its three
    teammates were deciding, all of them looking at the same stale board, so
    they reached for the same pixel. The server has always accepted live board
    posts outside the paint barrier - it just was never used. Now a seat can
    read the board, claim a pixel out loud, read again to see what a teammate
    claimed in the meantime, and only then paint.
    """
    messages = list(history)
    messages.append({"role": "user", "content": build_prompt(observation, slot)})
    image = render_png(observation) if os.environ.get("CANVAS_IMAGE") == "1" else None
    turn = int(observation["turn"])
    posted = ""

    for step in range(TURN_STEPS):
        last_error: str | None = None
        name, args = "", {}
        for attempt in range(attempts):
            try:
                wire, payload = await asyncio.to_thread(call_model, messages, timeout, image)
                name, args = extract_call(wire, payload)
                break
            except HTTPError as error:
                detail = ""
                try:
                    detail = error.read().decode()[:300]
                except Exception:
                    pass
                last_error = f"HTTP {error.code} {detail}"
                if error.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                    await asyncio.sleep(min(8.0, (2**attempt) + random.random()))
                    continue
                break
            except (URLError, TimeoutError, ValueError, RuntimeError) as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt < attempts - 1:
                    await asyncio.sleep(1.0 + attempt)
                    continue
                break
        if not name:
            print(f"[seat {slot}] model call failed: {last_error}", flush=True)
            break

        if name == "paint_pixel":
            return {
                "paint": sanitize(args, observation, slot),
                "message": str(args.get("message") or "")[:MESSAGE_LIMIT],
            }

        if name == "message_board_read":
            visible = "\n".join(
                f"T{m['turn']} {m.get('player', '?')}: {m['text']}" for m in board[-25:]
            ) or "(nothing posted yet)"
            messages.append({"role": "assistant", "content": "I read the board."})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Board right now, including posts made this turn:\n{visible}\n\n"
                        "Now claim a pixel nobody else has taken and call paint_pixel."
                    ),
                }
            )
            continue

        if name == "message_board_send":
            text = str(args.get("text") or "")[:MESSAGE_LIMIT]
            if text:
                posted = text
                await websocket.send(
                    json.dumps(
                        {
                            "turn": turn,
                            "type": "message",
                            "text": text,
                            "public": bool(args.get("public")),
                        }
                    )
                )
            messages.append({"role": "assistant", "content": f"I posted: {text}"})
            messages.append(
                {
                    "role": "user",
                    "content": "Posted, and your teammates can see it now. Read the board again or paint.",
                }
            )
            continue

    return {"paint": fallback_pixel(observation, slot), "message": posted}


async def main() -> None:
    url = ws_url()
    attempts = int(os.environ.get("MODEL_MAX_ATTEMPTS", "3"))
    timeout = float(os.environ.get("MODEL_TIMEOUT_SECONDS", "35"))
    if os.environ.get("OPENROUTER_API_KEY"):
        route = f"openrouter model={os.environ.get('OPENROUTER_MODEL', '(unset)')}"
    elif os.environ.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME"):
        route = f"bedrock-sidecar model={os.environ.get('BEDROCK_MODEL', '(unset)')}"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        route = "anthropic-direct"
    else:
        route = "NONE (every turn will fall back to the template)"
    print(f"[player] llm route: {route}", flush=True)

    async with websockets.connect(url, max_size=None) as websocket:
        slot: int | None = None
        history: list[dict[str, Any]] = []
        board: list[dict[str, Any]] = []
        observations: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        finished = asyncio.Event()

        async def receive() -> None:
            """Keep draining the socket so board posts arrive while we think."""
            nonlocal slot
            try:
                async for raw in websocket:
                    event = cast(dict[str, Any], json.loads(raw))
                    kind = event.get("type")
                    if kind == "welcome":
                        slot = int(event["slot"])
                        print(f"[player] connected as seat {slot}", flush=True)
                    elif kind == "board_update":
                        board.append(event["message"])
                    elif kind == "observation":
                        board.extend(event.get("recent_messages") or [])
                        del board[:-50]
                        await observations.put(event)
                    elif kind == "final":
                        finished.set()
                        return
            finally:
                finished.set()

        receiver = asyncio.create_task(receive())
        try:
            while not finished.is_set():
                getter = asyncio.create_task(observations.get())
                done, _ = await asyncio.wait(
                    {getter, asyncio.create_task(finished.wait())},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if getter not in done:
                    getter.cancel()
                    break
                observation = getter.result()
                if slot is None or team_of(observation, slot) is None:
                    await websocket.send(json.dumps({"turn": observation["turn"]}))
                    continue

                decision = await run_turn(
                    websocket, observation, slot, history, board, attempts, timeout
                )

                # Keep the seat's own decisions, drop the boards they were made
                # against: the reasoning has to survive, the grids do not.
                for entry in history:
                    if entry["role"] == "user" and len(entry["content"]) > 400:
                        entry["content"] = "(earlier board, omitted)"
                history.append(
                    {"role": "user", "content": f"Turn {observation['turn']}: board as shown above."}
                )
                history.append(
                    {"role": "assistant", "content": turn_recap(decision, observation, slot)}
                )
                if len(history) > 2 * HISTORY_TURNS:
                    history[:] = history[:2] + history[-2 * (HISTORY_TURNS - 1) :]

                payload: dict[str, Any] = {
                    "turn": observation["turn"],
                    "paint": decision["paint"],
                }
                if decision["message"]:
                    payload["message"] = decision["message"]
                await websocket.send(json.dumps(payload))
        finally:
            receiver.cancel()
        print(f"[seat {slot}] episode finished", flush=True)


asyncio.run(main())
