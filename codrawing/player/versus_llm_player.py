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

PAINT_TOOL = {
    "name": "paint_pixel",
    "description": "Post one short message to your team's board and paint one canvas pixel.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["message", "paint"],
        "properties": {
            "message": {"type": "string", "maxLength": 240},
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

    rows = ["    " + "".join(str(x % 10) for x in range(width))]
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

    return f"""You are seat {slot} on {team['name']} in an adversarial pixel-art game.

YOUR TARGET: draw a recognizable "{team['target']}".
Your team scores ONLY on the rectangle x={region['x']}..{region['x'] + region['width'] - 1},
y={region['y']}..{region['y'] + region['height'] - 1}. A frozen image classifier crops that
rectangle every turn and scores it against "{team['target']}".
{opponent_line}
Regions are scoring boundaries, not painting limits: any seat may paint any pixel on the
whole {width}x{height} canvas, including inside the other team's rectangle. Only the score on
the FINAL turn decides the winner, so a lead can be taken away and must be defended.

Canvas: {width}x{height}; x grows right, y grows down; x=0..{width - 1}, y=0..{height - 1}.
Turn {observation['turn']} of {observation['max_turns']}.
Pick whatever colour the drawing needs at that pixel, as #RRGGBB - a gold body
and a green crown read as a pineapple; four seats each painting one fixed colour
never will. Use #FFFFFF to erase. Agree on the palette with your team and stick
to it. (Your default colour if you cannot decide is {color}.)
Your teammates are seats {mates}. All seats act SIMULTANEOUSLY. If two seats paint the same
pixel on the same turn, BOTH writes are dropped, so claim your next coordinate on the board
and route around what your teammates claimed.

The canvas, one character per pixel ('.' is unpainted). Column numbers run
across the top, row numbers down the left, so the character at column x and
row y IS pixel (x, y):
{canvas_art}
{palette_legend}

Your team's board (only your team can read this):
{board}

{score_block}

How to play well:
- Agree on ONE shape plan early, then execute it. Do not renegotiate a working plan.
- Draw the target big enough to fill a good part of your rectangle. A few scattered pixels
  never reads as anything.
- The score delta after each turn is your only ground truth. If your accepted pixel dropped
  the score, erase that exact pixel with #FFFFFF next turn.
- Late in the episode, protect a good score instead of experimenting. Repainting one of your
  own pixels in its existing color is a legal no-op HOLD move.
- Attacking the other team's rectangle costs you a turn you could have spent drawing. Do it
  only when your own drawing is finished and their score is close to yours.

Call the paint_pixel tool exactly once. Your message should say which part of the plan you
just advanced and which pixel you will take next. Do not write any prose outside the tool call.
"""


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return cast(dict[str, Any], json.loads(response.read()))


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": PAINT_TOOL["name"],
            "description": PAINT_TOOL["description"],
            "parameters": PAINT_TOOL["input_schema"],
        },
    }
]


def call_model(prompt: str, timeout: float, image: bytes | None = None) -> tuple[str, dict[str, Any]]:
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
        content: Any = prompt
        if image is not None:
            encoded = base64.b64encode(image).decode()
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
        payload = _post_json(
            f"{base}/chat/completions",
            {
                "model": model,
                "max_tokens": 640,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": content}],
                "tools": OPENAI_TOOLS,
                "tool_choice": {"type": "function", "function": {"name": "paint_pixel"}},
            },
            headers,
            timeout,
        )
        return "openai", payload

    endpoint = os.environ.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
    body: dict[str, Any] = {
        "max_tokens": 640,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [PAINT_TOOL],
        "tool_choice": {"type": "tool", "name": "paint_pixel"},
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


def extract_decision(wire: str, payload: dict[str, Any]) -> dict[str, Any]:
    if wire == "openai":
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError(f"no choices in response: {json.dumps(payload)[:300]}")
        message = choices[0].get("message") or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") == "paint_pixel":
                arguments = function.get("arguments")
                value = json.loads(arguments) if isinstance(arguments, str) else arguments
                if isinstance(value, dict):
                    return cast(dict[str, Any], value)
        text = message.get("content") or ""
        match = re.search(r"\{", text)
        if match is None:
            raise ValueError("model returned no paint_pixel tool call")
        value, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        if not isinstance(value, dict):
            raise ValueError("model returned a non-object decision")
        return cast(dict[str, Any], value)

    for block in payload.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "paint_pixel":
            value = block.get("input")
            if isinstance(value, dict):
                return cast(dict[str, Any], value)
    text = "".join(
        block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
    )
    match = re.search(r"\{", text)
    if match is None:
        raise ValueError("model returned no paint_pixel tool call")
    value, _ = json.JSONDecoder().raw_decode(text[match.start() :])
    if not isinstance(value, dict):
        raise ValueError("model returned a non-object decision")
    return cast(dict[str, Any], value)


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
    return {"x": x, "y": y, "color": color}


def decide(observation: dict[str, Any], slot: int, attempts: int, timeout: float) -> dict[str, Any]:
    """Return {'paint': ..., 'message': ...}; never raise."""
    prompt = build_prompt(observation, slot)
    image = render_png(observation) if os.environ.get("CANVAS_IMAGE") == "1" else None
    last_error: str | None = None
    for attempt in range(attempts):
        try:
            wire, payload = call_model(prompt, timeout, image)
            decision = extract_decision(wire, payload)
            message = str(decision.get("message") or "")[:240]
            return {"paint": sanitize(decision, observation, slot), "message": message}
        except HTTPError as error:
            detail = ""
            try:
                detail = error.read().decode()[:400]
            except Exception:
                pass
            last_error = f"HTTP {error.code} {detail}"
            # 429 is the throttle/spend-limit signal: back off and retry.
            if error.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(min(8.0, (2**attempt) + random.random()))
                continue
            break
        except (URLError, TimeoutError, ValueError, RuntimeError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < attempts - 1:
                time.sleep(min(4.0, 1.0 + attempt))
                continue
            break
    print(f"[seat {slot}] model call failed, using template fallback: {last_error}", flush=True)
    return {"paint": fallback_pixel(observation, slot), "message": ""}


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
        async for raw in websocket:
            observation = cast(dict[str, Any], json.loads(raw))
            kind = observation.get("type")
            if kind == "welcome":
                slot = int(observation["slot"])
                print(f"[player] connected as seat {slot}", flush=True)
                continue
            if kind == "final":
                print(f"[seat {slot}] episode finished", flush=True)
                return
            if kind != "observation" or slot is None:
                continue
            if team_of(observation, slot) is None:
                await websocket.send(json.dumps({"turn": observation["turn"]}))
                continue

            # The model call blocks; keep it off the event loop so the socket
            # stays responsive to board updates while this seat is thinking.
            decision = await asyncio.to_thread(decide, observation, slot, attempts, timeout)
            payload: dict[str, Any] = {"turn": observation["turn"], "paint": decision["paint"]}
            if decision["message"]:
                payload["message"] = decision["message"]
            await websocket.send(json.dumps(payload))


asyncio.run(main())
