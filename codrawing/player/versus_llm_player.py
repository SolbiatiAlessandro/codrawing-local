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
import json
import os
import random
import re
import time
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


def build_prompt(observation: dict[str, Any], slot: int) -> str:
    width, height = observation["width"], observation["height"]
    team = team_of(observation, slot)
    assert team is not None
    index = team_index_of(observation, slot)
    region = team["region"]
    mates = sorted(team["slots"])
    color = SEAT_COLORS[slot % len(SEAT_COLORS)]

    painted = [
        f"{position % width},{position // width}:{value}"
        for position, value in enumerate(observation["canvas"])
        if value != "#FFFFFF"
    ]
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
Your paint color is {color}. Use exactly that color, or #FFFFFF to erase a pixel.
Your teammates are seats {mates}. All seats act SIMULTANEOUSLY. If two seats paint the same
pixel on the same turn, BOTH writes are dropped, so claim your next coordinate on the board
and route around what your teammates claimed.

Painted pixels as x,y:#RRGGBB (everything else is white):
{'; '.join(painted) if painted else '(blank canvas)'}

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


def call_model(prompt: str, timeout: float) -> dict[str, Any]:
    """One InvokeModel call, through the sidecar when the platform provides it."""
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
        return _post_json(url, body, headers, timeout)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("no AWS_ENDPOINT_URL_BEDROCK_RUNTIME and no ANTHROPIC_API_KEY")
    body["model"] = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    return _post_json("https://api.anthropic.com/v1/messages", body, headers, timeout)


def extract_decision(payload: dict[str, Any]) -> dict[str, Any]:
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
    last_error: str | None = None
    for attempt in range(attempts):
        try:
            payload = call_model(prompt, timeout)
            decision = extract_decision(payload)
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
    endpoint = os.environ.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
    print(
        f"[player] bedrock_endpoint={'set' if endpoint else 'MISSING'} "
        f"model={os.environ.get('BEDROCK_MODEL', '(unset)')}",
        flush=True,
    )

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
