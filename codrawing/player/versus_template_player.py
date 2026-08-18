"""Deterministic no-LLM baseline for the adversarial two-team game.

Certification needs a player that finishes an episode the same way every time,
without a model behind it. This one draws its team's target inside its team's
own region and never raids the other half: a baseline should establish what
honest play scores, so a policy that does raid is measured against it.

Seats split the template between them by stride, so four seats drawing at once
never contend for the same pixel.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast

import websockets

from codrawing.player.pixel_templates import make_template


def team_of(observation: dict[str, Any], slot: int) -> dict[str, Any] | None:
    for team in observation.get("teams") or []:
        if slot in team["slots"]:
            return team
    return None


def plan_for(team: dict[str, Any], slot: int) -> list[tuple[int, int, str]]:
    """The pixels this seat will paint, in order, in canvas coordinates."""
    region = team["region"]
    # The template is drawn at the region's size and then shifted into place,
    # so a team that owns the right half draws a whole target there rather
    # than the right-hand slice of a canvas-sized one.
    pixels = make_template(team["target"], region["width"], region["height"])
    rank = sorted(team["slots"]).index(slot)
    stride = max(len(team["slots"]), 1)
    return [
        (x + region["x"], y + region["y"], color)
        for x, y, color in pixels[rank::stride]
    ]


async def main() -> None:
    url = os.environ.get("COWORLD_PLAYER_WS_URL") or os.environ["CODRAWING_PLAYER_WS_URL"]
    async with websockets.connect(url) as websocket:
        slot: int | None = None
        plan: list[tuple[int, int, str]] = []
        async for raw_message in websocket:
            observation = cast(dict[str, Any], json.loads(raw_message))
            kind = observation.get("type")
            if kind == "welcome":
                slot = int(observation["slot"])
                continue
            if kind == "final":
                return
            if kind != "observation" or slot is None:
                continue

            team = team_of(observation, slot)
            if team is None:
                # Not a team episode: nothing this baseline knows how to do.
                await websocket.send(json.dumps({"turn": observation["turn"]}))
                continue
            if not plan:
                plan = plan_for(team, slot)
            if not plan:
                await websocket.send(json.dumps({"turn": observation["turn"]}))
                continue

            # Past the end of the plan the seat repaints its last pixel: a
            # harmless no-op that keeps the turn barrier moving.
            x, y, color = plan[min(observation["turn"], len(plan) - 1)]
            payload: dict[str, Any] = {
                "turn": observation["turn"],
                "paint": {"x": x, "y": y, "color": color},
            }
            if observation["turn"] == 0:
                payload["message"] = (
                    f"Seat {slot} on {team['name']}: painting my share of the "
                    f"{team['target']} inside our own region."
                )
            await websocket.send(json.dumps(payload))


asyncio.run(main())
