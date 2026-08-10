from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast

import websockets

from codrawing.player.pixel_templates import make_template


async def main() -> None:
    url = os.environ["CODRAWING_PLAYER_WS_URL"]
    async with websockets.connect(url) as websocket:
        slot: int | None = None
        async for raw_message in websocket:
            observation = cast(dict[str, Any], json.loads(raw_message))
            if observation["type"] == "welcome":
                slot = int(observation["slot"])
                continue
            if observation["type"] == "final":
                return
            if observation["type"] != "observation" or slot is None:
                continue

            plan = make_template(observation["target"], observation["width"], observation["height"])
            player_count = len(observation["player_names"])
            assigned = plan[slot::player_count]
            index = min(observation["turn"], max(len(assigned) - 1, 0))
            x, y, color = assigned[index] if assigned else (slot, 0, "#FFFFFF")
            message = (
                f"I am seat {slot}; painting my share of the {observation['target']} template."
                if observation["turn"] == 0
                else ""
            )
            await websocket.send(
                json.dumps(
                    {
                        "turn": observation["turn"],
                        "message": message,
                        "paint": {"x": x, "y": y, "color": color},
                    }
                )
            )


asyncio.run(main())
