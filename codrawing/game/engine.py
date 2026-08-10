from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
import re
from typing import Any


COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
MAX_MESSAGE_LENGTH = 4000


@dataclass(frozen=True)
class Paint:
    x: int
    y: int
    color: str


@dataclass(frozen=True)
class Action:
    message: str
    paint: Paint


def choose_target(targets: list[str], seed: int | str | None) -> str:
    if not targets:
        raise ValueError("targets must not be empty")
    if seed is None:
        raise ValueError("the game must mint a seed before choosing a target")
    return random.Random(str(seed)).choice(targets)


class PixelArtEngine:
    """Pure, deterministic simultaneous-turn pixel-art rules."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        max_turns: int,
        target: str,
        player_names: list[str],
        turns_per_round: int | None = None,
    ) -> None:
        if width < 1 or height < 1 or max_turns < 1:
            raise ValueError("width, height, and max_turns must be positive")
        if not player_names:
            raise ValueError("at least one player is required")
        if turns_per_round is not None and turns_per_round < 1:
            raise ValueError("turns_per_round must be positive")
        self.width = width
        self.height = height
        self.max_turns = max_turns
        self.turns_per_round = turns_per_round or max_turns
        self.target = target
        self.player_names = player_names.copy()
        self.turn = 0
        self.canvas = ["#FFFFFF"] * (width * height)
        self.owners = [-1] * (width * height)
        self.accepted_pixels = [0] * len(player_names)
        self.messages: list[dict[str, Any]] = []
        self.completed: set[int] = set()

    @property
    def done(self) -> bool:
        return self.turn >= self.max_turns or len(self.completed) >= len(self.player_names)

    def mark_complete(self, slot: int) -> bool:
        """Record a seat's standing vote to end the episode early."""
        if not (0 <= slot < len(self.player_names)) or self.turn >= self.max_turns:
            return False
        self.completed.add(slot)
        return True

    @property
    def rounds(self) -> int:
        return -(-self.max_turns // self.turns_per_round)

    @property
    def round(self) -> int:
        """1-based round of the current turn (clamped to the last round when done)."""
        return min(self.turn // self.turns_per_round, self.rounds - 1) + 1

    @property
    def round_turn(self) -> int:
        """0-based turn index within the current round."""
        return self.turn - (self.round - 1) * self.turns_per_round

    def post_message(self, slot: int, text: str) -> bool:
        """Post one board message immediately, outside the paint barrier."""
        if self.done or not (0 <= slot < len(self.player_names)):
            return False
        if not isinstance(text, str):
            return False
        text = text.strip()
        if not text or len(text) > MAX_MESSAGE_LENGTH:
            return False
        posted_this_turn = sum(
            1 for m in self.messages if m["turn"] == self.turn and m["slot"] == slot
        )
        if posted_this_turn >= 8:
            return False
        self.messages.append(
            {
                "turn": self.turn,
                "slot": slot,
                "player": self.player_names[slot],
                "text": text,
            }
        )
        return True

    def parse_action(self, raw: Any) -> Action | None:
        if not isinstance(raw, dict):
            return None
        message = raw.get("message", "")
        paint = raw.get("paint")
        if not isinstance(message, str) or not isinstance(paint, dict):
            return None
        if len(message) > MAX_MESSAGE_LENGTH:
            return None
        x, y, color = paint.get("x"), paint.get("y"), paint.get("color")
        if isinstance(x, bool) or isinstance(y, bool):
            return None
        if not isinstance(x, int) or not isinstance(y, int) or not isinstance(color, str):
            return None
        if not (0 <= x < self.width and 0 <= y < self.height and COLOR_RE.fullmatch(color)):
            return None
        return Action(message=message.strip(), paint=Paint(x, y, color.upper()))

    def resolve(self, raw_actions: dict[int, Any]) -> dict[str, Any]:
        # Completion votes may land mid-turn; still resolve that turn's paints.
        if self.turn >= self.max_turns:
            raise RuntimeError("episode is already complete")

        valid: dict[int, Action] = {}
        for slot, raw in raw_actions.items():
            if 0 <= slot < len(self.player_names):
                action = self.parse_action(raw)
                if action is not None:
                    valid[slot] = action

        by_pixel: dict[tuple[int, int], list[int]] = defaultdict(list)
        for slot, action in valid.items():
            by_pixel[(action.paint.x, action.paint.y)].append(slot)

        accepted: list[int] = []
        collided: list[int] = []
        for slot in sorted(valid):
            action = valid[slot]
            if action.message:
                self.messages.append(
                    {
                        "turn": self.turn,
                        "slot": slot,
                        "player": self.player_names[slot],
                        "text": action.message,
                    }
                )
            contenders = by_pixel[(action.paint.x, action.paint.y)]
            if len(contenders) > 1:
                collided.append(slot)
                continue
            index = action.paint.y * self.width + action.paint.x
            self.canvas[index] = action.paint.color
            self.owners[index] = slot
            self.accepted_pixels[slot] += 1
            accepted.append(slot)

        turn_messages = [message.copy() for message in self.messages if message["turn"] == self.turn]
        self.turn += 1
        return {
            "turn": self.turn,
            "accepted_slots": accepted,
            "collision_slots": collided,
            "messages": turn_messages,
        }

    def snapshot(self, *, turn_messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "type": "state",
            "width": self.width,
            "height": self.height,
            "target": self.target,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "turns_per_round": self.turns_per_round,
            "rounds": self.rounds,
            "round": self.round,
            "round_turn": self.round_turn,
            "canvas": self.canvas.copy(),
            "owners": self.owners.copy(),
            "accepted_pixels": self.accepted_pixels.copy(),
            "player_names": self.player_names.copy(),
            "messages": (turn_messages or []).copy(),
            "completed_slots": sorted(self.completed),
            "done": self.done,
        }

    def results(self) -> dict[str, Any]:
        return {
            # This MVP is human-scored. Zeroes keep the platform contract honest:
            # accepted pixel counts are diagnostics, not a quality score.
            "scores": [0.0] * len(self.player_names),
            "target": self.target,
            "turns": self.turn,
            "max_turns": self.max_turns,
            "completed_slots": sorted(self.completed),
            "ended_by_agents": len(self.completed) >= len(self.player_names),
            "accepted_pixels": self.accepted_pixels.copy(),
            "final_canvas": self.canvas.copy(),
        }
