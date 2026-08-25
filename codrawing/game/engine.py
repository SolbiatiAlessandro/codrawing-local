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


@dataclass(frozen=True)
class Team:
    """One side of an adversarial episode: a target and the region scored for it.

    The region is the rectangle cropped out of the shared canvas and handed to
    the scorer as its own image. It is *not* a painting restriction: any seat
    may paint any pixel of the canvas, including inside the other team's
    region, which is what makes sabotage possible.
    """

    name: str
    target: str
    x: int
    y: int
    width: int
    height: int
    slots: tuple[int, ...]

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "Team":
        region = payload.get("region", {})
        return cls(
            name=str(payload["name"]),
            target=str(payload["target"]),
            x=int(region["x"]),
            y=int(region["y"]),
            width=int(region["width"]),
            height=int(region["height"]),
            slots=tuple(int(slot) for slot in payload["slots"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "region": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "slots": list(self.slots),
        }

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


def _validate_teams(teams: list[Team], width: int, height: int, player_count: int) -> None:
    seen: set[int] = set()
    for team in teams:
        if team.width < 1 or team.height < 1:
            raise ValueError(f"team {team.name!r} has an empty region")
        if team.x < 0 or team.y < 0 or team.x + team.width > width or team.y + team.height > height:
            raise ValueError(f"team {team.name!r} has a region outside the canvas")
        for slot in team.slots:
            if not (0 <= slot < player_count):
                raise ValueError(f"team {team.name!r} claims unknown slot {slot}")
            if slot in seen:
                raise ValueError(f"slot {slot} belongs to more than one team")
            seen.add(slot)
    if len(seen) != player_count:
        raise ValueError("every player must belong to exactly one team")


def _team_index_by_slot(teams: list[Team], player_count: int) -> list[int | None]:
    by_slot: list[int | None] = [None] * player_count
    for index, team in enumerate(teams):
        for slot in team.slots:
            by_slot[slot] = index
    return by_slot


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
        teams: list[Team] | None = None,
    ) -> None:
        if width < 1 or height < 1 or max_turns < 1:
            raise ValueError("width, height, and max_turns must be positive")
        if not player_names:
            raise ValueError("at least one player is required")
        if turns_per_round is not None and turns_per_round < 1:
            raise ValueError("turns_per_round must be positive")
        if teams:
            _validate_teams(teams, width, height, len(player_names))
        self.teams = list(teams) if teams else []
        self.team_of = _team_index_by_slot(self.teams, len(player_names))
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

    def region_canvas(self, team_index: int) -> tuple[list[str], int, int]:
        """Crop one team's region out of the shared canvas as its own image."""
        team = self.teams[team_index]
        pixels: list[str] = []
        for y in range(team.y, team.y + team.height):
            start = y * self.width + team.x
            pixels.extend(self.canvas[start : start + team.width])
        return pixels, team.width, team.height

    def _message_record(self, slot: int, text: str, public: bool) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "slot": slot,
            "player": self.player_names[slot],
            "text": text,
            # Without teams every message is public, which is the old behavior.
            "team": self.team_of[slot] if self.teams else None,
            "public": True if not self.teams else public,
        }

    def messages_visible_to(self, slot: int) -> list[dict[str, Any]]:
        """The board as one seat sees it: public posts plus its own team's."""
        if not self.teams:
            return self.messages
        team = self.team_of[slot]
        return [m for m in self.messages if m["public"] or m["team"] == team]

    def post_message(self, slot: int, text: str, public: bool = False) -> bool:
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
        self.messages.append(self._message_record(slot, text, public))
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
                # A message bundled with paint is team-scoped in team play; use
                # an explicit public post to talk across the table.
                self.messages.append(self._message_record(slot, action.message, public=False))
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
            "collision_pixels": [
                {"x": x, "y": y, "slots": slots}
                for (x, y), slots in by_pixel.items()
                if len(slots) > 1
            ],
            "messages": turn_messages,
        }

    def snapshot(self, *, turn_messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        snapshot = {
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
        if self.teams:
            snapshot["teams"] = [team.to_dict() for team in self.teams]
        return snapshot

    def results(self) -> dict[str, Any]:
        if self.teams:
            return {
                "scores": [0.0] * len(self.player_names),
                "target": self.target,
                "teams": [
                    team.to_dict()
                    | {"accepted_pixels": sum(self.accepted_pixels[slot] for slot in team.slots)}
                    for team in self.teams
                ],
                "turns": self.turn,
                "max_turns": self.max_turns,
                "completed_slots": sorted(self.completed),
                "ended_by_agents": len(self.completed) >= len(self.player_names),
                "accepted_pixels": self.accepted_pixels.copy(),
                "final_canvas": self.canvas.copy(),
            }
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
