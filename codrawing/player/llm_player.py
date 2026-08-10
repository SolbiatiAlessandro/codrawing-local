from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, cast
from urllib.request import Request, urlopen

import websockets

from codrawing.player.pixel_templates import make_template

SEAT_COLORS = ("#EF4444", "#3B82F6", "#22C55E", "#F59E0B", "#A855F7")


@dataclass
class AgentMemory:
    best_score: float = -1.0
    best_turn: int = 0
    recent_scores: list[tuple[int, float]] = field(default_factory=list)
    last_action: dict[str, Any] | None = None
    last_outcome: str = "no prior action"

    def observe(self, observation: dict[str, Any], slot: int) -> None:
        feedback = observation.get("image_model_feedback")
        if feedback:
            score = float(feedback["target_score"])
            turn = int(feedback["turn"])
            self.recent_scores.append((turn, score))
            self.recent_scores = self.recent_scores[-8:]
            if score > self.best_score:
                self.best_score = score
                self.best_turn = turn

        if self.last_action is None or observation["turn"] == 0:
            return
        if slot in observation.get("previous_collision_slots", []):
            self.last_outcome = "collision-dropped; the canvas did not receive it"
        elif slot in observation.get("previous_accepted_slots", []):
            delta = float(feedback["score_delta"]) if feedback else 0.0
            self.last_outcome = f"accepted; simultaneous team score delta {delta:+.8f}"
        else:
            self.last_outcome = "not accepted (late or invalid)"

    def remember_action(self, decision: dict[str, Any]) -> None:
        paint = decision["paint"]
        self.last_action = {
            "x": int(paint["x"]),
            "y": int(paint["y"]),
            "color": str(paint["color"]),
        }

    def prompt_summary(self) -> str:
        history = ", ".join(
            f"T{turn}={score:.8f}" for turn, score in self.recent_scores
        ) or "none"
        last_action = json.dumps(self.last_action, separators=(",", ":")) if self.last_action else "none"
        best = (
            f"{self.best_score:.8f} at turn {self.best_turn}"
            if self.best_score >= 0
            else "none"
        )
        return f"""Private experimental memory carried across model calls:
- best target score seen: {best}
- recent score history: {history}
- your last submitted pixel: {last_action}
- its observed outcome: {self.last_outcome}"""


def extract_action(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    match = re.search(r"\{", text)
    if match is None:
        raise ValueError("model response did not contain JSON")
    value, _ = json.JSONDecoder().raw_decode(text[match.start() :])
    if not isinstance(value, dict):
        raise ValueError("model response was not a JSON object")
    return cast(dict[str, Any], value)


def prompt_for(
    observation: dict[str, Any],
    slot: int,
    memory: AgentMemory | None = None,
) -> str:
    width, height = observation["width"], observation["height"]
    painted = []
    for index, color in enumerate(observation["canvas"]):
        if color != "#FFFFFF":
            painted.append(f"{index % width},{index // width}:{color}")
    messages = "\n".join(
        f"T{item['turn']} {item['player']}: {item['text']}" for item in observation["recent_messages"]
    ) or "(none yet)"
    feedback = observation.get("image_model_feedback")
    if feedback:
        top_predictions = ", ".join(
            f"{item['label']} {item['probability']:.2%}"
            for item in feedback["top_predictions"]
        )
        image_model_feedback = f"""Shared image-model feedback after turn {feedback['turn']}:
- target score: {feedback['target_score']:.6f} ({feedback['score_delta']:+.6f} this turn)
- evaluation: {'PASSING' if feedback['passing'] else 'NOT PASSING'}; the team's recorded score is the BEST score reached
  during the episode, and the team passes only if that best score strictly exceeds {feedback['pass_threshold']:.0%}
- best target label: {feedback['best_target_label']} (rank {feedback['target_rank']} of {feedback.get('label_count', 1000)})
- top predictions: {top_predictions}
This small classifier is imperfect. Treat score changes as team evidence, not as an instruction to erase a
recognizable drawing or chase unrelated labels."""
    else:
        image_model_feedback = """Shared image-model feedback: unavailable in this run.
The team's recorded score is the best classifier score reached during the episode."""
    memory_summary = memory.prompt_summary() if memory else "Private experimental memory: none yet."
    scorer_knowledge = """The scorer is a BLACK BOX - you do not know how it judges the canvas, and it may be swapped
for a different one at any time, so never assume; MEASURE:
- The only ground truth is the score delta after each simultaneous turn, attributed to the writes that landed that
  turn. Read the top-prediction labels for hints about what the canvas currently resembles.
- Treat the first round as the team's laboratory: run five DIFFERENT experiments (for example a thin outline stroke,
  a filled patch, a far-away isolated pixel, an erase, a shape detail) and report on the board what each did to the
  score. From the next round, exploit what the deltas proved; return to small probes only when progress stalls.
- Generalize from evidence: when a kind of edit repeatedly helps or hurts, say so explicitly on the board so the team
  builds a shared model of the scorer - then play to that model, not to guesses."""
    score_protocol = f"""Every turn, first assess what has happened so far: the score history, the outcome of your last
action, and what the other agents said and painted. You cannot draw a recognizable {observation['target']} by yourself
with one pixel per turn - the only way to score is to collaborate with the other LLM agents.
PLAN FIRST: on the first turns, converge on ONE shared shape plan on the public board (which outline, where its parts
go), then execute it together for the rest of the episode. Once a plan exists, do not restate or renegotiate it -
place your pixel where the plan needs it most and say which plan segment you advanced.
All five seats act SIMULTANEOUSLY: if two or more seats paint the same pixel in the same turn, all of those writes are
dropped. The other seats see the same observation you do and will reach for the same obvious pixel, so never pick the
single most obvious next pixel unless the board shows it is yours: derive a distinct choice from your seat number
(for example, take the plan segment closest to share {slot} of 5), and claim your next coordinate in your message so
the others can route around you.
PROTECT GAINS: the recorded score is the best EVER reached, but a great drawing can still be ruined. If the score
dropped right after your accepted write, erase that exact pixel with #FFFFFF next turn instead of adding more.
CALIBRATE BOLDNESS TO THE SCORE, not to the best-so-far:
- LOW score (under about half the pass bar): the canvas is bad no matter what the best-so-far says. Never hold there.
  A round that ends flat and low means the current approach failed - regroup with BIGGER changes: erase whole failed
  strokes, redraw a different interpretation of the target, try a different scale or position.
- HIGH score (a large fraction of the pass bar): protect it. Repainting one of your own pixels with its existing
  color is a legal write that changes nothing - use that HOLD move instead of experimenting on a winning canvas, and
  make only careful single-pixel refinements with immediate rollback on any drop.
Your public message must cite the signed score delta and state which plan segment you advanced or what you learned."""
    rounds = int(observation.get("rounds", 1) or 1)
    if rounds > 1:
        round_scores = observation.get("round_scores") or []
        history = ", ".join(f"R{i + 1}={score:.4f}" for i, score in enumerate(round_scores)) or "none yet"
        round_context = f"""Round {observation['round']} of {rounds}; turn {observation['round_turn']} of {observation['turns_per_round']} in this round.
At the end of every round the score is LOGGED and compared against other teams: round scores so far: {history}.
"""
        if observation.get("round_turn") == 0 and observation.get("round", 1) > 1:
            round_context += """A NEW ROUND is starting: regroup before you act. Reassess the whole canvas against the plan, judge what last
round's log proved or disproved, and state in your message what the team should do differently this round.
"""
    else:
        round_context = ""
    return f"""You are artist seat {slot} in a five-agent collaborative pixel-art game.
Shared target: {observation['target']}
Canvas: {width}x{height}; x grows right, y grows down; valid x=0..{width - 1}, y=0..{height - 1}.
Turn: {observation['turn']} of {observation['max_turns']}.
{round_context}
Your assigned paint color is {SEAT_COLORS[slot]}. Use exactly this color for paint; #FFFFFF is allowed only to erase
a prior harmful pixel.
Painted pixels as x,y:#RRGGBB (all omitted pixels are white):
{'; '.join(painted) if painted else '(blank canvas)'}
Recent public board:
{messages}

{image_model_feedback}

{scorer_knowledge}

{memory_summary}

{score_protocol}

Choose exactly one experimental pixel and a short evidence-based public message. Call the paint_pixel tool exactly
once and do not add prose.
"""


def extract_model_output(payload: dict[str, Any]) -> str:
    for block in payload.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "paint_pixel":
            tool_input = block.get("input")
            if isinstance(tool_input, dict):
                return json.dumps(tool_input)
    return "".join(
        block.get("text", "")
        for block in payload.get("content", [])
        if block.get("type") == "text"
    )


def call_model(prompt: str, slot: int) -> str:
    request_body: dict[str, Any] = {
        "model": os.environ["ANTHROPIC_MODEL"],
        "max_tokens": 512,
        "temperature": 0.9,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "name": "paint_pixel",
                "description": "Post one public coordination message and paint one canvas pixel.",
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
                                "color": {
                                    "type": "string",
                                    "enum": [SEAT_COLORS[slot], "#FFFFFF"],
                                },
                            },
                        },
                    },
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "paint_pixel"},
    }
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    }
    body = json.dumps(request_body).encode()
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=float(os.environ.get("MODEL_TIMEOUT_SECONDS", "10"))) as response:
        payload = json.loads(response.read())
    return extract_model_output(payload)


def fallback_action(observation: dict[str, Any], slot: int) -> dict[str, Any]:
    plan = make_template(observation["target"], observation["width"], observation["height"])
    assigned = plan[slot :: len(observation["player_names"])]
    index = min(observation["turn"], max(len(assigned) - 1, 0))
    x, y, color = assigned[index] if assigned else (slot, 0, "#FFFFFF")
    return {"message": "", "paint": {"x": x, "y": y, "color": color}}


def validate_decision(decision: dict[str, Any], observation: dict[str, Any]) -> None:
    message, paint = decision.get("message"), decision.get("paint")
    if not isinstance(message, str) or len(message) > 240 or not isinstance(paint, dict):
        raise ValueError("invalid message or paint object")
    x, y, color = paint.get("x"), paint.get("y"), paint.get("color")
    if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, int) or not isinstance(y, int):
        raise ValueError("pixel coordinates must be integers")
    if not (0 <= x < observation["width"] and 0 <= y < observation["height"]):
        raise ValueError("pixel is outside the canvas")
    if not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None:
        raise ValueError("color must use #RRGGBB")


def normalize_decision(decision: dict[str, Any]) -> None:
    """Repair recoverable model output instead of burning a retry on it."""
    message = decision.get("message")
    if isinstance(message, str) and len(message) > 240:
        decision["message"] = message[:240]


def enforce_seat_color(decision: dict[str, Any], slot: int) -> None:
    paint = decision.get("paint")
    if not isinstance(paint, dict):
        return
    requested_color = str(paint.get("color", "")).upper()
    paint["color"] = "#FFFFFF" if requested_color == "#FFFFFF" else SEAT_COLORS[slot]


async def main() -> None:
    url = os.environ["CODRAWING_PLAYER_WS_URL"]
    require_llm = os.environ.get("REQUIRE_LLM", "").lower() in {"1", "true", "yes"}
    max_attempts = int(os.environ.get("MODEL_MAX_ATTEMPTS", "4" if require_llm else "1"))
    stagger_seconds = float(os.environ.get("MODEL_STAGGER_SECONDS", "3" if require_llm else "0"))
    memory = AgentMemory()
    async with websockets.connect(url) as websocket:
        slot: int | None = None
        while True:
            # A slow model call leaves newer observations queued; acting on a
            # stale one submits for an already-resolved turn and is dropped.
            # Drain the queue and act only on the latest observation.
            try:
                pending = [await websocket.recv()]
                while True:
                    try:
                        pending.append(await asyncio.wait_for(websocket.recv(), timeout=0.05))
                    except asyncio.TimeoutError:
                        break
            except websockets.ConnectionClosed:
                return
            observation: dict[str, Any] | None = None
            for raw_message in pending:
                payload = cast(dict[str, Any], json.loads(raw_message))
                if payload["type"] == "welcome":
                    slot = int(payload["slot"])
                elif payload["type"] == "final":
                    return
                elif payload["type"] == "observation":
                    observation = payload
            if observation is None or slot is None:
                continue
            memory.observe(observation, slot)
            if stagger_seconds:
                await asyncio.sleep(slot * stagger_seconds)
            decision: dict[str, Any] | None = None
            model_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    action = await asyncio.to_thread(
                        call_model,
                        prompt_for(observation, slot, memory),
                        slot,
                    )
                    decision = extract_action(action)
                    normalize_decision(decision)
                    enforce_seat_color(decision, slot)
                    validate_decision(decision, observation)
                    break
                except Exception as exc:
                    decision = None
                    model_error = exc
                    if attempt + 1 < max_attempts:
                        delay = 0.5 * (attempt + 1) + 0.15 * slot
                        print(
                            f"model attempt {attempt + 1}/{max_attempts} failed on turn "
                            f"{observation['turn']}; retrying in {delay:.2f}s: {exc}",
                            flush=True,
                        )
                        await asyncio.sleep(delay)

            if decision is None:
                assert model_error is not None
                if require_llm:
                    print(
                        f"required model call failed on turn {observation['turn']} after "
                        f"{max_attempts} attempts: {model_error}",
                        flush=True,
                    )
                    continue
                print(f"model call failed; using deterministic fallback: {model_error}", flush=True)
                decision = fallback_action(observation, slot)
            else:
                print(
                    json.dumps({"event": "llm_action", "slot": slot, "turn": observation["turn"]}),
                    flush=True,
                )
            try:
                decision["turn"] = observation["turn"]
                memory.remember_action(decision)
                await websocket.send(json.dumps(decision))
            except websockets.ConnectionClosed:
                return
            except Exception as exc:
                # A malformed decision must cost one turn, never the process.
                print(f"failed to submit decision on turn {observation['turn']}: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
