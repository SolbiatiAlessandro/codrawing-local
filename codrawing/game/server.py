from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import gzip
import json
import os
from pathlib import Path
import secrets
from typing import Any, Literal, cast
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
import zlib

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from codrawing.game.engine import PixelArtEngine, choose_target
from codrawing.game.image_model import TargetScorerRouter, scorer_from_environment


CLIENT_DIR = Path(__file__).parent / "client"
GAME_HOST = os.environ.get("COGAME_HOST", "0.0.0.0")
GAME_PORT = int(os.environ.get("COGAME_PORT", "8080"))
HTTP_USER_AGENT = "codrawing/0.1"


def read_data(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        with urlopen(Request(uri, headers={"User-Agent": HTTP_USER_AGENT}), timeout=30) as response:
            return response.read()
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_bytes()
    if parsed.scheme == "":
        return Path(uri).read_bytes()
    raise ValueError(f"unsupported URI: {uri}")


def write_data(
    uri: str,
    data: bytes | str,
    *,
    content_type: str,
    http_method: Literal["POST", "PUT"],
) -> None:
    payload = data.encode() if isinstance(data, str) else data
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        request = Request(uri, data=payload, method=http_method)
        request.add_header("Content-Type", content_type)
        request.add_header("User-Agent", HTTP_USER_AGENT)
        with urlopen(request, timeout=60):
            return
    path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(uri)
    if parsed.scheme not in ("file", ""):
        raise ValueError(f"unsupported URI: {uri}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def artifact_method(env_var: str) -> Literal["POST", "PUT"]:
    method = os.environ.get(env_var, "PUT").upper()
    if method not in {"POST", "PUT"}:
        raise ValueError(f"{env_var} must be PUT or POST")
    return cast(Literal["POST", "PUT"], method)


def load_replay(uri: str) -> dict[str, Any]:
    data = read_data(uri)
    if uri.endswith(".z"):
        data = zlib.decompress(data)
    elif uri.endswith(".gz"):
        data = gzip.decompress(data)
    return cast(dict[str, Any], json.loads(data))


REPLAY_MODE = "COGAME_LOAD_REPLAY_URI" in os.environ
REPLAY_LOAD_URI = os.environ.get("COGAME_LOAD_REPLAY_URI", "")

if REPLAY_MODE:
    CONFIG: dict[str, Any] = {"tokens": [], "players": []}
    RESULTS_URI = ""
    REPLAY_SAVE_URI = ""
else:
    CONFIG = json.loads(read_data(os.environ["COGAME_CONFIG_URI"]))
    RESULTS_URI = os.environ["COGAME_RESULTS_URI"]
    REPLAY_SAVE_URI = os.environ["COGAME_SAVE_REPLAY_URI"]


TOKENS = CONFIG.get("tokens", [])
PLAYER_NAMES = [player["name"] for player in CONFIG.get("players", [])]
CONNECT_TIMEOUT = float(CONFIG.get("player_connect_timeout_seconds", 30))
ACTION_TIMEOUT = float(CONFIG.get("action_timeout_seconds", 15))


class GameRuntime:
    def __init__(self) -> None:
        self.players: dict[int, WebSocket] = {}
        self.global_viewers: set[WebSocket] = set()
        self.pending_actions: dict[int, dict[str, Any]] = {}
        self.last_resolution: dict[str, Any] = {
            "accepted_slots": [],
            "collision_slots": [],
        }
        self.action_event = asyncio.Event()
        self.started = False
        self.finished = False
        self.frames: list[dict[str, Any]] = []
        self.image_model: TargetScorerRouter | None = scorer_from_environment() if TOKENS else None
        self.image_model_feedback: dict[str, Any] | None = None
        self.image_model_score_trace: list[dict[str, Any]] = []
        self.round_scores: list[float] = []
        episode_seed = CONFIG.get("seed")
        if TOKENS and episode_seed is None:
            episode_seed = secrets.randbits(63)
            CONFIG["seed"] = episode_seed
        self.engine = (
            PixelArtEngine(
                width=int(CONFIG["width"]),
                height=int(CONFIG["height"]),
                max_turns=int(CONFIG["max_turns"]),
                target=choose_target(CONFIG["targets"], episode_seed),
                player_names=PLAYER_NAMES,
                turns_per_round=(
                    int(CONFIG["turns_per_round"]) if CONFIG.get("turns_per_round") else None
                ),
            )
            if TOKENS
            else None
        )
        if self.engine is not None and self.image_model is not None:
            self.image_model_feedback = self._score_canvas()
            self.image_model_score_trace.append(self.image_model_feedback)

    def _score_canvas(self) -> dict[str, Any]:
        assert self.engine is not None
        assert self.image_model is not None
        previous_score = (
            float(self.image_model_feedback["target_score"])
            if self.image_model_feedback is not None
            else None
        )
        return self.image_model.score(
            canvas=self.engine.canvas,
            width=self.engine.width,
            height=self.engine.height,
            target=self.engine.target,
            turn=self.engine.turn,
            previous_score=previous_score,
        )

    def score_canvas(self) -> None:
        if self.image_model is None:
            return
        try:
            self.image_model_feedback = self._score_canvas()
        except Exception as error:
            # A scorer failure must not kill the episode; keep the previous
            # feedback so agents see a stalled score instead of a dead game.
            print(f"scorer failed on turn {self.engine.turn if self.engine else '?'}: {error!r}", flush=True)
            if self.image_model_feedback is None:
                return
            self.image_model_feedback = {**self.image_model_feedback, "score_delta": 0.0}
        self.image_model_score_trace.append(self.image_model_feedback)

    def snapshot(self, *, turn_messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        assert self.engine is not None
        snapshot = self.engine.snapshot(turn_messages=turn_messages)
        if self.image_model_feedback is not None:
            snapshot["image_model_feedback"] = self.image_model_feedback.copy()
        snapshot["round_scores"] = self.round_scores.copy()
        return snapshot


runtime = GameRuntime()
server: uvicorn.Server


@asynccontextmanager
async def lifespan(_app: FastAPI):
    timeout_task = asyncio.create_task(_start_after_connect_timeout()) if TOKENS else None
    yield
    if timeout_task is not None:
        timeout_task.cancel()
        with suppress(asyncio.CancelledError):
            await timeout_task


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


def client(name: str) -> HTMLResponse:
    return HTMLResponse((CLIENT_DIR / name).read_text())


@app.get("/client/global")
def global_client() -> HTMLResponse:
    return client("viewer.html")


@app.get("/client/replay")
def replay_client() -> HTMLResponse:
    return client("viewer.html")


@app.get("/client/player")
def player_client() -> HTMLResponse:
    return client("player.html")


@app.websocket("/global")
async def global_viewer(websocket: WebSocket) -> None:
    await websocket.accept()
    runtime.global_viewers.add(websocket)
    try:
        if runtime.engine is not None:
            await websocket.send_json(runtime.snapshot())
        async for _ in websocket.iter_json():
            pass
    finally:
        runtime.global_viewers.discard(websocket)


@app.websocket("/replay")
async def replay_viewer(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "replay", **load_replay(REPLAY_LOAD_URI)})
    async for _ in websocket.iter_json():
        pass


@app.websocket("/player")
async def player(websocket: WebSocket) -> None:
    slot = int(websocket.query_params.get("slot", "-1"))
    token = websocket.query_params.get("token", "")
    if slot < 0 or slot >= len(TOKENS) or token != TOKENS[slot]:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    runtime.players[slot] = websocket
    await websocket.send_json({"type": "welcome", "slot": slot, "players": PLAYER_NAMES})
    if len(runtime.players) == len(TOKENS) and not runtime.started:
        runtime.started = True
        asyncio.create_task(_play_game())

    try:
        async for raw in websocket.iter_json():
            engine = runtime.engine
            if engine is None or runtime.finished:
                continue
            if raw.get("type") == "complete":
                # Standing vote to end the episode early; a completed seat
                # leaves the paint barrier immediately.
                if engine.mark_complete(slot) and _barrier_met():
                    runtime.action_event.set()
                continue
            if raw.get("turn") != engine.turn:
                continue
            if raw.get("type") == "message":
                # Live board post: visible to every seat immediately, outside
                # the paint barrier.
                if engine.post_message(slot, str(raw.get("text", ""))):
                    await _broadcast_board_update(engine.messages[-1])
                continue
            if slot in runtime.pending_actions:
                continue
            runtime.pending_actions[slot] = raw
            # Barrier: the turn resolves once every active (connected and not
            # completed) player has painted, so latency never costs a write.
            if _barrier_met():
                runtime.action_event.set()
    finally:
        if runtime.players.get(slot) is websocket:
            del runtime.players[slot]


def _barrier_met() -> bool:
    engine = runtime.engine
    if engine is None:
        return False
    active = [slot for slot in runtime.players if slot not in engine.completed]
    return all(slot in runtime.pending_actions for slot in active)


async def _start_after_connect_timeout() -> None:
    await asyncio.sleep(CONNECT_TIMEOUT)
    if not runtime.started and not runtime.finished:
        runtime.started = True
        asyncio.create_task(_play_game())


async def _play_game() -> None:
    assert runtime.engine is not None
    engine = runtime.engine
    await asyncio.sleep(0.2)
    while not engine.done:
        runtime.pending_actions = {}
        runtime.action_event.clear()
        await _broadcast_players()
        try:
            await asyncio.wait_for(runtime.action_event.wait(), timeout=ACTION_TIMEOUT)
        except TimeoutError:
            pass

        resolution = engine.resolve(runtime.pending_actions)
        runtime.last_resolution = resolution
        runtime.score_canvas()
        if runtime.image_model_feedback is not None and (
            engine.turn % engine.turns_per_round == 0 or engine.done
        ):
            runtime.round_scores.append(float(runtime.image_model_feedback["target_score"]))
        snapshot = runtime.snapshot(turn_messages=resolution["messages"])
        snapshot["accepted_slots"] = resolution["accepted_slots"]
        snapshot["collision_slots"] = resolution["collision_slots"]
        runtime.frames.append(snapshot)
        await _broadcast_globals(snapshot)

    results = engine.results()
    if runtime.image_model_feedback is not None:
        # The team competes on the best classifier score reached within the
        # episode's turn budget, so a late regression cannot erase progress.
        best_score = max(
            float(feedback["target_score"])
            for feedback in runtime.image_model_score_trace
        )
        threshold = float(runtime.image_model_feedback["pass_threshold"])
        results["scores"] = [best_score] * len(engine.player_names)
        results["best_target_score"] = best_score
        results["round_scores"] = runtime.round_scores.copy()
        results["image_model"] = runtime.image_model_feedback["model"]
        results["evaluation_threshold"] = threshold
        results["evaluation_passed"] = best_score > threshold
        results["final_image_model_feedback"] = runtime.image_model_feedback
        results["image_model_score_trace"] = runtime.image_model_score_trace
    replay = {"config": CONFIG, "frames": runtime.frames, "results": results}
    write_data(
        RESULTS_URI,
        json.dumps(results),
        content_type="application/json",
        http_method=artifact_method("COGAME_RESULTS_METHOD"),
    )
    write_data(
        REPLAY_SAVE_URI,
        json.dumps(replay),
        content_type="application/json",
        http_method=artifact_method("COGAME_SAVE_REPLAY_METHOD"),
    )
    runtime.finished = True
    await _broadcast_players(final=True)
    await asyncio.sleep(0.5)
    server.should_exit = True


async def _broadcast_players(*, final: bool = False) -> None:
    assert runtime.engine is not None
    engine = runtime.engine
    stale: list[int] = []
    for slot, websocket in list(runtime.players.items()):
        payload = runtime.snapshot()
        payload.update(
            {
                "type": "final" if final else "observation",
                "slot": slot,
                "recent_messages": engine.messages[-25:],
                "previous_accepted_slots": runtime.last_resolution["accepted_slots"],
                "previous_collision_slots": runtime.last_resolution["collision_slots"],
            }
        )
        try:
            await websocket.send_json(payload)
        except Exception:
            stale.append(slot)
    for slot in stale:
        runtime.players.pop(slot, None)


async def _broadcast_board_update(message: dict[str, Any]) -> None:
    payload = {"type": "board_update", "message": message.copy()}
    for sockets in (list(runtime.players.values()), list(runtime.global_viewers)):
        for websocket in sockets:
            try:
                await websocket.send_json(payload)
            except Exception:
                pass


async def _broadcast_globals(snapshot: dict[str, Any]) -> None:
    stale: list[WebSocket] = []
    for websocket in list(runtime.global_viewers):
        try:
            await websocket.send_json(snapshot)
        except Exception:
            stale.append(websocket)
    for websocket in stale:
        runtime.global_viewers.discard(websocket)


if __name__ == "__main__":
    server = uvicorn.Server(uvicorn.Config(app, host=GAME_HOST, port=GAME_PORT))
    server.run()
