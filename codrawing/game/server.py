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

from codrawing.game.engine import PixelArtEngine, Team, choose_target
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
TEAMS = [Team.from_config(payload) for payload in CONFIG.get("teams", [])]
CONNECT_TIMEOUT = float(CONFIG.get("player_connect_timeout_seconds", 30))
ACTION_TIMEOUT = float(CONFIG.get("action_timeout_seconds", 15))
CHECKPOINT_EVERY = int(CONFIG.get("checkpoint_every_turns", 5))


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
        # Both teams' targets go into CLIP's label set, so each half is scored
        # against the other's target rather than distractors alone.
        self.image_model: TargetScorerRouter | None = (
            scorer_from_environment(tuple(team.target for team in TEAMS)) if TOKENS else None
        )
        self.image_model_feedback: dict[str, Any] | None = None
        self.image_model_score_trace: list[dict[str, Any]] = []
        self.round_scores: list[float] = []
        # Adversarial play: one feedback stream per team, each scored from that
        # team's own crop of the shared canvas.
        self.team_feedback: list[dict[str, Any] | None] = [None] * len(TEAMS)
        self.team_score_traces: list[list[dict[str, Any]]] = [[] for _ in TEAMS]
        self.team_round_scores: list[list[float]] = [[] for _ in TEAMS]
        episode_seed = CONFIG.get("seed")
        if TOKENS and episode_seed is None:
            episode_seed = secrets.randbits(63)
            CONFIG["seed"] = episode_seed
        self.engine = (
            PixelArtEngine(
                width=int(CONFIG["width"]),
                height=int(CONFIG["height"]),
                max_turns=int(CONFIG["max_turns"]),
                target=(
                    " vs ".join(team.target for team in TEAMS)
                    if TEAMS
                    else choose_target(CONFIG["targets"], episode_seed)
                ),
                player_names=PLAYER_NAMES,
                turns_per_round=(
                    int(CONFIG["turns_per_round"]) if CONFIG.get("turns_per_round") else None
                ),
                teams=TEAMS or None,
            )
            if TOKENS
            else None
        )

    async def score_initial_canvas(self) -> None:
        if self.engine is not None and self.image_model is not None:
            await self.score_canvas()

    def _score_region(self, team_index: int) -> dict[str, Any]:
        assert self.engine is not None
        assert self.image_model is not None
        team = self.engine.teams[team_index]
        pixels, width, height = self.engine.region_canvas(team_index)
        previous = self.team_feedback[team_index]
        return self.image_model.score(
            canvas=pixels,
            width=width,
            height=height,
            target=team.target,
            turn=self.engine.turn,
            previous_score=float(previous["target_score"]) if previous is not None else None,
        )

    def _score_whole_canvas(self) -> dict[str, Any]:
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

    async def score_canvas(self) -> None:
        """Score the canvas off the event loop; both teams are judged at once.

        A VLM judge takes tens of seconds, which would otherwise stall every
        player websocket while the turn is graded.
        """
        if self.image_model is None or self.engine is None:
            return
        if self.engine.teams:
            scored = await asyncio.gather(
                *(
                    asyncio.to_thread(self._score_region, index)
                    for index in range(len(self.engine.teams))
                ),
                return_exceptions=True,
            )
            for index, outcome in enumerate(scored):
                if isinstance(outcome, BaseException):
                    print(f"scorer failed for team {index} on turn {self.engine.turn}: {outcome!r}", flush=True)
                    previous = self.team_feedback[index]
                    if previous is None:
                        continue
                    # Stall the score rather than kill the episode.
                    self.team_feedback[index] = {**previous, "score_delta": 0.0}
                else:
                    self.team_feedback[index] = outcome
                feedback = self.team_feedback[index]
                if feedback is not None:
                    self.team_score_traces[index].append(feedback)
            # Team 0 also fills the single-target field so any legacy viewer
            # still draws a score line.
            self.image_model_feedback = self.team_feedback[0]
            return
        try:
            self.image_model_feedback = await asyncio.to_thread(self._score_whole_canvas)
        except Exception as error:
            # A scorer failure must not kill the episode; keep the previous
            # feedback so agents see a stalled score instead of a dead game.
            print(f"scorer failed on turn {self.engine.turn}: {error!r}", flush=True)
            if self.image_model_feedback is None:
                return
            self.image_model_feedback = {**self.image_model_feedback, "score_delta": 0.0}
        self.image_model_score_trace.append(self.image_model_feedback)

    def team_feedback_payload(self) -> list[dict[str, Any]]:
        assert self.engine is not None
        payload = []
        for index, team in enumerate(self.engine.teams):
            feedback = self.team_feedback[index]
            payload.append(
                {
                    "team": index,
                    "name": team.name,
                    "target": team.target,
                    **(feedback.copy() if feedback is not None else {}),
                }
            )
        return payload

    def snapshot(self, *, turn_messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        assert self.engine is not None
        snapshot = self.engine.snapshot(turn_messages=turn_messages)
        if self.image_model_feedback is not None:
            snapshot["image_model_feedback"] = self.image_model_feedback.copy()
        if self.engine.teams:
            snapshot["team_feedback"] = self.team_feedback_payload()
        snapshot["round_scores"] = self.round_scores.copy()
        # The viewer shows the model behind each seat when the config names one.
        if CONFIG.get("team_models"):
            snapshot["team_models"] = list(CONFIG["team_models"])
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
                # Live board post: visible to its audience immediately, outside
                # the paint barrier. In team play a post reaches only the
                # sender's team unless it is explicitly public.
                if engine.post_message(slot, str(raw.get("text", "")), bool(raw.get("public"))):
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
    # Baseline score of the blank canvas, so turn 1 already has a delta.
    await runtime.score_initial_canvas()
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
        await runtime.score_canvas()
        if engine.turn % engine.turns_per_round == 0 or engine.done:
            for index, feedback in enumerate(runtime.team_feedback):
                if feedback is not None:
                    runtime.team_round_scores[index].append(float(feedback["target_score"]))
            if not engine.teams and runtime.image_model_feedback is not None:
                runtime.round_scores.append(float(runtime.image_model_feedback["target_score"]))
        snapshot = runtime.snapshot(turn_messages=resolution["messages"])
        snapshot["accepted_slots"] = resolution["accepted_slots"]
        snapshot["collision_slots"] = resolution["collision_slots"]
        runtime.frames.append(snapshot)
        await _broadcast_globals(snapshot)
        # Checkpoint: a long episode that is killed part-way still leaves a
        # readable replay of everything it played. The results artifact is NOT
        # written here. The hosted runner treats results.json as the episode's
        # final output, so a checkpoint that wrote it ended every hosted episode
        # at the first checkpoint turn.
        if engine.turn % CHECKPOINT_EVERY == 0 and not engine.done:
            await asyncio.to_thread(_save_artifacts, _build_artifacts(engine), False)

    _save_artifacts(_build_artifacts(engine))
    runtime.finished = True
    await _broadcast_players(final=True)
    await asyncio.sleep(0.5)
    server.should_exit = True


def _build_artifacts(engine: PixelArtEngine) -> tuple[dict[str, Any], dict[str, Any]]:
    results = engine.results()
    if engine.teams:
        _finish_team_results(results, engine)
    elif runtime.image_model_feedback is not None:
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
    return results, replay


def _save_artifacts(
    artifacts: tuple[dict[str, Any], dict[str, Any]],
    write_results: bool = True,
) -> None:
    results, replay = artifacts
    if write_results:
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


def _finish_team_results(results: dict[str, Any], engine: PixelArtEngine) -> None:
    """Decide the adversarial episode on the FINAL score of each team's region.

    Best-ever is kept as a statistic only: if a peak could not be taken away,
    erasing an opponent's pixels would be pointless and there would be no
    reason to defend a drawing once it was good.
    """
    final_scores: list[float] = []
    for index, team in enumerate(engine.teams):
        feedback = runtime.team_feedback[index]
        trace = runtime.team_score_traces[index]
        final_score = float(feedback["target_score"]) if feedback is not None else 0.0
        best_score = max((float(entry["target_score"]) for entry in trace), default=0.0)
        final_scores.append(final_score)
        results["teams"][index].update(
            {
                "final_score": final_score,
                "best_score": best_score,
                "round_scores": runtime.team_round_scores[index].copy(),
                "score_trace": trace,
                "final_feedback": feedback,
                "pass_threshold": float(feedback["pass_threshold"]) if feedback else None,
            }
        )
    if final_scores:
        results["image_model"] = (runtime.team_feedback[0] or {}).get("model")
    best = max(final_scores, default=0.0)
    leaders = [index for index, score in enumerate(final_scores) if score == best]
    results["winner"] = leaders[0] if len(leaders) == 1 else None
    results["winner_name"] = engine.teams[leaders[0]].name if len(leaders) == 1 else "tie"
    results["final_scores"] = final_scores

    def seat_score(slot: int) -> float:
        team = engine.team_of[slot]
        return final_scores[team] if team is not None else 0.0

    results["scores"] = [seat_score(slot) for slot in range(len(engine.player_names))]


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
                "recent_messages": engine.messages_visible_to(slot)[-25:],
                "previous_accepted_slots": runtime.last_resolution["accepted_slots"],
                "previous_collision_slots": runtime.last_resolution["collision_slots"],
            }
        )
        if engine.teams:
            payload["your_team"] = engine.team_of[slot]
        try:
            await websocket.send_json(payload)
        except Exception:
            stale.append(slot)
    for slot in stale:
        runtime.players.pop(slot, None)


async def _broadcast_board_update(message: dict[str, Any]) -> None:
    payload = {"type": "board_update", "message": message.copy()}
    engine = runtime.engine
    for slot, websocket in list(runtime.players.items()):
        # Team-scoped posts never reach the other side of the table.
        if engine is not None and engine.teams and not message["public"]:
            if engine.team_of[slot] != message["team"]:
                continue
        try:
            await websocket.send_json(payload)
        except Exception:
            pass
    # Viewers watch the whole episode, both teams included.
    for websocket in list(runtime.global_viewers):
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
