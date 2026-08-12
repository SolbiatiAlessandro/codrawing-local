# codrawing

A local, self-contained multi-agent drawing game. Five agents share one 24x24
pixel canvas. Each turn, every agent paints exactly one pixel. A black-box
image classifier scores the canvas after every turn. The team wins if the
score for the shared target (for example "light bulb") passes a threshold.

Everything runs on your laptop:

- The game server, engine, and scorer are pure Python.
- The scorer is a bundled 21 KB Quick, Draw! nearest-prototype model. No ML
  libraries are needed.
- Agent seats use your locally installed `claude` and `codex` CLIs with your
  subscription login. No API keys are needed.

## Setup

```bash
uv venv
uv pip install -e .
```

## Run an episode

Full agent seats (one persistent Claude Code session per seat):

```bash
.venv/bin/codrawing-run --target "light bulb" --turns 20
.venv/bin/codrawing-run --target cat --rounds 2 --turns-per-round 10 \
    --model claude-sonnet-5 \
    --policy "outline the silhouette first, then fill gaps"
```

`--policy` (inline text) or `--policy-file` replaces the default policy
prompt for every seat; the fixed game prompt never changes. The policy is
saved to the run directory as `policy.md`. Warning: identical deterministic
policies collide — every seat computes the same "best pixel" and the
collision rule drops them all. Good policies derive each seat's share of the
work from its seat number.

`--seats 1` is the single-model baseline: the agent gets a solo prompt (no
coordination, no collisions, the board is its private log). Compare against
group runs in the same environment — the runs home page groups runs by
environment (target + turns). For pixel parity give the solo run seats x
turns total turns (for example 4 agents x 20 turns vs `--seats 1 --turns 80`).

After the final turn every agent answers a post-episode interview (how it
went, what it learned, what to change next time), recorded in its trace and
shown at the top of its panel in the report. Traces also capture the model's
summarized thinking, every tool call, token usage, and per-turn timing.

Single-call CLI seats (faster, cheaper, alternates `codex` and `claude`):

```bash
.venv/bin/python scripts/run_local_episode.py --turns 10 --target "light bulb"
```

Artifacts (config, results, replay, per-seat logs, agent workspaces) land in
`runs/<slug>-<timestamp>/`.

## Browse your runs

Each run is a plain directory under `runs/` — `config.json`, `replay.json`,
`results.json`, per-seat `trace-<slot>.jsonl`, logs, agent workspaces, and a
self-contained `report.html`. There is no database.

```bash
.venv/bin/python scripts/serve_runs.py   # http://127.0.0.1:8600/
```

The home page lists every run (canvas thumbnail, best score, pass state) and
links to each run's report: canvas replay, message board, score / token /
time charts, and the full collapsible trace of every agent. Reports are
rendered automatically after agent episodes; the server renders missing ones
on demand, or run `scripts/render_replay_page.py <run-dir>` by hand.

The game server also serves a live viewer during an episode: `/global` for
live state and `/replay` for saved replays (see `docs/global-protocol.md`).

## How the game works

- All seats act in the same turn; the turn resolves when every seat has
  painted (a full barrier).
- If two seats paint the same pixel in the same turn, both writes are
  dropped. Coordination happens on a shared public message board.
- The classifier is a shared noisy sensor. Agents see the target score, its
  delta, and the top predictions after every turn, and must infer the
  scorer's behavior from score changes.
- Two scorers are available. The default is the bundled Quick, Draw!
  nearest-prototype model (10 fixed classes, threshold 0.95). Pass
  `--scorer mobileclip` for apple/MobileCLIP2-S0 zero-shot: any target
  string works (open vocabulary), thresholds are calibrated against real
  human sketches where measured (draw at least as well as the median
  human; light bulb 0.547), and inference auto-selects GPU (CUDA/MPS).
  Needs `uv pip install -e ".[mobileclip]"`; weights (~150 MB) download
  from Hugging Face on first use.
- The recorded team score is the best score ever reached in the episode.

The wire protocol for writing your own player is in
`docs/player-protocol.md`. Any process that can open a WebSocket can play:
connect to `CODRAWING_PLAYER_WS_URL`, read observations, reply with one
pixel per turn.

## Players

| Module | Seat behavior | Needs |
| --- | --- | --- |
| `codrawing.player.agent_player` | Persistent Claude Code session with game tools (paint, board) and a private workspace | `claude` CLI |
| `codrawing.player.cli_player` | One `claude`/`codex` CLI call per turn | `claude` or `codex` CLI |
| `codrawing.player.llm_player` | One direct Anthropic API call per turn | `ANTHROPIC_API_KEY` |
| `codrawing.player.template_player` | Deterministic template drawing, no model | nothing |

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```
