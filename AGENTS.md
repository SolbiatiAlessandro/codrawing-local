# Agent instructions for codrawing-local

This repo is a self-contained multi-agent drawing game used as an alignment
research environment (coordination capabilities of LLM agents). See README.md
for how to run it.

## Log findings in the alignment knowledge base (required)

Findings from work in this repo must be logged in the knowledge base at
`../alignment-knowledge-base`. A finding is anything a future session would
want to know: run results, agent behaviors observed in transcripts, game-rule
changes and why, harness gotchas, dead ends.

How to log, following that repo's own AGENTS.md conventions:

1. Update `../alignment-knowledge-base/wiki/concepts/codrawing-local-environment.md`
   — the living state-of-the-environment page (current rules, results ladder,
   behavioral findings, next steps).
2. Prepend an entry to `../alignment-knowledge-base/log.md` (newest at top,
   format `## YYYY-MM-DD HH:MM — title`, time in PT).
3. If the KB has uncommitted changes from another session, edit files but do
   not `git commit` there.

What matters most: **agent behavior over scores.** Alessandro (2026-08-11):
the classifier is a noisy grader, so the score alone means little; the
research signal is in the transcripts — how agents plan, divide work, argue,
recover from collisions, decide to stop. Log behaviors with pointers to the
run directory (`runs/<slug>/`) that shows them.

## Repo orientation

- `codrawing/game/` — server, engine, Quick, Draw! scorer (pure Python).
- `codrawing/player/agent_player.py` — the agent seat: fixed GAME_PROMPT +
  per-seat policy prompt, game tools (paint_pixel, message_board_send,
  message_board_read, complete), trace recording, post-episode interview.
- `codrawing-run` (from `codrawing/cli.py`) — start an episode; `--seats 1`
  is the solo baseline.
- `scripts/serve_runs.py` — home page of all runs at http://127.0.0.1:8600/.
- `runs/<slug>/` — one directory per run: config, replay, results, per-seat
  `trace-<slot>.jsonl`, `report.html`. No database.
- Tests: `.venv/bin/python -m unittest discover -s tests`.
