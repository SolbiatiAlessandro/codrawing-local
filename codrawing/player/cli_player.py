"""Local-only seat driven by the `claude` or `codex` CLI.

Reuses the hosted LLM player's prompt, memory, and validation, but obtains
each decision from a locally installed agent CLI instead of the Anthropic
API, so a five-agent episode can run on a laptop without API credentials.
Hosted evidence runs keep using `codrawing.player.llm_player`.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast

import websockets

from codrawing.player.llm_player import (
    AgentMemory,
    enforce_seat_color,
    extract_action,
    normalize_decision,
    prompt_for,
    validate_decision,
)

JSON_INSTRUCTION = """
IMPORTANT OVERRIDE: no tools are available in this session. Do not attempt any tool call and do not run
commands. Reply with exactly one JSON object of the form
{"message": "<public message, max 240 chars>", "paint": {"x": <int>, "y": <int>, "color": "#RRGGBB"}}
and nothing else: no prose, no code fences.
"""


def build_command(provider: str, model: str) -> list[str]:
    if provider == "claude":
        return ["claude", "--print", "--output-format", "json", "--model", model]
    if provider == "codex":
        return [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-m",
            model,
        ]
    raise ValueError(f"unsupported CLI provider {provider!r}")


def parse_output(provider: str, output: str) -> str:
    if provider == "claude":
        return str(json.loads(output)["result"])
    message: str | None = None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            message = str(item.get("text", ""))
    if message is None:
        raise ValueError("codex output contained no agent message")
    return message


async def call_cli(provider: str, model: str, prompt: str, timeout: float) -> str:
    process = await asyncio.create_subprocess_exec(
        *build_command(provider, model),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(prompt.encode()), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        raise
    if process.returncode != 0:
        raise RuntimeError(f"{provider} CLI failed: {stderr.decode()[-400:]}")
    return parse_output(provider, stdout.decode())


async def main() -> None:
    url = os.environ["CODRAWING_PLAYER_WS_URL"]
    provider = os.environ.get("CLI_PROVIDER", "claude")
    model = os.environ["CLI_MODEL"]
    timeout = float(os.environ.get("MODEL_TIMEOUT_SECONDS", "90"))
    max_attempts = int(os.environ.get("MODEL_MAX_ATTEMPTS", "2"))
    memory = AgentMemory()
    async with websockets.connect(url, max_size=None) as websocket:
        slot: int | None = None
        while True:
            # Slow CLI calls leave newer observations queued; act only on the
            # latest one so we never submit for an already-resolved turn.
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
            prompt = prompt_for(observation, slot, memory) + JSON_INSTRUCTION
            decision: dict[str, Any] | None = None
            for attempt in range(max_attempts):
                try:
                    text = await call_cli(provider, model, prompt, timeout)
                    decision = extract_action(text)
                    normalize_decision(decision)
                    enforce_seat_color(decision, slot)
                    validate_decision(decision, observation)
                    break
                except Exception as exc:
                    decision = None
                    print(
                        f"cli attempt {attempt + 1}/{max_attempts} failed on turn "
                        f"{observation['turn']}: {exc}",
                        flush=True,
                    )
            if decision is None:
                continue
            print(
                json.dumps(
                    {
                        "event": "llm_action",
                        "slot": slot,
                        "turn": observation["turn"],
                        "provider": provider,
                        "model": model,
                    }
                ),
                flush=True,
            )
            decision["turn"] = observation["turn"]
            memory.remember_action(decision)
            await websocket.send(json.dumps(decision))


if __name__ == "__main__":
    asyncio.run(main())
