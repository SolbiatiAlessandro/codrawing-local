"""Localhost bridge: the claude CLI's streaming Anthropic API on one side,
the hosted Bedrock sidecar's non-streaming InvokeModel on the other.

The sidecar accepts only plain `/model/{id}/invoke` (its gateway rejects the
streaming operation), while the claude CLI always streams. So the CLI is
pointed at this shim via ANTHROPIC_BASE_URL; each /v1/messages request is
forwarded to the sidecar unauthenticated (it re-signs with the runner
identity), and the complete JSON message comes back replayed as the SSE event
sequence the CLI expects. No model credentials exist in the pod at any point.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHIM_PORT = int(os.environ.get("SIDECAR_SHIM_PORT", "8377"))


def _sidecar_url() -> str:
    base = os.environ["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"].rstrip("/")
    model = os.environ["BEDROCK_MODEL"]
    return f"{base}/model/{model}/invoke"


# The sidecar validates the InvokeModel body strictly at EVERY level, not just
# the top: unknown fields nested inside system blocks, content blocks, tools,
# or the thinking config are rejected the same way top-level extras are
# ("request body does not match the supported Bedrock shape"). So the whole
# payload is rebuilt from per-shape whitelists of the documented
# anthropic_version=bedrock-2023-05-31 schema, and anything the CLI adds in a
# future version falls away instead of 400ing every turn.
_BEDROCK_FIELDS = frozenset({
    "max_tokens", "messages", "system", "tools", "tool_choice",
    "temperature", "top_p", "top_k", "stop_sequences", "thinking",
})
_CACHE_CONTROL_FIELDS = frozenset({"type"})
_BLOCK_FIELDS = {
    "text": frozenset({"type", "text", "cache_control"}),
    "image": frozenset({"type", "source", "cache_control"}),
    "tool_use": frozenset({"type", "id", "name", "input", "cache_control"}),
    "tool_result": frozenset({"type", "tool_use_id", "content", "is_error", "cache_control"}),
    "thinking": frozenset({"type", "thinking", "signature"}),
    "redacted_thinking": frozenset({"type", "data"}),
}
_IMAGE_SOURCE_FIELDS = frozenset({"type", "media_type", "data"})
_TOOL_FIELDS = frozenset({"name", "description", "input_schema"})
_TOOL_CHOICE_FIELDS = frozenset({"type", "name", "disable_parallel_tool_use"})


def _clean_block(block: object) -> dict | None:
    """One content block, reduced to its documented fields; unknown kinds drop."""
    if not isinstance(block, dict):
        return None
    kind = block.get("type")
    fields = _BLOCK_FIELDS.get(kind)
    if fields is None:
        return None
    clean = {k: v for k, v in block.items() if k in fields}
    if isinstance(clean.get("cache_control"), dict):
        clean["cache_control"] = {
            k: v for k, v in clean["cache_control"].items() if k in _CACHE_CONTROL_FIELDS
        }
    if kind == "image" and isinstance(clean.get("source"), dict):
        clean["source"] = {k: v for k, v in clean["source"].items() if k in _IMAGE_SOURCE_FIELDS}
    if kind == "tool_result":
        clean["content"] = _clean_content(block.get("content"))
    return clean


def _clean_content(content: object) -> object:
    """Message (or tool_result) content: a plain string, or a block list."""
    if not isinstance(content, list):
        return content
    blocks = [clean for block in content if (clean := _clean_block(block)) is not None]
    # A message whose blocks all fell away must not become empty content —
    # the sidecar rejects that too — so leave a placeholder in the transcript.
    return blocks or [{"type": "text", "text": "(elided)"}]


def _clean_thinking(thinking: object) -> dict | None:
    """The sidecar knows only the documented enabled/disabled shapes. The CLI
    sends {"type": "adaptive", "display": ...}, which is first-party-only
    surface; drop it and let the model run at its Bedrock default."""
    if not isinstance(thinking, dict):
        return None
    if thinking.get("type") == "enabled" and isinstance(thinking.get("budget_tokens"), int):
        return {"type": "enabled", "budget_tokens": thinking["budget_tokens"]}
    if thinking.get("type") == "disabled":
        return {"type": "disabled"}
    return None


def _clean_payload(body: dict) -> dict:
    payload = {k: v for k, v in body.items() if k in _BEDROCK_FIELDS}
    if isinstance(payload.get("system"), list):
        payload["system"] = _clean_content(payload["system"])
    if isinstance(payload.get("messages"), list):
        # The CLI sends mid-conversation {"role": "system"} messages (the
        # AGENT_MODEL alias advertises that capability); Bedrock knows only
        # user/assistant, so anything else is delivered as a user message.
        payload["messages"] = [
            {
                "role": m.get("role") if m.get("role") in ("user", "assistant") else "user",
                "content": _clean_content(m.get("content")),
            }
            for m in payload["messages"]
            if isinstance(m, dict)
        ]
    if isinstance(payload.get("tools"), list):
        # input_schema is free-form JSON Schema and passes through untouched.
        payload["tools"] = [
            {k: v for k, v in tool.items() if k in _TOOL_FIELDS}
            for tool in payload["tools"]
            if isinstance(tool, dict)
        ]
    if isinstance(payload.get("tool_choice"), dict):
        payload["tool_choice"] = {
            k: v for k, v in payload["tool_choice"].items() if k in _TOOL_CHOICE_FIELDS
        }
    thinking = _clean_thinking(payload.pop("thinking", None))
    if thinking is not None:
        payload["thinking"] = thinking
    payload["anthropic_version"] = "bedrock-2023-05-31"
    return payload


def _key_signature(node: object, prefix: str = "") -> set[str]:
    """Every nested dict-key path in a payload — keys only, no values — so a
    hosted 400 names the offending shape straight from the policy log."""
    paths: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            if key != "input_schema":  # tool schemas are free-form noise
                paths.update(_key_signature(value, path))
    elif isinstance(node, list):
        paths.update(path for item in node for path in _key_signature(item, prefix + "[]"))
    return paths


def _invoke(body: dict) -> tuple[int, dict]:
    payload = _clean_payload(body)
    request = urllib.request.Request(
        _sidecar_url(),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        if error.code == 400:
            print(
                f"sidecar-shim: sidecar 400 ({detail[:200]}); outbound keys: "
                + " ".join(sorted(_key_signature(payload))),
                file=sys.stderr,
                flush=True,
            )
        try:
            return error.code, json.loads(detail)
        except json.JSONDecodeError:
            return error.code, {"type": "error", "error": {"type": "api_error", "message": detail[:500]}}


def _sse_events(message: dict):
    """Replay a complete Anthropic message as its streaming event sequence."""
    content = message.get("content") or []
    usage = message.get("usage") or {}
    head = {k: v for k, v in message.items() if k != "content"}
    head["content"] = []
    yield "message_start", {"type": "message_start", "message": head}
    for index, block in enumerate(content):
        if block.get("type") == "tool_use":
            start = {"type": "tool_use", "id": block.get("id"), "name": block.get("name"), "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})}
        elif block.get("type") == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block.get("thinking") or ""}
        else:
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block.get("text") or ""}
        yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": start}
        yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}
        yield "content_block_stop", {"type": "content_block_stop", "index": index}
    yield "message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": message.get("stop_reason"), "stop_sequence": message.get("stop_sequence")},
        "usage": {"output_tokens": usage.get("output_tokens", 0)},
    }
    yield "message_stop", {"type": "message_stop"}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # the game log is for the game
        pass

    def do_POST(self) -> None:
        if not self.path.split("?", 1)[0].endswith("/messages"):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        status, message = _invoke(body)
        if status != 200 or message.get("type") == "error":
            data = json.dumps(message).encode()
            self.send_response(status if status != 200 else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event, payload in _sse_events(message):
                self.wfile.write(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
        else:
            data = json.dumps(message).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def start() -> str:
    """Start the shim in a daemon thread; returns the base URL for the CLI."""
    server = ThreadingHTTPServer(("127.0.0.1", SHIM_PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{SHIM_PORT}"


if __name__ == "__main__":
    print(start(), flush=True)
    threading.Event().wait()
