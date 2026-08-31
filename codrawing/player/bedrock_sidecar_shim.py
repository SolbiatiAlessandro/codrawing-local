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
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHIM_PORT = int(os.environ.get("SIDECAR_SHIM_PORT", "8377"))


def _sidecar_url() -> str:
    base = os.environ["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"].rstrip("/")
    model = os.environ["BEDROCK_MODEL"]
    return f"{base}/model/{model}/invoke"


# The Bedrock InvokeModel request schema is a strict subset of the Anthropic
# API's: newer beta fields the CLI sends (context_management, service tiers,
# ...) are rejected with 400s, so only known-good fields are forwarded.
_BEDROCK_FIELDS = frozenset({
    "max_tokens", "messages", "system", "tools", "tool_choice",
    "temperature", "top_p", "top_k", "stop_sequences", "thinking",
})


def _invoke(body: dict) -> tuple[int, dict]:
    payload = {k: v for k, v in body.items() if k in _BEDROCK_FIELDS}
    payload["anthropic_version"] = "bedrock-2023-05-31"
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
