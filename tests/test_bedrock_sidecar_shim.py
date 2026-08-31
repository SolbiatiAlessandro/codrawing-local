from __future__ import annotations

import contextlib
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from codrawing.player import bedrock_sidecar_shim

CANNED_MESSAGE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "one pixel at a time"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

# Trimmed from a real Claude Code 2.1.251 request captured inside the agent
# image (the version the npm-unpinned image actually ships). Everything the
# hosted sidecar chokes on is represented: the context_management beta field
# (the v5 failure), thinking {"type": "adaptive", "display": ...} and
# mid-conversation {"role": "system"} messages (the v7 failure), plus
# cache_control placements and a tool_use/tool_result round trip.
CLI_REQUEST = {
    "model": "claude-sonnet-5",
    "max_tokens": 32000,
    "stream": True,
    "system": [{"type": "text", "text": "You are seat 0."}],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Run echo"}]},
        {"role": "system", "content": "Operator note: stay terse."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check something."},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                 "input": {"command": "echo recorded", "description": "Echo a marker"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": False,
                 "content": [{"type": "text", "text": "recorded"}]},
            ],
        },
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Reminder plate.",
                 "cache_control": {"type": "ephemeral"}},
            ],
        },
    ],
    "thinking": {"type": "adaptive", "display": "summarized"},
    "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    "output_config": {"effort": "high"},
    "metadata": {"user_id": "device-and-session-json"},
    "tools": [
        {
            "name": "Bash",
            "description": "Executes a bash command",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "command": {"type": "string", "description": "The command"},
                    "timeout": {"type": "number", "maximum": 600000},
                },
                "required": ["command"],
            },
        },
    ],
}


class StrictSidecar(BaseHTTPRequestHandler):
    """The hosted sidecar's behavior: plain invoke only, and strict shape
    validation at every level of the body, not just the top."""

    protocol_version = "HTTP/1.1"
    seen: list[dict] = []

    TOP_FIELDS = bedrock_sidecar_shim._BEDROCK_FIELDS | {"anthropic_version"}
    BLOCK_FIELDS = {
        "text": {"type", "text", "cache_control"},
        "image": {"type", "source", "cache_control"},
        "tool_use": {"type", "id", "name", "input", "cache_control"},
        "tool_result": {"type", "tool_use_id", "content", "is_error", "cache_control"},
        "thinking": {"type", "thinking", "signature"},
        "redacted_thinking": {"type", "data"},
    }

    def log_message(self, *args) -> None:
        pass

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @classmethod
    def _shape_error(cls, body: dict) -> str | None:
        self = cls
        extras = sorted(set(body) - self.TOP_FIELDS)
        if extras:
            return f"{extras[0]}: Extra inputs are not permitted"
        for block in body.get("system", []) if isinstance(body.get("system"), list) else []:
            if set(block) - self.BLOCK_FIELDS["text"]:
                return "request body does not match the supported Bedrock shape"
        thinking = body.get("thinking")
        if thinking is not None:
            if thinking.get("type") not in ("enabled", "disabled") or set(thinking) - {"type", "budget_tokens"}:
                return "request body does not match the supported Bedrock shape"
        for message in body.get("messages", []):
            if set(message) - {"role", "content"} or message.get("role") not in ("user", "assistant"):
                return "request body does not match the supported Bedrock shape"
            content = message.get("content")
            if isinstance(content, list):
                if not content:
                    return "request body does not match the supported Bedrock shape"
                for block in content:
                    fields = self.BLOCK_FIELDS.get(block.get("type"))
                    if fields is None or set(block) - fields:
                        return "request body does not match the supported Bedrock shape"
                    if isinstance(block.get("cache_control"), dict) and set(block["cache_control"]) - {"type"}:
                        return "request body does not match the supported Bedrock shape"
        for tool in body.get("tools", []):
            if set(tool) - {"name", "description", "input_schema"}:
                return "request body does not match the supported Bedrock shape"
        return None

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StrictSidecar.seen.append(body)
        error = None
        if not self.path.endswith("/invoke"):
            error = "streaming operation is not permitted"
        else:
            error = self._shape_error(body)
        if error:
            self._reply(400, {"type": "error", "error": {"type": "invalid_request_error", "message": error}})
            return
        self._reply(200, CANNED_MESSAGE)


class BedrockSidecarShimTest(unittest.TestCase):
    def setUp(self) -> None:
        StrictSidecar.seen = []
        self.sidecar = ThreadingHTTPServer(("127.0.0.1", 0), StrictSidecar)
        threading.Thread(target=self.sidecar.serve_forever, daemon=True).start()
        self.shim = ThreadingHTTPServer(("127.0.0.1", 0), bedrock_sidecar_shim._Handler)
        threading.Thread(target=self.shim.serve_forever, daemon=True).start()
        environ = {
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": f"http://127.0.0.1:{self.sidecar.server_address[1]}",
            "BEDROCK_MODEL": "us.anthropic.claude-sonnet-4-6",
        }
        patcher = mock.patch.dict("os.environ", environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for server in (self.sidecar, self.shim):
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

    def post_shim(self, body: dict) -> tuple[int, str, bytes]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.shim.server_address[1]}/v1/messages?beta=true",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.headers.get("Content-Type", ""), response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get("Content-Type", ""), error.read()

    def test_sidecar_rejects_the_raw_cli_request(self) -> None:
        # The failure mode this shim exists for: forwarded verbatim, the CLI
        # request 400s (top-level extras first; strip those by hand and the
        # nested shapes still 400).
        request = urllib.request.Request(
            bedrock_sidecar_shim._sidecar_url(),
            data=json.dumps(CLI_REQUEST).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 400)

        top_stripped = {
            k: v for k, v in CLI_REQUEST.items() if k in bedrock_sidecar_shim._BEDROCK_FIELDS
        } | {"anthropic_version": "bedrock-2023-05-31"}
        request = urllib.request.Request(
            bedrock_sidecar_shim._sidecar_url(),
            data=json.dumps(top_stripped).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        detail = json.loads(caught.exception.read())
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(
            detail["error"]["message"],
            "request body does not match the supported Bedrock shape",
        )

    def test_shim_deep_cleans_and_replays_sse(self) -> None:
        status, content_type, payload = self.post_shim(CLI_REQUEST)
        self.assertEqual(status, 200, payload)
        self.assertIn("text/event-stream", content_type)
        forwarded = StrictSidecar.seen[-1]
        self.assertIsNone(StrictSidecar._shape_error(forwarded))
        # Top level: extras gone, version stamped, adaptive thinking dropped.
        self.assertLessEqual(set(forwarded), StrictSidecar.TOP_FIELDS)
        self.assertEqual(forwarded["anthropic_version"], "bedrock-2023-05-31")
        self.assertNotIn("thinking", forwarded)
        # System-role messages became user messages, content intact.
        roles = [m["role"] for m in forwarded["messages"]]
        self.assertEqual(roles, ["user", "user", "assistant", "user", "user"])
        self.assertEqual(forwarded["messages"][1]["content"], "Operator note: stay terse.")
        # cache_control survives where the sidecar accepts it.
        plate = forwarded["messages"][4]["content"][0]
        self.assertEqual(plate["cache_control"], {"type": "ephemeral"})
        # The tool round trip and the free-form input_schema pass untouched.
        self.assertEqual(forwarded["messages"][2]["content"][1]["id"], "toolu_1")
        self.assertEqual(forwarded["messages"][3]["content"][0]["tool_use_id"], "toolu_1")
        self.assertIn("$schema", forwarded["tools"][0]["input_schema"])
        events = [
            json.loads(line[len("data: "):])
            for line in payload.decode().splitlines()
            if line.startswith("data: ")
        ]
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[0], "message_start")
        self.assertEqual(kinds[-1], "message_stop")
        deltas = [e for e in events if e["type"] == "content_block_delta"]
        self.assertEqual(deltas[0]["delta"]["text"], "one pixel at a time")

    def test_explicit_thinking_shapes_still_forward(self) -> None:
        body = {**CLI_REQUEST, "thinking": {"type": "enabled", "budget_tokens": 2048}}
        status, _, payload = self.post_shim(body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            StrictSidecar.seen[-1]["thinking"],
            {"type": "enabled", "budget_tokens": 2048},
        )

    def test_emptied_content_gets_a_placeholder(self) -> None:
        body = {
            **CLI_REQUEST,
            "messages": [
                {"role": "user", "content": [{"type": "unknown_future_block", "data": "x"}]},
            ],
        }
        status, _, payload = self.post_shim(body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            StrictSidecar.seen[-1]["messages"][0]["content"],
            [{"type": "text", "text": "(elided)"}],
        )

    def test_shim_returns_plain_json_when_not_streaming(self) -> None:
        status, content_type, payload = self.post_shim({**CLI_REQUEST, "stream": False})
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(json.loads(payload)["content"], CANNED_MESSAGE["content"])

    def test_sidecar_400_passes_through_and_logs_the_key_signature(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(bedrock_sidecar_shim, "_sidecar_url", lambda: (
            f"http://127.0.0.1:{self.sidecar.server_address[1]}/model/m/invoke-with-response-stream"
        )), contextlib.redirect_stderr(stderr):
            status, _, payload = self.post_shim(CLI_REQUEST)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["message"], "streaming operation is not permitted")
        diagnostic = stderr.getvalue()
        self.assertIn("sidecar-shim: sidecar 400", diagnostic)
        self.assertIn("outbound keys:", diagnostic)
        self.assertIn("messages[].content[].cache_control.type", diagnostic)
        self.assertNotIn("Operator note", diagnostic)  # keys only, never content


if __name__ == "__main__":
    unittest.main()
