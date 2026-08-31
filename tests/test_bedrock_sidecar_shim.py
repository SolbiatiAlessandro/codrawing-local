from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from codrawing.player import bedrock_sidecar_shim

# The InvokeModel schema the hosted sidecar enforces (pydantic, extra="forbid").
SIDECAR_FIELDS = bedrock_sidecar_shim._BEDROCK_FIELDS | {"anthropic_version"}

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

# Trimmed from a real Claude Code 2.1.226 request (ANTHROPIC_BASE_URL mode):
# the CLI sends context_management on the first request whenever thinking is
# on, which is exactly the field the sidecar 400s on.
CLI_REQUEST = {
    "model": "claude-sonnet-5",
    "max_tokens": 32000,
    "stream": True,
    "system": [{"type": "text", "text": "You are seat 0."}],
    "messages": [{"role": "user", "content": "Say hi"}],
    "thinking": {"type": "adaptive", "display": "summarized"},
    "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    "output_config": {"effort": "high"},
    "metadata": {"user_id": "device-and-session-json"},
}


class StrictSidecar(BaseHTTPRequestHandler):
    """The hosted sidecar's behavior: plain invoke only, no extra fields."""

    protocol_version = "HTTP/1.1"
    seen: list[dict] = []

    def log_message(self, *args) -> None:
        pass

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StrictSidecar.seen.append(body)
        extras = sorted(set(body) - SIDECAR_FIELDS)
        if not self.path.endswith("/invoke") or extras:
            message = (
                f"{extras[0]}: Extra inputs are not permitted"
                if extras
                else "streaming operation is not permitted"
            )
            self._reply(400, {"type": "error", "error": {"type": "invalid_request_error", "message": message}})
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
        # request 400s on context_management.
        request = urllib.request.Request(
            bedrock_sidecar_shim._sidecar_url(),
            data=json.dumps(CLI_REQUEST).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        detail = json.loads(caught.exception.read())
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(detail["error"]["message"], "context_management: Extra inputs are not permitted")

    def test_shim_strips_unknown_fields_and_replays_sse(self) -> None:
        status, content_type, payload = self.post_shim(CLI_REQUEST)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", content_type)
        forwarded = StrictSidecar.seen[-1]
        self.assertLessEqual(set(forwarded), SIDECAR_FIELDS)
        self.assertEqual(forwarded["anthropic_version"], "bedrock-2023-05-31")
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

    def test_shim_returns_plain_json_when_not_streaming(self) -> None:
        status, content_type, payload = self.post_shim({**CLI_REQUEST, "stream": False})
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(json.loads(payload)["content"], CANNED_MESSAGE["content"])

    def test_sidecar_errors_pass_through_as_errors(self) -> None:
        # A request the sidecar still rejects (no max_tokens -> our fake keeps
        # 200s for valid bodies, so force a 400 by making the shim forward to
        # the streaming path the sidecar refuses).
        with mock.patch.object(bedrock_sidecar_shim, "_sidecar_url", lambda: (
            f"http://127.0.0.1:{self.sidecar.server_address[1]}/model/m/invoke-with-response-stream"
        )):
            status, _, payload = self.post_shim(CLI_REQUEST)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["message"], "streaming operation is not permitted")


if __name__ == "__main__":
    unittest.main()
