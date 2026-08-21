# The versus coworld image: it serves the game and also runs the baseline
# player, so one image covers both roles in the manifest.
#
# Scoring is MobileCLIP2-S0 only. The composite scorer used for local research
# shells out to the `claude` CLI on subscription auth, which has no place in a
# hosted container; CLIP alone is deterministic and reproducible, which is what
# a league needs anyway.

FROM docker.io/library/python:3.12-slim AS weights

RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0 \
 && pip install --no-cache-dir open-clip-torch==3.3.0

# Pull the weights at build time so an episode never depends on the network.
ENV HF_HOME=/weights
RUN python -c "import open_clip; open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b'); open_clip.get_tokenizer('MobileCLIP2-S0')"


FROM docker.io/library/python:3.12-slim AS game

# One RUN for install + cleanup: files deleted in a later layer would still
# occupy the image, and the platform caps images at 5120 MiB. The deleted
# torch dirs are compile-time material (C++ headers, test suites, protoc)
# that inference never reads.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0 \
 && pip install --no-cache-dir \
      fastapi==0.115.5 \
      uvicorn[standard]==0.34.2 \
      websockets==15.0.1 \
      open-clip-torch==3.3.0 \
      pillow==11.3.0 \
 && SP=/usr/local/lib/python3.12/site-packages \
 && rm -rf "$SP"/torch/include "$SP"/torch/test \
      "$SP"/torch/share "$SP"/torch/utils/benchmark \
      "$SP"/torchgen/packaged \
 && find /usr/local/lib/python3.12 -depth -name __pycache__ -type d -exec rm -rf {} + \
 && python -c "import torch, torchvision, open_clip, fastapi, uvicorn, websockets, PIL; print('imports survive cleanup; torch', torch.__version__)"

COPY --from=weights /weights /weights

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/weights \
    HF_HUB_OFFLINE=1 \
    CODRAWING_SCORER=mobileclip \
    CODRAWING_MOBILECLIP_DEVICE=cpu

WORKDIR /app
COPY codrawing /app/codrawing

CMD ["python", "-m", "codrawing.game.server"]


# The baseline template player: no scoring, no torch — it connects a
# websocket and paints a fixed template, so it ships as its own tiny image
# (player-role images are capped at 512 MiB, far below the game image).
FROM docker.io/library/python:3.12-slim AS player

RUN pip install --no-cache-dir websockets==15.0.1

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY codrawing /app/codrawing

CMD ["python", "-m", "codrawing.player.versus_template_player"]


# The real harness: each seat is a persistent Claude Agent SDK session driving
# the game through MCP tools. The SDK spawns the claude CLI, and the CLI talks
# to whatever ANTHROPIC_BASE_URL points at - so the same code that runs locally
# on a subscription runs hosted against OpenRouter, and AGENT_MODEL picks the
# model. Node comes from the slim node image rather than apt, to stay well
# under the 512 MiB player-image cap.
FROM docker.io/library/node:22-slim AS agent

RUN npm install -g @anthropic-ai/claude-code \
 && npm cache clean --force

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/local/bin/python \
 && pip3 install --break-system-packages --no-cache-dir \
      websockets==15.0.1 "claude-agent-sdk>=0.2" \
 && find /usr/lib/python3* -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    CLAUDE_CONFIG_DIR=/tmp/claude \
    AGENT_WORKSPACE=/tmp/agent-workspace \
    DISABLE_TELEMETRY=1 \
    DISABLE_AUTOUPDATER=1

WORKDIR /app
COPY codrawing /app/codrawing

CMD ["python3", "-m", "codrawing.player.agent_player"]
