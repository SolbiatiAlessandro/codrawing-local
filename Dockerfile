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


FROM docker.io/library/python:3.12-slim

RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0 \
 && pip install --no-cache-dir \
      fastapi==0.115.5 \
      uvicorn[standard]==0.34.2 \
      websockets==15.0.1 \
      open-clip-torch==3.3.0 \
      pillow==11.3.0

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
