# Codrawing player protocol

Each player container receives `CODRAWING_PLAYER_WS_URL` and connects to it. The server first sends a welcome frame with the player's zero-based `slot`.

At the start of every turn the server sends an `observation`. It contains the shared target, current canvas, player names, accepted-pixel counts, the latest public messages, and shared image-model feedback from the current canvas:

```json
{
  "type": "observation",
  "slot": 0,
  "target": "cat",
  "turn": 3,
  "max_turns": 50,
  "width": 24,
  "height": 24,
  "canvas": ["#FFFFFF", "#111827"],
  "owners": [-1, 2],
  "player_names": ["Artist 1", "Artist 2", "Artist 3", "Artist 4", "Artist 5"],
  "accepted_pixels": [3, 3, 3, 3, 2],
  "previous_accepted_slots": [0, 1, 2, 3, 4],
  "previous_collision_slots": [],
  "image_model_feedback": {
    "model": "squeezenet1_1_imagenet1k_v1",
    "turn": 3,
    "target_score": 0.0132,
    "score_delta": 0.0011,
    "pass_threshold": 0.5,
    "passing": false,
    "target_rank": 18,
    "best_target_label": "tabby",
    "top_predictions": [
      {"label": "comic book", "probability": 0.08},
      {"label": "envelope", "probability": 0.06},
      {"label": "spotlight", "probability": 0.04},
      {"label": "tabby", "probability": 0.03},
      {"label": "web site", "probability": 0.02}
    ]
  },
  "recent_messages": []
}
```

`target_score` is the probability the image model assigns to the target label. The default local scorer is `quickdraw_nearest_prototype_v1`, a pure-Python nearest-prototype classifier over Quick, Draw! sketch classes bundled at `codrawing/game/models/quickdraw_prototypes.json`; alternative scorers (for example an ImageNet CNN) can be plugged in through the same feedback shape. The team passes only if its final score is strictly greater than `pass_threshold`. `target_rank` is the rank of the target label. The model is a noisy shared sensor, so players should use its changes as evidence while preserving a human-recognizable drawing.

The player replies once for that turn with one public message and one pixel:

```json
{
  "turn": 3,
  "message": "I will outline the left ear.",
  "paint": {"x": 7, "y": 5, "color": "#AAB4C4"}
}
```

Messages may be empty and have a maximum length of 240 characters. Colors use six-digit hex form. The bundled score-aware LLM policy uses its assigned seat color for new paint and `#FFFFFF` to erase a harmful prior experiment. Invalid or late actions are ignored. Actions resolve at the same time; when two or more players choose the same pixel, every write to that pixel is dropped. The next observation reports the accepted and collided slots so a persistent player process can interpret the score change. A slow or disconnected player does not block the episode beyond `action_timeout_seconds`.

The final server frame has `type: "final"`; the player must then exit.
