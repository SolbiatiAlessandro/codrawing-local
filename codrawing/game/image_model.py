from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


MODEL_NAME = "squeezenet1_1_imagenet1k_v1"
QUICKDRAW_MODEL_NAME = "quickdraw_nearest_prototype_v1"
QUICKDRAW_MODEL_PATH = Path(__file__).parent / "models" / "quickdraw_prototypes.json"
PASS_THRESHOLD = 0.5

# ILSVRC-2012 indices in TorchVision's canonical category order. A group score
# is useful here because ImageNet splits cats, dogs, and elephants into breeds.
TARGET_INDICES = {
    "cat": tuple(range(281, 286)),
    "dog": tuple(range(151, 269)),
    "elephant": (101, 385, 386),
}


class ImageModelScorer:
    """Small ImageNet classifier used as shared, per-turn team feedback."""

    def __init__(self, model_path: Path, labels_path: Path) -> None:
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = json.loads(labels_path.read_text())
        if len(self.labels) != 1000:
            raise ValueError("image model labels must contain 1000 ImageNet classes")

    def score(
        self,
        *,
        canvas: list[str],
        width: int,
        height: int,
        target: str,
        turn: int,
        previous_score: float | None,
    ) -> dict[str, Any]:
        from PIL import Image

        if target not in TARGET_INDICES:
            raise ValueError(f"image model has no target mapping for {target!r}")
        if len(canvas) != width * height:
            raise ValueError("canvas length does not match its dimensions")

        pixels = [tuple(bytes.fromhex(color.removeprefix("#"))) for color in canvas]
        image = Image.new("RGB", (width, height))
        image.putdata(pixels)
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        image = image.crop((16, 16, 240, 240))

        array = self.np.asarray(image, dtype=self.np.float32) / 255.0
        array = (array - self.np.array([0.485, 0.456, 0.406], dtype=self.np.float32)) / self.np.array(
            [0.229, 0.224, 0.225], dtype=self.np.float32
        )
        batch = self.np.transpose(array, (2, 0, 1))[None, ...]
        logits = self.session.run(None, {self.input_name: batch})[0][0]
        probabilities = self.np.exp(logits - logits.max())
        probabilities /= probabilities.sum()

        target_indices = TARGET_INDICES[target]
        target_score = float(probabilities[list(target_indices)].sum())
        best_target_index = max(target_indices, key=lambda index: float(probabilities[index]))
        ordered = self.np.argsort(probabilities)[::-1]
        target_rank = int(self.np.where(ordered == best_target_index)[0][0]) + 1
        top_indices = ordered[:5]
        delta = 0.0 if previous_score is None else target_score - previous_score

        return {
            "model": MODEL_NAME,
            "turn": turn,
            "target_score": target_score,
            "score_delta": delta,
            "pass_threshold": PASS_THRESHOLD,
            "passing": target_score > PASS_THRESHOLD,
            "target_rank": target_rank,
            "label_count": len(self.labels),
            "best_target_label": self.labels[best_target_index],
            "top_predictions": [
                {
                    "label": self.labels[int(index)],
                    "probability": float(probabilities[int(index)]),
                }
                for index in top_indices
            ],
        }


class QuickDrawScorer:
    """Color-blind shape classifier over learned Quick, Draw! prototypes.

    The occupied (non-white) pixels are cropped, rescaled into a centered
    28x28 mask, and compared with per-class mean bitmaps; a softmax over the
    distances yields the score. Pure Python, no ONNX or numpy required.
    """

    MODEL_SIZE = 28
    DRAWING_SIZE = 22

    def __init__(self, model_path: Path = QUICKDRAW_MODEL_PATH) -> None:
        payload = json.loads(model_path.read_text())
        self.classes = tuple(payload["classes"])
        self.temperature = float(payload["temperature"])
        self.pass_threshold = float(payload.get("pass_threshold", 0.95))
        self.prototypes = {
            label: tuple(value / 255.0 for value in payload["prototypes"][label])
            for label in self.classes
        }
        if any(
            len(values) != self.MODEL_SIZE * self.MODEL_SIZE
            for values in self.prototypes.values()
        ):
            raise ValueError("every model prototype must be 28x28")

    def _mask(self, canvas: list[str], width: int, height: int) -> list[float]:
        occupied = [
            (index % width, index // width)
            for index, color in enumerate(canvas)
            if color.upper() != "#FFFFFF"
        ]
        output = [0.0] * (self.MODEL_SIZE * self.MODEL_SIZE)
        if not occupied:
            return output

        min_x = min(x for x, _ in occupied)
        max_x = max(x for x, _ in occupied)
        min_y = min(y for _, y in occupied)
        max_y = max(y for _, y in occupied)
        source_width = max_x - min_x + 1
        source_height = max_y - min_y + 1
        scale = min(self.DRAWING_SIZE / source_width, self.DRAWING_SIZE / source_height)
        scaled_width = max(1, round(source_width * scale))
        scaled_height = max(1, round(source_height * scale))
        offset_x = (self.MODEL_SIZE - scaled_width) // 2
        offset_y = (self.MODEL_SIZE - scaled_height) // 2

        for x, y in occupied:
            target_x = offset_x + min(scaled_width - 1, int((x - min_x) * scale))
            target_y = offset_y + min(scaled_height - 1, int((y - min_y) * scale))
            output[target_y * self.MODEL_SIZE + target_x] = 1.0
        return output

    def score(
        self,
        *,
        canvas: list[str],
        width: int,
        height: int,
        target: str,
        turn: int,
        previous_score: float | None,
    ) -> dict[str, Any]:
        if target not in self.classes:
            raise ValueError(f"image model has no target mapping for {target!r}")
        if len(canvas) != width * height:
            raise ValueError("canvas length does not match its dimensions")

        mask = self._mask(canvas, width, height)
        distances = {
            label: sum((value - expected) ** 2 for value, expected in zip(mask, prototype))
            / len(mask)
            for label, prototype in self.prototypes.items()
        }
        minimum = min(distances.values())
        weights = {
            label: math.exp(-(distance - minimum) / self.temperature)
            for label, distance in distances.items()
        }
        total = sum(weights.values())
        probabilities = {label: weight / total for label, weight in weights.items()}
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        target_score = probabilities[target]
        delta = 0.0 if previous_score is None else target_score - previous_score

        return {
            "model": QUICKDRAW_MODEL_NAME,
            "turn": turn,
            "target_score": target_score,
            "score_delta": delta,
            "pass_threshold": self.pass_threshold,
            "passing": target_score > self.pass_threshold,
            "target_rank": next(
                index for index, item in enumerate(ordered, 1) if item[0] == target
            ),
            "label_count": len(self.classes),
            "best_target_label": target,
            "top_predictions": [
                {"label": label, "probability": probability}
                for label, probability in ordered[:5]
            ],
        }


class TargetScorerRouter:
    """Route each target to the classifier that knows it.

    ImageNet targets (cat, dog, elephant) use SqueezeNet; Quick, Draw! targets
    (light bulb) use the prototype scorer. SqueezeNet loads lazily so episodes
    with a Quick, Draw! target never touch ONNX Runtime.
    """

    def __init__(
        self,
        imagenet_model_path: Path | None,
        imagenet_labels_path: Path | None,
        quickdraw_model_path: Path = QUICKDRAW_MODEL_PATH,
    ) -> None:
        self._imagenet_model_path = imagenet_model_path
        self._imagenet_labels_path = imagenet_labels_path
        self._imagenet: ImageModelScorer | None = None
        self.quickdraw = QuickDrawScorer(quickdraw_model_path)

    def score(self, *, target: str, **kwargs: Any) -> dict[str, Any]:
        if target in TARGET_INDICES:
            if self._imagenet is None:
                if self._imagenet_model_path is None or self._imagenet_labels_path is None:
                    raise ValueError(
                        "CODRAWING_IMAGE_MODEL is required for ImageNet targets"
                    )
                self._imagenet = ImageModelScorer(
                    self._imagenet_model_path, self._imagenet_labels_path
                )
            return self._imagenet.score(target=target, **kwargs)
        if target in self.quickdraw.classes:
            return self.quickdraw.score(target=target, **kwargs)
        raise ValueError(f"image model has no target mapping for {target!r}")


def scorer_from_environment() -> TargetScorerRouter | None:
    model_path = os.environ.get("CODRAWING_IMAGE_MODEL")
    quickdraw_override = os.environ.get("CODRAWING_QUICKDRAW_MODEL")
    if not model_path and not quickdraw_override:
        return None
    labels_path = os.environ.get("CODRAWING_IMAGE_MODEL_LABELS")
    if model_path and not labels_path:
        raise ValueError("CODRAWING_IMAGE_MODEL_LABELS is required when the image model is enabled")
    quickdraw_path = Path(quickdraw_override) if quickdraw_override else QUICKDRAW_MODEL_PATH
    return TargetScorerRouter(
        Path(model_path) if model_path else None,
        Path(labels_path) if labels_path else None,
        quickdraw_path,
    )
