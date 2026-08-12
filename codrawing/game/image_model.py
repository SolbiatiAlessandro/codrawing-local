from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


MODEL_NAME = "squeezenet1_1_imagenet1k_v1"
QUICKDRAW_MODEL_NAME = "quickdraw_nearest_prototype_v1"
QUICKDRAW_MODEL_PATH = Path(__file__).parent / "models" / "quickdraw_prototypes.json"
MOBILECLIP_MODEL_NAME = "mobileclip2_s0_zeroshot_v1"
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


class MobileClipScorer:
    """Open-vocabulary zero-shot scorer: apple/MobileCLIP2-S0 via open_clip.

    Any target string works — the target is scored against a distractor label
    set with the prompt "a drawing of a <label>". Pass thresholds are the
    median score of 500 real human Quick, Draw! sketches where measured
    (draw at least as well as the median human), else DEFAULT_THRESHOLD.
    Weights (~150 MB) download from Hugging Face on first use. Requires the
    `mobileclip` extra: `uv pip install -e ".[mobileclip]"`.
    """

    PROMPT_TEMPLATE = "a drawing of a {}"
    DEFAULT_LABELS = (
        "light bulb", "hot air balloon", "candle", "flashlight", "lollipop",
        "pear", "sun", "wine glass", "clock", "snowman",
    )
    # Median MobileCLIP2 score of 500 real human sketches (measured 2026-08-11).
    HUMAN_MEDIANS = {"light bulb": 0.547, "candle": 0.964, "pear": 0.999, "sun": 0.987}
    DEFAULT_THRESHOLD = 0.9
    UPSCALE = 384

    def __init__(self, model_name: str = "MobileCLIP2-S0", pretrained: str = "dfndr2b") -> None:
        import open_clip
        import torch

        self.torch = torch
        device = os.environ.get("CODRAWING_MOBILECLIP_DEVICE")
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._text_features: dict[tuple[str, ...], Any] = {}

    def _labels_for(self, target: str) -> tuple[str, ...]:
        labels = tuple(self.DEFAULT_LABELS)
        return labels if target in labels else labels + (target,)

    def _encode_labels(self, labels: tuple[str, ...]) -> Any:
        if labels not in self._text_features:
            text = self.tokenizer([self.PROMPT_TEMPLATE.format(label) for label in labels])
            with self.torch.no_grad():
                features = self.model.encode_text(text.to(self.device))
                features /= features.norm(dim=-1, keepdim=True)
            self._text_features[labels] = features
        return self._text_features[labels]

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

        if len(canvas) != width * height:
            raise ValueError("canvas length does not match its dimensions")
        labels = self._labels_for(target)
        text_features = self._encode_labels(labels)

        pixels = [tuple(bytes.fromhex(color.removeprefix("#"))) for color in canvas]
        image = Image.new("RGB", (width, height))
        image.putdata(pixels)
        # Nearest upscale keeps pixel-art edges crisp for the CLIP preprocessor.
        image = image.resize((self.UPSCALE, self.UPSCALE), Image.Resampling.NEAREST)
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            image_features = self.model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probabilities = (
                (100.0 * image_features @ text_features.T).softmax(dim=-1)[0].cpu().tolist()
            )

        ordered = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
        target_score = probabilities[labels.index(target)]
        threshold = self.HUMAN_MEDIANS.get(target, self.DEFAULT_THRESHOLD)
        delta = 0.0 if previous_score is None else target_score - previous_score

        return {
            "model": MOBILECLIP_MODEL_NAME,
            "turn": turn,
            "target_score": target_score,
            "score_delta": delta,
            "pass_threshold": threshold,
            "passing": target_score > threshold,
            "target_rank": next(
                index for index, item in enumerate(ordered, 1) if item[0] == target
            ),
            "label_count": len(labels),
            "best_target_label": target,
            "top_predictions": [
                {"label": label, "probability": probability}
                for label, probability in ordered[:5]
            ],
        }


JUDGE_MODEL_NAME = "vlm_judge_v1"


class VlmJudgeScorer:
    """Rubric-based judge: a vision LLM scores how much the canvas looks like
    the target, rewarding correct context (a traffic light on a street scores
    above a bare traffic light). Score is monotonic in scene completeness
    instead of saturating at class membership, which also blunts sparse-pixel
    hacks. Uses the local `claude` CLI on subscription auth; one call per turn.
    """

    PASS_THRESHOLD = 0.85
    RUBRIC = (
        "Read the image at {path}. It is a {width}x{height} pixel drawing. "
        "Judge how much it looks like a {target}, 0-100: 0 = nothing recognizable, "
        "40 = the class is guessable but crude, 70 = clearly a {target}, "
        "90+ = a {target} drawn well in a coherent scene with correct context. "
        "Written words in the image do not count as drawing. "
        'Reply with ONLY JSON: {{"score": <int>, "looks_like": ["<top guess>", "<second>", "<third>"]}}'
    )

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("CODRAWING_JUDGE_MODEL", "claude-haiku-4-5-20251001")
        # Single judgments vary a lot on the same canvas; a small median tames it.
        self.samples = max(1, int(os.environ.get("CODRAWING_JUDGE_SAMPLES", "3")))

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
        import subprocess
        import tempfile

        from PIL import Image

        if len(canvas) != width * height:
            raise ValueError("canvas length does not match its dimensions")
        pixels = [tuple(bytes.fromhex(color.removeprefix("#"))) for color in canvas]
        image = Image.new("RGB", (width, height))
        image.putdata(pixels)
        image = image.resize((width * 16, height * 16), Image.Resampling.NEAREST)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image.save(handle.name)
            path = handle.name
        prompt = self.RUBRIC.format(path=path, width=width, height=height, target=target)

        def one_judgment() -> dict[str, Any]:
            completed = subprocess.run(
                ["claude", "-p", prompt, "--model", self.model, "--allowedTools", "Read"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            raw = completed.stdout.strip()
            start, end = raw.find("{"), raw.rfind("}")
            return json.loads(raw[start : end + 1])

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.samples) as pool:
            futures = [pool.submit(one_judgment) for _ in range(self.samples)]
            payloads = []
            for future in futures:
                try:
                    payloads.append(future.result())
                except Exception:
                    pass
        os.unlink(path)
        if not payloads:
            raise RuntimeError("judge produced no valid judgments")
        scores = sorted(int(p["score"]) for p in payloads)
        median = scores[len(scores) // 2]
        target_score = max(0.0, min(1.0, median / 100.0))
        guesses = [str(g) for g in payloads[0].get("looks_like", [])][:3]
        delta = 0.0 if previous_score is None else target_score - previous_score

        return {
            "model": JUDGE_MODEL_NAME,
            "turn": turn,
            "target_score": target_score,
            "score_delta": delta,
            "pass_threshold": self.PASS_THRESHOLD,
            "passing": target_score > self.PASS_THRESHOLD,
            "target_rank": 1 if guesses and guesses[0].lower() == target.lower() else 2,
            "label_count": 0,
            "best_target_label": target,
            "top_predictions": [
                {"label": guess, "probability": target_score if index == 0 else 0.0}
                for index, guess in enumerate(guesses)
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
        use_mobileclip: bool = False,
        use_judge: bool = False,
    ) -> None:
        self._imagenet_model_path = imagenet_model_path
        self._imagenet_labels_path = imagenet_labels_path
        self._imagenet: ImageModelScorer | None = None
        self.quickdraw = QuickDrawScorer(quickdraw_model_path)
        self.mobileclip = MobileClipScorer() if use_mobileclip else None
        self.judge = VlmJudgeScorer() if use_judge else None

    def score(self, *, target: str, **kwargs: Any) -> dict[str, Any]:
        if self.judge is not None:
            return self.judge.score(target=target, **kwargs)
        if self.mobileclip is not None:
            return self.mobileclip.score(target=target, **kwargs)
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
    scorer_name = os.environ.get("CODRAWING_SCORER", "").lower()
    if not model_path and not quickdraw_override and scorer_name not in ("mobileclip", "judge"):
        return None
    labels_path = os.environ.get("CODRAWING_IMAGE_MODEL_LABELS")
    if model_path and not labels_path:
        raise ValueError("CODRAWING_IMAGE_MODEL_LABELS is required when the image model is enabled")
    quickdraw_path = Path(quickdraw_override) if quickdraw_override else QUICKDRAW_MODEL_PATH
    return TargetScorerRouter(
        Path(model_path) if model_path else None,
        Path(labels_path) if labels_path else None,
        quickdraw_path,
        use_mobileclip=scorer_name == "mobileclip",
        use_judge=scorer_name == "judge",
    )
