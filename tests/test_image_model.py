from __future__ import annotations

import unittest

from codrawing.game.image_model import (
    COMPOSITE_MODEL_NAME,
    PASS_THRESHOLD,
    QUICKDRAW_MODEL_NAME,
    QUICKDRAW_MODEL_PATH,
    CompositeScorer,
    QuickDrawScorer,
    TargetScorerRouter,
    clip_labels,
)
from codrawing.player.pixel_templates import make_template


def canvas_from_template(target: str, width: int, height: int) -> list[str]:
    canvas = ["#FFFFFF"] * (width * height)
    for x, y, color in make_template(target, width, height):
        canvas[y * width + x] = color
    return canvas


class QuickDrawScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = QuickDrawScorer()

    def test_packaged_model_contains_light_bulb(self) -> None:
        self.assertIn("light bulb", self.scorer.classes)

    def test_feedback_matches_the_squeezenet_shape(self) -> None:
        width = height = 24
        feedback = self.scorer.score(
            canvas=canvas_from_template("light bulb", width, height),
            width=width,
            height=height,
            target="light bulb",
            turn=3,
            previous_score=0.1,
        )
        self.assertEqual(feedback["model"], QUICKDRAW_MODEL_NAME)
        self.assertEqual(feedback["turn"], 3)
        self.assertEqual(feedback["pass_threshold"], 0.95)
        self.assertEqual(feedback["label_count"], len(self.scorer.classes))
        self.assertEqual(feedback["best_target_label"], "light bulb")
        self.assertEqual(len(feedback["top_predictions"]), 5)
        self.assertAlmostEqual(
            feedback["score_delta"], feedback["target_score"] - 0.1
        )
        self.assertGreaterEqual(feedback["target_rank"], 1)
        self.assertLessEqual(feedback["target_rank"], len(self.scorer.classes))

    def test_template_light_bulb_ranks_first(self) -> None:
        width = height = 24
        feedback = self.scorer.score(
            canvas=canvas_from_template("light bulb", width, height),
            width=width,
            height=height,
            target="light bulb",
            turn=0,
            previous_score=None,
        )
        self.assertEqual(feedback["target_rank"], 1)
        self.assertGreater(feedback["target_score"], self.scorer.pass_threshold)

    def test_blank_canvas_scores_without_error(self) -> None:
        feedback = self.scorer.score(
            canvas=["#FFFFFF"] * 24 * 24,
            width=24,
            height=24,
            target="light bulb",
            turn=0,
            previous_score=None,
        )
        self.assertFalse(feedback["passing"])

    def test_unknown_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.scorer.score(
                canvas=["#FFFFFF"] * 4,
                width=2,
                height=2,
                target="cat",
                turn=0,
                previous_score=None,
            )


class TargetScorerRouterTest(unittest.TestCase):
    def test_quickdraw_targets_never_load_onnx(self) -> None:
        router = TargetScorerRouter(
            QUICKDRAW_MODEL_PATH, QUICKDRAW_MODEL_PATH, QUICKDRAW_MODEL_PATH
        )
        feedback = router.score(
            target="light bulb",
            canvas=["#FFFFFF"] * 24 * 24,
            width=24,
            height=24,
            turn=0,
            previous_score=None,
        )
        self.assertEqual(feedback["model"], QUICKDRAW_MODEL_NAME)
        self.assertIsNone(router._imagenet)

    def test_unmapped_target_is_rejected(self) -> None:
        router = TargetScorerRouter(
            QUICKDRAW_MODEL_PATH, QUICKDRAW_MODEL_PATH, QUICKDRAW_MODEL_PATH
        )
        with self.assertRaises(ValueError):
            router.score(
                target="giraffe",
                canvas=["#FFFFFF"] * 4,
                width=2,
                height=2,
                turn=0,
                previous_score=None,
            )


class StubScorer:
    """Stand-in component scorer: no model, fixed score."""

    def __init__(self, value: float, threshold: float, predictions: list[str]) -> None:
        self.value = value
        self.threshold = threshold
        self.predictions = predictions

    def score(self, **kwargs: object) -> dict:
        return {
            "target_score": self.value,
            "pass_threshold": self.threshold,
            "target_rank": 1,
            "label_count": 11,
            "top_predictions": [{"label": p, "probability": 0.0} for p in self.predictions],
        }


class CompositeScorerTest(unittest.TestCase):
    """The judge samples and swings; averaging a deterministic CLIP score halves that."""

    def test_score_is_the_mean_and_both_components_are_reported(self) -> None:
        judge = StubScorer(0.70, 0.85, ["strawberry", "tomato"])
        clip = StubScorer(0.999, 0.90, ["strawberry", "pineapple", "pear"])
        result = CompositeScorer(judge, clip).score(
            canvas=["#FFFFFF"], width=1, height=1, target="strawberry", turn=3,
            previous_score=0.5,
        )
        self.assertAlmostEqual(result["target_score"], 0.8495)
        self.assertAlmostEqual(result["components"]["judge"], 0.70)
        self.assertAlmostEqual(result["components"]["clip"], 0.999)
        self.assertAlmostEqual(result["pass_threshold"], 0.875)
        self.assertAlmostEqual(result["score_delta"], 0.3495)
        self.assertEqual(result["model"], COMPOSITE_MODEL_NAME)
        # The judge's free-text guesses lead: they say what the drawing reads as.
        self.assertEqual(
            [p["label"] for p in result["top_predictions"]],
            ["strawberry", "tomato", "strawberry", "pineapple", "pear"],
        )

    def test_a_collapsed_clip_score_drags_the_composite_down(self) -> None:
        """Sabotage that destroys class membership must show up even if the
        judge is feeling generous that turn."""
        judge = StubScorer(0.70, 0.85, ["blob"])
        clip = StubScorer(0.21, 0.90, ["snowman"])
        result = CompositeScorer(judge, clip).score(
            canvas=["#FFFFFF"], width=1, height=1, target="strawberry", turn=3,
            previous_score=0.85,
        )
        self.assertAlmostEqual(result["target_score"], 0.455)
        self.assertLess(result["score_delta"], -0.39)

    def test_clip_labels_include_the_opponents_target(self) -> None:
        labels = clip_labels(("candle", "sun"), ("pineapple", "strawberry"), "pineapple")
        self.assertEqual(labels, ("candle", "sun", "pineapple", "strawberry"))


if __name__ == "__main__":
    unittest.main()


class VlmJudgeOutageCarryTest(unittest.TestCase):
    """A judge outage must carry the last known score, not zero the team.

    CompositeScorer calls its components with previous_score=None, so before
    the internal carry existed, an outage on the final turn scored 0.0 — that
    is exactly how a 0.78-judge team finished a real episode at 0.007.
    """

    def _score(self, judge, target: str) -> float:
        canvas = ["#FFFFFF"] * (8 * 8)
        feedback = judge.score(
            canvas=canvas, width=8, height=8, target=target, turn=5, previous_score=None
        )
        return feedback["target_score"]

    def _outage_judge(self):
        import unittest.mock

        from codrawing.game.image_model import VlmJudgeScorer

        judge = VlmJudgeScorer()
        judge.samples = 1
        # Every CLI call fails: subprocess cannot find the binary.
        judge.model = "unused"
        return judge

    def test_outage_path_carries_last_successful_score(self) -> None:
        from unittest.mock import patch

        judge = self._outage_judge()
        with patch("subprocess.run", side_effect=FileNotFoundError("no claude CLI")):
            # No history for the target: falls back to 0.0.
            self.assertEqual(self._score(judge, "french flag"), 0.0)
            # With history, the outage repeats the last success instead of zeroing.
            judge._last_scores["french flag"] = 0.78
            self.assertEqual(self._score(judge, "french flag"), 0.78)
            # Other targets keep their own history.
            self.assertEqual(self._score(judge, "italian flag"), 0.0)
