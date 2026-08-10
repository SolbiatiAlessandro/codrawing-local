from __future__ import annotations

import unittest

from codrawing.game.image_model import (
    PASS_THRESHOLD,
    QUICKDRAW_MODEL_NAME,
    QUICKDRAW_MODEL_PATH,
    QuickDrawScorer,
    TargetScorerRouter,
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


if __name__ == "__main__":
    unittest.main()
