# SPDX-License-Identifier: AGPL-3.0-only
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sparkscore import answers  # noqa: E402


class Confidence(unittest.TestCase):
    """Values from docs.typesafe.ai/confidence and the formulas of
    system-one-adapter-python 0.2.1 (_utils/confidence_metrics.py)."""

    def test_choice(self):
        self.assertAlmostEqual(answers.choice_confidence([0.8, 0.1, 0.1]), (0.8 - 1 / 3) / (1 - 1 / 3))
        self.assertAlmostEqual(answers.choice_confidence([0.5, 0.5]), 0.0)
        self.assertAlmostEqual(answers.choice_confidence([1.0, 0.0, 0.0, 0.0]), 1.0)
        self.assertEqual(answers.choice_confidence([1.0]), 1.0)

    def test_score_published_examples(self):
        # "With three levels, (0, 0.5, 0.5) has confidence 0.25 ... (0.5, 0, 0.5) has confidence 0."
        self.assertAlmostEqual(answers.score_confidence([0, 0.5, 0.5]), 0.25)
        self.assertAlmostEqual(answers.score_confidence([0.5, 0, 0.5]), 0.0)
        # bug_severity: (0, 0.57, 0.43) -> 1 - 0.43 / (2/3) ~ 0.355
        self.assertAlmostEqual(answers.score_confidence([0, 0.57, 0.43]), 1 - 0.43 / (2 / 3))
        # five levels: MAD_unif is 1.2
        self.assertAlmostEqual(answers.score_confidence([0, 0, 0.9, 0.1, 0]), 1 - 0.1 / 1.2)
        self.assertEqual(answers.score_confidence([1.0]), 1.0)

    def test_score_floor_and_mode_ties(self):
        self.assertEqual(answers.score_confidence([0.5, 0, 0, 0.5]), 0.0)
        # The first most likely level is the mode, as max() picks it in the
        # adapter: level 1 gives 1 - 0.6 / (2/3) = 0.1; level 2 would give 0.
        self.assertAlmostEqual(answers.score_confidence([0.2, 0.4, 0.4]), 0.1)

    def test_inputs_are_normalised_and_zero_totals_are_uniform(self):
        self.assertAlmostEqual(answers.choice_confidence([8, 1, 1]), answers.choice_confidence([0.8, 0.1, 0.1]))
        self.assertAlmostEqual(answers.choice_confidence([0, 0, 0]), 0.0)
        self.assertAlmostEqual(answers.score_confidence([0, 0, 0]), 0.0)


class Answer(unittest.TestCase):
    def test_noul(self):
        a = answers.answer({"type": "noul"}, ["true", "false"], [0.98, 0.02])
        self.assertEqual(a, {"type": "noul", "noul": 0.98})

    def test_choice_first_argmax(self):
        q = {"type": "choice", "criteria": {"x": None, "y": None, "z": None}}
        a = answers.answer(q, ["x", "y", "z"], [0.4, 0.4, 0.2])
        self.assertEqual(a["choice"], "x")
        self.assertEqual(a["probabilities"], {"x": 0.4, "y": 0.4, "z": 0.2})
        self.assertAlmostEqual(a["confidence"], (0.4 - 1 / 3) / (2 / 3))

    def test_score(self):
        q = {"type": "score", "criteria": ["low", {"level": "mid"}, "high"]}
        a = answers.answer(q, ["0", "1", "2"], [0.0, 0.57, 0.43])
        self.assertAlmostEqual(a["score"], 0.57 + 2 * 0.43)
        self.assertEqual(a["legend"], {"0": "low", "1": {"level": "mid"}, "2": "high"})
        self.assertEqual(list(a["probabilities"]), ["0", "1", "2"])
        self.assertAlmostEqual(a["confidence"], 1 - 0.43 / (2 / 3))

    def test_reference_shape(self):
        # The live TypeSafe response recorded in the adapter's test suite.
        q = {"type": "score", "criteria": ["a", "b", "c", "d", "e"]}
        a = answers.answer(q, [str(i) for i in range(5)], [0, 0, 0, 0, 1.0])
        self.assertEqual((a["score"], a["confidence"]), (4.0, 1.0))
        c = answers.answer({"type": "choice", "criteria": {"fiction": "", "nonfiction": ""}}, ["fiction", "nonfiction"], [1.0, 0.0])
        self.assertEqual((c["choice"], c["confidence"]), ("fiction", 1.0))


class Temper(unittest.TestCase):
    def test_identity_and_scaling(self):
        self.assertEqual(answers.temper([0.25, 0.75], 1.0), [0.25, 0.75])
        hot = answers.temper([0.2, 0.8], 2.0)
        self.assertAlmostEqual(hot[1] / hot[0], math.sqrt(4))
        self.assertAlmostEqual(sum(hot), 1.0)
        self.assertEqual(answers.temper([0.0, 1.0], 0.5), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
