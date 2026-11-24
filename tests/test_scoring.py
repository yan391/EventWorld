import unittest

import numpy as np

from eventworld.assets import HEIGHT, WIDTH
from eventworld.scoring import aggregate, score_depth


class EventScapeScoringTests(unittest.TestCase):
    def test_fit_uses_the_lower_median_on_the_non_sky_30m_support(self):
        reference = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
        semantics = np.zeros_like(reference, dtype=np.uint8)
        prediction = np.ones_like(reference)
        reference[0, :50] = 2.0
        reference[0, 50:100] = 8.0
        semantics[0, :100] = 1
        reference[0, 100:200] = 31.0
        semantics[0, 100:200] = 1
        reference[0, 200:400] = 1.0

        result = score_depth(prediction, reference, semantics)
        self.assertEqual(result["pixels"], 100)
        self.assertEqual(result["scale"], 2.0)
        self.assertEqual(result["absrel"], 0.375)
        self.assertEqual(result["delta1"], 0.5)

    def test_recordings_have_equal_weight(self):
        rows = [
            {"horizon_ms": 200, "recording_group": "a", "absrel": 0.0, "delta1": 1.0},
            {"horizon_ms": 200, "recording_group": "a", "absrel": 0.0, "delta1": 1.0},
            {"horizon_ms": 200, "recording_group": "b", "absrel": 1.0, "delta1": 0.0},
        ]
        self.assertEqual(aggregate(rows), [
            {"horizon_ms": 200, "recordings": 2, "absrel": 0.5, "delta1": 0.5},
        ])
