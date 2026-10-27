import unittest
from metrics import precision_recall


class MetricsTests(unittest.TestCase):
    def test_normal_case(self):
        self.assertEqual(precision_recall([1, 1, 0, 0], [1, 0, 1, 0]), (0.5, 0.5))

    def test_zero_denominators(self):
        self.assertEqual(precision_recall([0, 0], [0, 0]), (0.0, 0.0))
        self.assertEqual(precision_recall([], []), (0.0, 0.0))

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            precision_recall([1, 0], [1])
