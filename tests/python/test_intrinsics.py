import unittest
from typing import Tuple
import math
from lila import verify, i64, f64


@verify
def get_coords() -> Tuple[i64, i64]:
    return 10, 20


@verify
def swap(x: i64, y: i64) -> Tuple[i64, i64]:
    return y, x


@verify
def compute_math(x: f64) -> f64:
    return math.sqrt(x) + math.sin(x) + math.cos(x)


@verify
def compute_pow(b: f64, e: f64) -> f64:
    return math.pow(b, e)


class TestNewIntrinsics(unittest.TestCase):
    def test_tuples(self):
        x, y = get_coords()
        self.assertEqual(x, 10)
        self.assertEqual(y, 20)

        a, b = swap(1, 2)
        self.assertEqual(a, 2)
        self.assertEqual(b, 1)

    def test_math_intrinsics(self):
        x = 1.0
        expected = math.sqrt(x) + math.sin(x) + math.cos(x)
        res = compute_math(x)
        self.assertAlmostEqual(res, expected)

        self.assertAlmostEqual(compute_pow(2.0, 3.0), 8.0)


if __name__ == "__main__":
    unittest.main()
