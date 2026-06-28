import unittest
import math
from lirien import verify, i64, f64, Refined


@verify
def get_coords() -> tuple[i64, i64]:
    return 10, 20


@verify
def swap(x: i64, y: i64) -> tuple[i64, i64]:
    return y, x


NonNegative = Refined[f64, lambda x: x >= 0.0]


@verify(timeout=20000)
def compute_math(x: NonNegative) -> f64:
    return math.sqrt(x.val) + math.sin(x.val) + math.cos(x.val)


Positive = Refined[f64, lambda x: x > 0.0]


@verify(timeout=20000)
def compute_pow(b: Positive, e: f64) -> f64:
    return math.pow(b.val, e)


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

    def test_more_math_intrinsics(self):
        @verify(timeout=20000)
        def more_math(x: f64) -> f64:
            return (
                math.tan(x)
                + math.asin(0.5)
                + math.acos(0.5)
                + math.atan(x)
                + math.exp(x)
                + math.log(x)
                + math.log10(x)
            )

        x = 0.5
        expected = (
            math.tan(x)
            + math.asin(0.5)
            + math.acos(0.5)
            + math.atan(x)
            + math.exp(x)
            + math.log(x)
            + math.log10(x)
        )
        self.assertAlmostEqual(more_math(x), expected)

    def test_rounding_intrinsics(self):
        @verify
        def round_ops(x: f64) -> f64:
            return math.floor(x) + math.ceil(x) + math.trunc(x)

        x = 1.7
        expected = math.floor(x) + math.ceil(x) + math.trunc(x)
        self.assertAlmostEqual(round_ops(x), expected)

    def test_new_scalar_intrinsics(self):
        @verify
        def scalar_ops(a: i64, b: i64) -> i64:
            return abs(a) + min(a, b) + max(a, b)

        # a=-10, b=20
        # abs(-10) = 10
        # min(-10, 20) = -10
        # max(-10, 20) = 20
        # sum = 10 - 10 + 20 = 20
        self.assertEqual(scalar_ops(-10, 20), 20)


if __name__ == "__main__":
    unittest.main()
