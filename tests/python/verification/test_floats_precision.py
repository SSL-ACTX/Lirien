import unittest
import math
from lila import verify, f64, i64, Refined
from lila import VerificationError

NonNegative = Refined[f64, lambda x: x >= 0.0]
Positive = Refined[f64, lambda x: x > 0.0]


@verify(log_level="info")
def safe_sqrt(x: NonNegative) -> f64:
    return math.sqrt(x.val)


@verify(log_level="info")
def safe_div(x: f64, y: Positive) -> f64:
    return x / y.val


@verify(log_level="info")
def float_to_int_clip(x: f64) -> i64:
    # Test conversion verification
    return i64(x)


@verify(log_level="info")
def check_precision() -> i64:
    # In IEEE 754, 0.1 + 0.2 is slightly greater than 0.3
    # Real theory would say they are equal.
    # Lila's new Float theory should prove this branch is always taken.
    a = 0.1
    b = 0.2
    if (a + b) > 0.3:
        return 1
    return 0


@verify(log_level="info")
def stability_check(x: f64) -> f64:
    # A simple example where we might want to prove we don't get NaN
    # by ensuring the denominator is not zero.
    denom = x * x + 1.0
    # denom is always >= 1.0, so this is safe.
    return 1.0 / denom


class TestFloatsPrecision(unittest.TestCase):
    def test_safe_ops(self):
        self.assertAlmostEqual(safe_sqrt(4.0), 2.0)
        self.assertAlmostEqual(safe_div(10.0, 2.0), 5.0)

    def test_precision_accuracy(self):
        # Verify that the JIT actually produces the hardware-accurate result
        self.assertEqual(check_precision(), 1)

    def test_stability(self):
        self.assertAlmostEqual(stability_check(2.0), 0.2)

    def test_sqrt_fail(self):
        with self.assertRaises(VerificationError):

            @verify
            def unsafe_sqrt(x: f64) -> f64:
                return math.sqrt(x)

    def test_div_zero_fail(self):
        with self.assertRaises(VerificationError):

            @verify
            def unsafe_div(x: f64, y: f64) -> f64:
                return x / y

    def test_pow_domain_fail(self):
        with self.assertRaises(VerificationError):

            @verify
            def unsafe_pow(x: f64, y: f64) -> f64:
                return math.pow(x, y)


if __name__ == "__main__":
    unittest.main()
