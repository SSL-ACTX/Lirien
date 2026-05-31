import unittest
from lila import verify, i64, Refined
from lila.compiler import VerificationError


class TestRecursion(unittest.TestCase):
    def test_recursive_sum(self):
        # Summing 0 to n
        SmallPos = Refined[i64, lambda x: (0 <= x) & (x <= 1000)]
        Positive = Refined[i64, lambda x: 0 <= x]

        @verify
        def recursive_sum(n: SmallPos) -> Positive:
            if n <= 0:
                return 0
            return n + recursive_sum(n - 1)

        self.assertEqual(recursive_sum(5), 15)
        self.assertEqual(recursive_sum(0), 0)

    def test_factorial(self):
        # Factorial: n!
        # n must be >= 0, and result is always >= 1
        SmallPos = Refined[i64, lambda x: (0 <= x) & (x <= 20)]  # 20! fits in i64
        StrictPositive = Refined[i64, lambda x: x >= 1]

        @verify
        def factorial(n: SmallPos) -> StrictPositive:
            if n <= 1:
                return 1
            return n * factorial(n - 1)

        self.assertEqual(factorial(5), 120)
        self.assertEqual(factorial(0), 1)

    def test_fibonacci(self):
        # Fibonacci: Multiple recursive calls
        # n >= 0, result >= 0
        SmallPos = Refined[i64, lambda x: (0 <= x) & (x <= 40)]
        Positive = Refined[i64, lambda x: 0 <= x]

        @verify
        def fib(n: SmallPos) -> Positive:
            if n <= 0:
                return 0
            if n == 1:
                return 1
            return fib(n - 1) + fib(n - 2)

        self.assertEqual(fib(10), 55)
        self.assertEqual(fib(0), 0)

    def test_recursive_sum_violation(self):
        # Should fail: returns 0 but type says > 0
        SmallPos = Refined[i64, lambda x: (0 <= x) & (x <= 1000)]
        StrictPositive = Refined[i64, lambda x: x > 0]

        @verify(strict=False)
        def recursive_sum_fail(n: SmallPos) -> StrictPositive:
            if n <= 0:
                return 0
            return n + recursive_sum_fail(n - 1)

        self.assertFalse(getattr(recursive_sum_fail, "__lila_jit__", False))


if __name__ == "__main__":
    unittest.main()
