import unittest
import numpy as np
from lila import verify, i64, Refined, Buffer, struct
from lila.compiler import VerificationError

Positive = Refined[i64, lambda x: x > 0]
InRange = Refined[i64, lambda x: (x >= 0) & (x < 100)]


@verify
def divide_verified(n: i64, d: Positive) -> i64:
    return n // d


@verify(strict=False)
def divide_unsafe(n: i64, d: i64) -> i64:
    return n // d


@struct
class Point:
    x: i64
    y: i64


class TestRefinements(unittest.TestCase):
    def test_safe_division(self):
        res = divide_verified(100, 10)
        self.assertEqual(res, 10)

    def test_unsafe_fallback(self):
        # This should fail verification and fall back to Python
        # Because d is not guaranteed to be non-zero
        self.assertFalse(getattr(divide_unsafe, "__lila_jit__", True))
        with self.assertRaises(ZeroDivisionError):
            divide_unsafe(100, 0)

    def test_refined_alias(self):
        @verify
        def test_in_range(x: InRange) -> i64:
            return x + 1

        self.assertTrue(getattr(test_in_range, "__lila_jit__", False))
        self.assertEqual(test_in_range(50), 51)

    def test_buffer_len_refinement(self):
        @verify
        def head(buf: Buffer[i64]) -> i64:
            if len(buf) > 0:
                return buf[0]
            return -1

        data = np.array([42, 43], dtype=np.int64)
        self.assertEqual(head(data), 42)

        empty = np.array([], dtype=np.int64)
        self.assertEqual(head(empty), -1)

    def test_buffer_len_fail(self):
        with self.assertRaises(VerificationError):

            @verify
            def unsafe_head(buf: Buffer[i64]) -> i64:
                return buf[0]

    def test_liquid_buffer(self):
        NonEmptyBuffer = Refined[Buffer[i64], lambda x: x > 0]

        @verify
        def first_elt(buf: NonEmptyBuffer) -> i64:
            return buf[0]

        data = np.array([1337], dtype=np.int64)
        self.assertEqual(first_elt(data), 1337)

    def test_complex_predicate(self):
        # Test complex nested logic and arithmetic in refinements
        VerySpecific = Refined[i64, lambda x: (x > 0) and (x < 100) and (x % 2 == 0)]

        @verify
        def test_specific(x: VerySpecific) -> i64:
            # Lila knows x is positive and non-zero
            return 100 // x

        self.assertEqual(test_specific(50), 2)

    def test_struct_field_refinement(self):
        # Test refinement on a struct field
        SafePoint = Refined[Point, lambda p: p.x > 0]

        @verify
        def get_inverse_x(p: SafePoint) -> i64:
            return 100 // p.x

        p = Point(10, 20)
        self.assertEqual(get_inverse_x(p), 10)


if __name__ == "__main__":
    unittest.main()
