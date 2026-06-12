import unittest
import numpy as np
from lila import verify, i64, Refined, Buffer, struct
from lila import VerificationError

# --- Common Refinement Types ---
Positive = Refined[i64, lambda x: x > 0]
InRange = Refined[i64, lambda x: (x >= 0) & (x < 100)]
Even = Refined[i64, lambda x: (x & 1) == 0]
# if x > 0 then x < 10 else x > -10
Bounded = Refined[i64, lambda x: x < 10 if x > 0 else x > -10]


@struct
class Point:
    x: i64
    y: i64


# --- Helper Functions for Interprocedural Tests ---
@verify
def divide_verified(n: i64, d: Positive) -> i64:
    return n // d


@verify(strict=False)
def divide_unsafe(n: i64, d: i64) -> i64:
    return n // d


@verify
def pass_even(x: Even) -> i64:
    return x


@verify
def take_bounded(x: Bounded) -> i64:
    return x


class TestRefinements(unittest.TestCase):
    # --- Basic Refinements ---

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

    # --- Interprocedural Refinements (from test_advanced_refinements.py) ---

    def test_interprocedural_violation(self):
        with self.assertRaises(VerificationError) as cm:

            @verify
            def call_even_illegal() -> i64:
                return pass_even(3)  # This should FAIL verification

        self.assertIn("Argument refinement violation", str(cm.exception))
        self.assertIn("pass_even", str(cm.exception))

    def test_interprocedural_success(self):
        @verify
        def call_even_safe() -> i64:
            return pass_even(2)

        self.assertEqual(call_even_safe(), 2)

    # --- Return Refinements (from test_return_refinement.py) ---

    def test_return_refinement_fails(self):
        with self.assertRaises(VerificationError) as ctx:

            @verify
            def should_fail_return(x: i64) -> Positive:
                return x

        self.assertIn("Return refinement", str(ctx.exception))
        self.assertIn("may be violated", str(ctx.exception))

    def test_return_refinement_succeeds(self):
        @verify
        def should_succeed_return(x: i64) -> Positive:
            if x > 0:
                return x
            return 1

        self.assertEqual(should_succeed_return(5), 5)
        self.assertEqual(should_succeed_return(-5), 1)

    # --- ITE Refinements (from test_ite_refinement.py) ---

    def test_bounded_ok(self):
        @verify
        def call_bounded_ok() -> i64:
            return take_bounded(5) + take_bounded(-5)

        self.assertEqual(call_bounded_ok(), 0)

    def test_bounded_bad_pos(self):
        with self.assertRaises(VerificationError) as cm:

            @verify
            def call_bounded_bad_pos() -> i64:
                return take_bounded(15)  # Fails x < 10

        self.assertIn("Argument refinement violation", str(cm.exception))

    def test_bounded_bad_neg(self):
        with self.assertRaises(VerificationError) as cm:

            @verify
            def call_bounded_bad_neg() -> i64:
                return take_bounded(-15)  # Fails x > -10

        self.assertIn("Argument refinement violation", str(cm.exception))

    # --- Bitwise Refinements (from test_bitwise_refinement.py) ---

    def test_mask_success(self):
        Masked = Refined[i64, lambda x: (x & 0xFF) == 0xAA]

        @verify
        def check_mask(x: Masked) -> i64:
            return x

        self.assertEqual(check_mask(0xAA), 170)

    def test_mask_failure(self):
        Masked = Refined[i64, lambda x: (x & 0xFF) == 0xAA]

        @verify
        def check_mask(x: Masked) -> i64:
            return x

        with self.assertRaises(ValueError) as cm:
            check_mask(0xBB)
        self.assertIn("Runtime Refinement Violation", str(cm.exception))

    def test_pow2_success(self):
        PowerOfTwo = Refined[i64, lambda x: x > 0 and (x & (x - 1)) == 0]

        @verify
        def is_pow2(x: PowerOfTwo) -> i64:
            return x

        self.assertEqual(is_pow2(1024), 1024)

    def test_bitwise_compilation_failure(self):
        Masked = Refined[i64, lambda x: (x & 0xFF) == 0xAA]

        @verify
        def pass_masked(x: Masked) -> i64:
            return x

        with self.assertRaises(VerificationError) as cm:

            @verify
            def call_masked_bad() -> i64:
                return pass_masked(0xBB)

        self.assertIn("Argument refinement violation", str(cm.exception))

    # --- Path-Aware and Transitive Refinements ---

    def test_transitive_refinement(self):
        # x > 5 implies x > 0
        StrictPositive = Refined[i64, lambda x: x > 5]

        @verify
        def test_transitive(x: StrictPositive) -> i64:
            return divide_verified(100, x)

        self.assertEqual(test_transitive(10), 10)

    def test_path_sensitive_inference(self):
        @verify
        def path_sensitive(x: i64) -> i64:
            if x > 10:
                # x > 10 implies x > 0
                return divide_verified(100, x)
            return 0

        self.assertEqual(path_sensitive(20), 5)
        self.assertEqual(path_sensitive(5), 0)

    def test_arithmetic_property_proof(self):
        @verify
        def sum_positives(a: Positive, b: Positive) -> Positive:
            # Lila should be able to prove that a + b > 0 if a > 0 and b > 0
            return a + b

        self.assertEqual(sum_positives(10, 20), 30)

    def test_nested_struct_refinement(self):
        @struct
        class Inner:
            val: i64

        @struct
        class Outer:
            inner: Inner
            other: i64

        SafeOuter = Refined[Outer, lambda o: o.inner.val > 0]

        @verify
        def get_inner_val(o: SafeOuter) -> i64:
            return o.inner.val

        o = Outer(Inner(42), 10)
        self.assertEqual(get_inner_val(o), 42)

    def test_deeply_nested_struct_refinement(self):
        @struct
        class Inner:
            val: i64

        @struct
        class Outer:
            inner: Inner

        @struct
        class Deep:
            outer: Outer

        SafeDeep = Refined[Deep, lambda d: d.outer.inner.val == 1337]

        @verify
        def get_deep_val(d: SafeDeep) -> i64:
            return d.outer.inner.val

        d = Deep(Outer(Inner(1337)))
        self.assertEqual(get_deep_val(d), 1337)


if __name__ == "__main__":
    unittest.main()
