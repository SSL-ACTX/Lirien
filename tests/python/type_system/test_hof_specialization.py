import unittest
from lirien import verify, i64, Closure
from typing import Callable


@verify
def apply_op(f: Callable[[i64], i64], x: i64) -> i64:
    """Generic HOF to be specialized."""
    return f(x)


@verify
def inc(x: i64) -> i64:
    return x + 1


@verify
def dec(x: i64) -> i64:
    return x - 1


@verify
def square(x: i64) -> i64:
    return x * x


class TestHOFSpecialization(unittest.TestCase):
    def test_static_function_specialization(self):
        """Test that passing top-level functions triggers specialization."""
        # This should generate apply_op_inc
        self.assertEqual(apply_op(inc, 10), 11)
        # This should generate apply_op_dec
        self.assertEqual(apply_op(dec, 10), 9)
        # This should generate apply_op_square
        self.assertEqual(apply_op(square, 5), 25)

    def test_closure_specialization(self):
        """Test that closures are correctly specialized with unique target names."""

        @verify
        def make_adder(x: i64) -> Closure[[i64], i64]:
            return lambda y: x + y

        add5 = make_adder(5)
        # The closure 'add5' should have a unique __name__ (e.g. make_adder_lambda_1)
        # apply_op should specialize to use a direct call to that lambda.
        self.assertEqual(apply_op(add5, 10), 15)

        add100 = make_adder(100)
        self.assertEqual(apply_op(add100, 10), 110)

    def test_nested_hof_specialization(self):
        """Test HOFs calling other HOFs with specialized functions."""

        @verify
        def double_apply(f: Callable[[i64], i64], x: i64) -> i64:
            return apply_op(f, apply_op(f, x))

        # This should propagate 'inc' through two layers of specialized calls
        self.assertEqual(double_apply(inc, 10), 12)
        # This should propagate 'square'
        self.assertEqual(double_apply(square, 3), 81)

    def test_multi_arg_hof(self):
        """Test HOFs with multiple callable arguments."""

        @verify
        def compose_and_apply(
            f: Callable[[i64], i64], g: Callable[[i64], i64], x: i64
        ) -> i64:
            return f(g(x))

        # Should specialize to direct calls for both f and g
        self.assertEqual(
            compose_and_apply(inc, square, 5), 26
        )  # inc(square(5)) = 25 + 1
        self.assertEqual(
            compose_and_apply(square, dec, 5), 16
        )  # square(dec(5)) = 4 * 4

    def test_polymorphic_closure_reuse(self):
        """Test that the same HOF can be reused with different closure instances from the same factory."""

        @verify
        def make_multiplier(factor: i64) -> Closure[[i64], i64]:
            return lambda x: x * factor

        mul2 = make_multiplier(2)
        mul10 = make_multiplier(10)

        # Both closures share the same 'lambda' code but different context.
        # apply_op should specialize to the same lambda wrapper but still work correctly.
        self.assertEqual(apply_op(mul2, 5), 10)
        self.assertEqual(apply_op(mul10, 5), 50)


if __name__ == "__main__":
    unittest.main()
