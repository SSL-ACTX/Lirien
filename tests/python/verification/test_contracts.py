import unittest
from lirien import verify, i64, VerificationError


class TestContracts(unittest.TestCase):
    def test_precondition_success(self):
        @verify
        def add_one(x: i64) -> i64:
            assert 0 < x < 100
            res = x + 1
            assert res > x
            return res

        self.assertEqual(add_one(5), 6)

    def test_precondition_failure_at_call_site(self):
        @verify
        def abs_val(x: i64) -> i64:
            assert x > 0
            return x

        with self.assertRaises(VerificationError):

            @verify
            def caller(y: i64) -> i64:
                # y could be negative or 0, violating precondition of abs_val
                return abs_val(y)

    def test_postcondition_failure(self):
        with self.assertRaises(VerificationError):

            @verify
            def bad_add(x: i64) -> i64:
                res = x + 1
                assert res < x  # Will fail verification
                return res

    def test_loop_invariant_success(self):
        @verify
        def sum_to_n(n: i64) -> i64:
            assert 0 <= n < 100
            total = 0
            i = 0
            while i < n:
                assert i >= 0
                assert total >= 0
                total = total + i
                i = i + 1
            return total

        self.assertEqual(sum_to_n(5), 10)

    def test_loop_invariant_failure(self):
        with self.assertRaises(VerificationError):

            @verify
            def bad_sum(n: i64) -> i64:
                total = 0
                i = 0
                while i < n:
                    assert i > 0  # Fails because on entry i is 0
                    total = total + i
                    i = i + 1
                return total

    def test_custom_precondition_message_runtime(self):
        @verify
        def check_pos(x: i64) -> i64:
            assert x > 0, "x must be positive"
            return x

        with self.assertRaises(AssertionError) as ctx:
            check_pos(-5)
        self.assertEqual(str(ctx.exception), "x must be positive")

    def test_custom_postcondition_message_compile_time(self):
        with self.assertRaises(VerificationError) as ctx:

            @verify
            def bad_add_msg(x: i64) -> i64:
                res = x + 1
                assert res < x, "result must be less than x"
                return res

        self.assertIn("result must be less than x", str(ctx.exception))

    def test_custom_loop_invariant_message_compile_time(self):
        with self.assertRaises(VerificationError) as ctx:

            @verify
            def bad_sum_msg(n: i64) -> i64:
                assert n > 0
                total = 0
                i = 1
                while i < n:
                    assert i == 1, "i must remain 1"
                    total = total + i
                    i = i + 1
                return total

        self.assertIn("i must remain 1", str(ctx.exception))

    def test_intermediate_assert_success(self):
        @verify
        def math_op(x: i64) -> i64:
            assert x > 10
            y = x * 2
            assert y > 20, "y must be greater than 20"
            return y

        self.assertEqual(math_op(11), 22)

    def test_intermediate_assert_failure(self):
        with self.assertRaises(VerificationError) as ctx:

            @verify
            def bad_math_op(x: i64) -> i64:
                assert x > 10
                y = x * 2
                assert y > 30, "y must be greater than 30"
                return y

        self.assertIn("y must be greater than 30", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
