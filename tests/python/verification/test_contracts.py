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


if __name__ == "__main__":
    unittest.main()
