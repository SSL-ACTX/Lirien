import unittest
from lila import verify, i64, Refined
from lila.compiler import VerificationError

Positive = Refined[i64, lambda x: x > 0]


class TestReturnRefinement(unittest.TestCase):
    def test_return_refinement_fails(self):
        # We expect a VerificationError because we try to return x, which might be <= 0
        with self.assertRaises(VerificationError) as ctx:

            @verify
            def should_fail_return(x: i64) -> Positive:
                return x

        self.assertIn("Return refinement", str(ctx.exception))
        self.assertIn("may be violated", str(ctx.exception))

    def test_return_refinement_succeeds(self):
        # We enforce x > 0 before returning
        @verify
        def should_succeed_return(x: i64) -> Positive:
            if x > 0:
                return x
            return 1

        self.assertEqual(should_succeed_return(5), 5)
        self.assertEqual(should_succeed_return(-5), 1)


if __name__ == "__main__":
    unittest.main()
