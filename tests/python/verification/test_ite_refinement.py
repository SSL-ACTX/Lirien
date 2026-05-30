from lila import verify, i64, Refined, VerificationError
import unittest

# if x > 0 then x < 10 else x > -10
Bounded = Refined[i64, lambda x: x < 10 if x > 0 else x > -10]


@verify
def take_bounded(x: Bounded) -> i64:
    return x


class TestIteRefinement(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
