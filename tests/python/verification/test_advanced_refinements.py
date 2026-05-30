from lila import verify, i64, Refined, VerificationError
import unittest

Even = Refined[i64, lambda x: (x & 1) == 0]


@verify
def pass_even(x: Even) -> i64:
    return x


class TestAdvancedRefinements(unittest.TestCase):
    def test_interprocedural_violation(self):
        # We define the failing function INSIDE the test to catch the VerificationError
        # during the @verify decoration process.
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


if __name__ == "__main__":
    unittest.main()
