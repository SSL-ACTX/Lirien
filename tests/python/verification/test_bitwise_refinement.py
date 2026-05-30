from lila import verify, i64, Refined, VerificationError
import unittest


class TestBitwiseRefinement(unittest.TestCase):
    def test_mask_success(self):
        Masked = Refined[i64, lambda x: (x & 0xFF) == 0xAA]

        @verify
        def check_mask(x: Masked) -> i64:
            return x

        self.assertEqual(check_mask(0xAA), 170)

    def test_mask_failure(self):
        Masked = Refined[i64, lambda x: (x & 0xFF) == 0xAA]

        # This is a RUNTIME violation (calling from Python)
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

    def test_pow2_runtime_failure(self):
        PowerOfTwo = Refined[i64, lambda x: x > 0 and (x & (x - 1)) == 0]

        @verify
        def is_pow2(x: PowerOfTwo) -> i64:
            return x

        with self.assertRaises(ValueError):
            is_pow2(3)

    def test_bitwise_compilation_failure(self):
        # Verify the compiler catches logical bitwise errors during JIT
        Masked = Refined[i64, lambda x: (x & 0xFF) == 0xAA]

        @verify
        def pass_masked(x: Masked) -> i64:
            return x

        with self.assertRaises(VerificationError) as cm:
            @verify
            def call_masked_bad() -> i64:
                return pass_masked(0xBB) # 0xBB & 0xFF != 0xAA
        
        self.assertIn("Argument refinement violation", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
