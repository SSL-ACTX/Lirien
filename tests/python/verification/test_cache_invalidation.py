import unittest
from typing import TypeVar
from lirien import verify, i64, Refined, VerificationError

T = TypeVar("T")


class TestCacheInvalidation(unittest.TestCase):
    def test_dependency_change_invalidates_cache(self):
        # 1. Define refinement types
        Positive = Refined[i64, lambda x: x > 0]
        Even = Refined[i64, lambda x: (x & 1) == 0]

        # 2. Define a dependency function g with Positive refinement
        @verify
        def g_dep(x: Positive) -> i64:
            return x

        # 3. Define a generic caller function f that calls g_dep with a positive value
        @verify
        def f_caller(t: T) -> i64:
            return g_dep(5)

        # Execute caller to compile and populate the cache
        self.assertEqual(f_caller(1), 5)

        # Clear Python-side MonomorphizedFunction cache to force a fresh verify_and_compile
        f_caller.cache.clear()

        # Call again to verify it works (this should be a cache hit)
        self.assertEqual(f_caller(1), 5)

        # 4. Redefine g_dep with a different refinement (Even instead of Positive)
        @verify
        def g_dep(x: Even) -> i64:
            return x

        # Clear Python-side MonomorphizedFunction cache of f_caller to compile again
        f_caller.cache.clear()

        # Since f_caller was compiled expecting Positive, and g_dep now requires Even,
        # calling f_caller again should:
        # - Detect dependency signature mismatch
        # - Invalidate the cache
        # - Re-run verification which should FAIL (since 5 is not Even)
        with self.assertRaises(VerificationError) as ctx:
            f_caller(1)

        self.assertIn("Argument refinement violation", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
