import unittest
from lila import i64, Owned, Ref, Mut, verify
from lila.compiler import VerificationError


@verify
def consume_owned(x: Owned[i64]) -> i64:
    return 1


@verify
def consume_mut(x: Mut[i64]) -> i64:
    return 1


@verify
def consume_ref(x: Ref[i64]) -> i64:
    return 1


class TestPermissions(unittest.TestCase):
    def test_permission_aliasing_conflict(self):
        with self.assertRaisesRegex(VerificationError, "Memory safety violation"):

            @verify
            def conflict(x: i64) -> i64:
                m = Mut(x)
                r = Ref(x)
                return consume_mut(m) + consume_ref(r)

    def test_permission_double_move(self):
        with self.assertRaisesRegex(VerificationError, "Memory safety violation"):

            @verify
            def double_move(x: Owned[i64]) -> i64:
                y = consume_owned(x)
                z = consume_owned(x)
                return y + z

    def test_permission_multiple_refs_ok(self):
        @verify
        def multiple_refs(x: i64) -> i64:
            r1 = Ref(x)
            r2 = Ref(x)
            r3 = Ref(x)
            return 1

        self.assertEqual(multiple_refs(10), 1)

    def test_permission_move_in_loop(self):
        with self.assertRaisesRegex(VerificationError, "Memory safety violation"):

            @verify
            def loop_move(n: i64, x: Owned[i64]) -> i64:
                i = 0
                while i < n:
                    y = consume_owned(x)
                    i = i + 1
                return 1


if __name__ == "__main__":
    unittest.main()
