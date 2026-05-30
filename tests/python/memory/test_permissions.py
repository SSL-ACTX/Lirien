import unittest
from lila import i64, Held, Peek, Hand, verify
from lila.compiler import VerificationError


@verify
def consume_held(x: Held[i64]) -> i64:
    return 1


@verify
def consume_hand(x: Hand[i64]) -> i64:
    return 1


@verify
def consume_peek(x: Peek[i64]) -> i64:
    return 1


class TestPermissions(unittest.TestCase):
    def test_permission_aliasing_conflict(self):
        with self.assertRaisesRegex(VerificationError, "Memory safety violation"):

            @verify
            def conflict(x: i64) -> i64:
                m = Hand(x)
                r = Peek(x)
                return consume_hand(m) + consume_peek(r)

    def test_permission_double_move(self):
        with self.assertRaisesRegex(VerificationError, "Memory safety violation"):

            @verify
            def double_move(x: Held[i64]) -> i64:
                y = consume_held(x)
                z = consume_held(x)
                return y + z

    def test_permission_multiple_refs_ok(self):
        @verify
        def multiple_refs(x: i64) -> i64:
            r1 = Peek(x)
            r2 = Peek(x)
            r3 = Peek(x)
            return 1

        self.assertEqual(multiple_refs(10), 1)

    def test_permission_move_in_loop(self):
        with self.assertRaisesRegex(VerificationError, "Memory safety violation"):

            @verify
            def loop_move(n: i64, x: Held[i64]) -> i64:
                i = 0
                while i < n:
                    y = consume_held(x)
                    i = i + 1
                return 1


if __name__ == "__main__":
    unittest.main()
