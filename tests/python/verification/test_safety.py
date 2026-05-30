import unittest
from lila import verify, i64, Owned, Mut, Ref
from lila.compiler import VerificationError


@verify
def consume_val(x: Owned[i64]) -> i64:
    return x


from lila import struct


@struct
class Dummy:
    val: i64


class TestSafety(unittest.TestCase):
    def test_use_after_move(self):
        # Lila should block use-after-move at compile time
        with self.assertRaises(VerificationError) as cm:

            @verify
            def illegal_use(x: Owned[i64]) -> i64:
                a = consume_val(x)
                return a + x

        self.assertIn("Memory safety violation", str(cm.exception))

    def test_aliasing_violation(self):
        # Lila should block Mut and Ref aliasing same root

        with self.assertRaises(VerificationError) as cm:

            @verify
            def illegal_alias_struct(d: Mut[Dummy]) -> i64:
                r1 = Ref(d)  # Shared permission
                d.val = 10  # Exclusive permission - violation!
                return r1.val

        self.assertIn("Memory safety violation", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
