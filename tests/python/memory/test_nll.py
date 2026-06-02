import unittest
from lila import verify, Hand, Held, Peek
from lila.types import i64, struct


@struct
class Point:
    x: i64
    y: i64


class TestNLL(unittest.TestCase):
    def test_nll_reborrow(self):
        @verify
        def nll_reborrow(p: Hand[Point]) -> None:
            px = p.x
            px.val = 10
            # px is not used after this.
            # p.x should be re-mutable.
            p.x = 20

        p = Point(x=0, y=0)
        nll_reborrow(p)
        self.assertEqual(p.x, 20)

    def test_nll_conditional(self):
        @verify
        def nll_cond(p: Hand[Point], cond: bool) -> None:
            if cond:
                px = p.x
                px.val = 10
                # px dead here
            else:
                py = p.y
                py.val = 20
                # py dead here
            
            p.x = 30 # Safe in both paths because px/py are dead

        p = Point(x=0, y=0)
        nll_cond(p, True)
        self.assertEqual(p.x, 30)
        
        p2 = Point(x=0, y=0)
        nll_cond(p2, False)
        self.assertEqual(p2.x, 30)

    def test_lexical_fail_still_happens_if_live(self):
        p = Point(x=0, y=0)
        with self.assertRaises(Exception) as cm:

            @verify
            def lexical_fail(p: Hand[Point]) -> None:
                px = p.x
                p.x = 20  # px is still live because it's used later
                px.val = 10

            lexical_fail(p)
        self.assertIn("Memory safety violation", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
