import unittest
from lila import verify, Hand, Held, Peek
from lila.types import i64, struct


@struct
class Point:
    x: i64
    y: i64


class TestStructuralPermissions(unittest.TestCase):
    def test_disjoint_fields(self):
        @verify
        def update_disjoint(p: Hand[Point]) -> None:
            # StructOffset(p, 0) -> path .0
            # StructOffset(p, 8) -> path .1
            px = p.x
            py = p.y
            px.val = 10
            py.val = 20

        p = Point(x=0, y=0)
        update_disjoint(p)
        self.assertEqual(p.x, 10)
        self.assertEqual(p.y, 20)

    def test_overlapping_whole_and_field_fail(self):
        p = Point(x=0, y=0)
        with self.assertRaises(Exception) as cm:

            @verify
            def fail_overlap(p: Hand[Point]) -> None:
                px = p.x
                # p is the whole struct, px is a field.
                # Hand[Point] for p is exclusive.
                # px is also exclusive (Hand[i64]).
                # This should fail because they overlap.
                p.x = 30
                px.val = 40

            fail_overlap(p)
        self.assertIn("Memory safety violation", str(cm.exception))

    def test_disjoint_mutation(self):
        @verify
        def disjoint_mut(p: Hand[Point]) -> None:
            px = p.x
            p.y = 100  # Mutate y while px (reference to x) is live
            px.val = 200

        p = Point(x=0, y=0)
        disjoint_mut(p)
        self.assertEqual(p.y, 100)
        self.assertEqual(p.x, 200)


if __name__ == "__main__":
    unittest.main()
