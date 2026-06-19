import unittest
from typing import Tuple
from lirien import verify, i64, struct


@struct
class Point:
    x: i64
    y: i64


@struct
class Nested:
    p: Point
    t: Tuple[i64, i64]
    val: i64


@verify
def test_nested_struct_tuple() -> i64:
    p = Point(10, 20)
    t = (30, 40)
    n = Nested(p, t, 50)

    # Extracting flat fields
    return n.p.x + n.p.y + n.t[0] + n.t[1] + n.val


class TestNestedLayouts(unittest.TestCase):
    def test_nested_struct_tuple(self):
        self.assertEqual(test_nested_struct_tuple(), 150)


if __name__ == "__main__":
    unittest.main()
