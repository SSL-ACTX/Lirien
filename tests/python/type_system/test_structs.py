import unittest
from lirien import struct, i64


@struct
class Point:
    x: i64
    y: i64

    def move(self, dx: i64, dy: i64) -> i64:
        self.x = self.x + dx
        self.y = self.y + dy
        return self.x + self.y

    def scale(self, factor: i64) -> None:
        self.x = self.x * factor
        self.y = self.y * factor


class TestStructs(unittest.TestCase):
    def test_method_jit(self):
        p = Point(10, 20)
        res = p.move(5, 5)
        self.assertEqual(res, 40)
        self.assertEqual(p.x, 15)
        self.assertEqual(p.y, 25)

    def test_void_method(self):
        p = Point(10, 20)
        p.scale(2)
        self.assertEqual(p.x, 20)
        self.assertEqual(p.y, 40)

    def test_nested_method_call(self):
        @struct
        class Vector:
            x: i64
            y: i64

            def move(self, dx: i64, dy: i64):
                self.x = self.x + dx
                self.y = self.y + dy

            def jump(self, dist: i64):
                self.move(dist, dist)

        v = Vector(0, 0)
        v.jump(5)
        self.assertEqual(v.x, 5)
        self.assertEqual(v.y, 5)

    def test_struct_repr(self):
        p = Point(10, 20)
        self.assertEqual(repr(p), "Point(x=10, y=20)")

    def test_struct_eq(self):
        p1 = Point(10, 20)
        p2 = Point(10, 20)
        p3 = Point(15, 20)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)


if __name__ == "__main__":
    unittest.main()
