import unittest
from lila import struct, i64


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


if __name__ == "__main__":
    unittest.main()
