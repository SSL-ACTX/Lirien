import unittest
from typing import NamedTuple
from lirien import verify, i64, f64


class Point(NamedTuple):
    x: i64
    y: i64


class Rect(NamedTuple):
    top_left: Point
    bottom_right: Point


class Point3D(NamedTuple):
    x: f64
    y: f64
    z: f64


@verify
def add_points(p1: Point, p2: Point) -> Point:
    return Point(p1.x + p2.x, p1.y + p2.y)


@verify
def dot_product(p1: Point3D, p2: Point3D) -> f64:
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z


@verify
def get_width(r: Rect) -> i64:
    return r.bottom_right.x - r.top_left.x


@verify
def scale_point(p: Point, factor: i64) -> Point:
    return Point(p.x * factor, p.y * factor)


from typing import TypeVar

T = TypeVar("T")


@verify
def identity(x: T) -> T:
    return x


@verify
def unpack_namedtuple(p: Point) -> i64:
    x, y = p
    return x + y


@verify
def unpack_namedtuple_list_target(p: Point) -> i64:
    [x, y] = p
    return x + y


@verify
def unpack_tuple_list_target(t: tuple[i64, i64]) -> i64:
    [x, y] = t
    return x + y


class TestNamedTuple(unittest.TestCase):
    def test_add_points(self):
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        res = add_points(p1, p2)
        self.assertIsInstance(res, Point)
        self.assertEqual(res.x, 4)
        self.assertEqual(res.y, 6)

    def test_dot_product(self):
        p1 = Point3D(1.0, 2.0, 3.0)
        p2 = Point3D(4.0, 5.0, 6.0)
        res = dot_product(p1, p2)
        self.assertEqual(res, 4.0 + 10.0 + 18.0)

    def test_nested_namedtuple(self):
        r = Rect(Point(10, 20), Point(50, 100))
        self.assertEqual(get_width(r), 40)

    def test_scale_point(self):
        p = Point(10, 20)
        res = scale_point(p, 3)
        self.assertEqual(res.x, 30)
        self.assertEqual(res.y, 60)

    def test_generic_namedtuple(self):
        p = Point(42, 84)
        res = identity(p)
        self.assertIsInstance(res, Point)
        self.assertEqual(res.x, 42)
        self.assertEqual(res.y, 84)

        p3d = Point3D(1.0, 2.0, 3.0)
        res3d = identity(p3d)
        self.assertIsInstance(res3d, Point3D)
        self.assertEqual(res3d.z, 3.0)

    def test_unpack_namedtuple(self):
        p = Point(5, 7)
        res = unpack_namedtuple(p)
        self.assertEqual(res, 12)

    def test_unpack_namedtuple_list_target(self):
        p = Point(8, 9)
        res = unpack_namedtuple_list_target(p)
        self.assertEqual(res, 17)

    def test_unpack_tuple_list_target(self):
        t = (15, 25)
        res = unpack_tuple_list_target(t)
        self.assertEqual(res, 40)


if __name__ == "__main__":
    unittest.main()
