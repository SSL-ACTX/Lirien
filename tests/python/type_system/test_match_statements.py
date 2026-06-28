import unittest
from lirien import verify, enum, struct, i64
from typing import NamedTuple


@struct
class SomePayload:
    val: i64


@struct
class Empty:
    pass


@enum
class Option:
    Some: SomePayload
    Empty: Empty


class Point(NamedTuple):
    x: i64
    y: i64


@struct
class Point3D:
    x: i64
    y: i64
    z: i64


class TestMatchStatements(unittest.TestCase):
    def test_match_basic(self):
        @verify
        def unwrap_or_zero(opt: Option) -> i64:
            match opt:
                case Option.Some(s):
                    return s.val
                case Option.Empty:
                    return 0

        self.assertEqual(unwrap_or_zero(Option.Some(10)), 10)
        self.assertEqual(unwrap_or_zero(Option.Empty()), 0)

    def test_match_tuple(self):
        @verify
        def match_tuple_fn(t: tuple[i64, i64]) -> i64:
            match t:
                case (x, y) if x > y:
                    return x - y
                case (x, y):
                    return y - x

        self.assertEqual(match_tuple_fn((10, 5)), 5)
        self.assertEqual(match_tuple_fn((5, 10)), 5)

    def test_match_namedtuple(self):
        @verify
        def match_namedtuple_fn(p: Point) -> i64:
            match p:
                case (x, y) if x == y:
                    return 1
                case (x, y):
                    return 0

        self.assertEqual(match_namedtuple_fn(Point(5, 5)), 1)
        self.assertEqual(match_namedtuple_fn(Point(5, 10)), 0)

    def test_match_struct(self):
        @verify
        def match_struct_fn(p: Point3D) -> i64:
            match p:
                case Point3D(x, y, z) if x > y:
                    return x
                case Point3D(x, y, z) if y > z:
                    return y
                case Point3D(x, y, z):
                    return z

        self.assertEqual(match_struct_fn(Point3D(10, 5, 2)), 10)
        self.assertEqual(match_struct_fn(Point3D(2, 10, 5)), 10)
        self.assertEqual(match_struct_fn(Point3D(2, 5, 10)), 10)


if __name__ == "__main__":
    unittest.main()
