import unittest
from lila import verify, adt, i64, f64, struct


@adt
class Shape:
    Circle: f64
    Rectangle: (i64, i64)
    Point: None


@struct
class Point2D:
    x: i64
    y: i64


@adt
class MyAdt:
    A: Point2D
    B: i64


class TestADT(unittest.TestCase):
    def test_adt_basic(self):
        @verify
        def get_area(s: Shape) -> f64:
            match s:
                case Shape.Circle(r):
                    return 3.14159 * r * r
                case Shape.Rectangle(w, h):
                    return float(w * h)
                case Shape.Point:
                    return 0.0

        c = Shape.Circle(10.0)
        self.assertAlmostEqual(get_area(c), 314.159)

        r = Shape.Rectangle(10, 20)
        self.assertEqual(get_area(r), 200.0)

        p = Shape.Point()
        self.assertEqual(get_area(p), 0.0)

    def test_adt_nested_struct(self):
        @verify
        def get_x(m: MyAdt) -> i64:
            match m:
                case MyAdt.A(p):
                    return p.x
                case MyAdt.B(val):
                    return val

        m1 = MyAdt.A(Point2D(10, 20))
        self.assertEqual(get_x(m1), 10)

        m2 = MyAdt.B(42)
        self.assertEqual(get_x(m2), 42)


if __name__ == "__main__":
    unittest.main()
