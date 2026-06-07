import unittest
from lila import verify, adt, i64, f64, struct, VerificationError


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

    def test_adt_non_exhaustive(self):
        # Explicit catch-all should be fine
        @verify
        def get_area_safe(s: Shape) -> f64:
            match s:
                case Shape.Circle(r):
                    return 3.14159 * r * r
                case _:
                    return 0.0

        c = Shape.Circle(10.0)
        self.assertAlmostEqual(get_area_safe(c), 314.159)

        p = Shape.Point()
        self.assertEqual(get_area_safe(p), 0.0)

        # Missing variant without catch-all should fail verification
        with self.assertRaises(VerificationError) as cm:

            @verify
            def fail_match(s: Shape) -> f64:
                match s:
                    case Shape.Circle(r):
                        return 1.0

        self.assertIn("Non-exhaustive match detected", str(cm.exception))

    def test_adt_nested_pattern(self):
        @struct
        class Rect:
            p1: Point2D
            p2: Point2D

        @adt
        class Geometry:
            Item: Rect
            Other: i64

        @verify
        def get_p1_x(g: Geometry) -> i64:
            match g:
                case Geometry.Item(Rect(Point2D(x, y), p2)):
                    return x
                case Geometry.Other(v):
                    return v
                case _:
                    return -1

        @verify
        def get_p2_y(g: Geometry) -> i64:
            match g:
                case Geometry.Item(Rect(p1, Point2D(x, y))):
                    return y
                case _:
                    return -1

        r = Rect(Point2D(10, 20), Point2D(30, 40))
        g1 = Geometry.Item(r)
        self.assertEqual(get_p1_x(g1), 10)
        self.assertEqual(get_p2_y(g1), 40)

        g2 = Geometry.Other(42)
        self.assertEqual(get_p1_x(g2), 42)


if __name__ == "__main__":
    unittest.main()
