import unittest
from lila import verify, adt, i64, f64


@adt
class Shape:
    Circle: f64
    Rectangle: (i64, i64)


class TestMatchGuards(unittest.TestCase):
    def test_guard_basic(self):
        @verify
        def classify_circle(s: Shape) -> i64:
            match s:
                case Shape.Circle(r) if r > 10.0:
                    return 1  # Large
                case Shape.Circle(r):
                    return 2  # Small
                case _:
                    return 0

        self.assertEqual(classify_circle(Shape.Circle(15.0)), 1)
        self.assertEqual(classify_circle(Shape.Circle(5.0)), 2)
        self.assertEqual(classify_circle(Shape.Rectangle(10, 10)), 0)

    def test_guard_nested(self):
        @verify
        def complex_guard(s: Shape) -> i64:
            match s:
                case Shape.Rectangle(w, h) if w == h:
                    return 1  # Square
                case Shape.Rectangle(w, h) if w > h:
                    return 2  # Wide
                case Shape.Rectangle(w, h):
                    return 3  # Tall
                case _:
                    return 0

        self.assertEqual(complex_guard(Shape.Rectangle(10, 10)), 1)
        self.assertEqual(complex_guard(Shape.Rectangle(20, 10)), 2)
        self.assertEqual(complex_guard(Shape.Rectangle(10, 20)), 3)


if __name__ == "__main__":
    unittest.main()
