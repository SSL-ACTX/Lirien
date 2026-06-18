import unittest
from typing import Protocol
from lila import verify, f32, i64, adt, struct


class Renderable(Protocol):
    def render(self) -> f32: ...


@adt
class MyEnum:
    A: i64
    B: f32

    def render(self) -> f32:
        match self:
            case MyEnum.A(v):
                return f32(v)
            case MyEnum.B(v):
                return v


@struct
class Circle:
    radius: f32

    def render(self) -> f32:
        return self.radius * 3.14


class TestEnumProtocol(unittest.TestCase):
    def test_enum_protocol(self):
        @verify
        def draw(obj: Renderable) -> f32:
            return obj.render()

        e1 = MyEnum.A(42)
        e2 = MyEnum.B(3.14)
        c = Circle(10.0)

        self.assertAlmostEqual(draw(c), 31.4, places=4)
        self.assertAlmostEqual(draw(e1), 42.0, places=4)
        self.assertAlmostEqual(draw(e2), 3.14, places=4)


if __name__ == "__main__":
    unittest.main()
