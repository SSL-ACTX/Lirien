import unittest
from typing import Protocol
from lila import verify, f32, struct


class Renderable(Protocol):
    def render(self) -> f32: ...


@struct
class Circle:
    radius: f32

    def render(self) -> f32:
        return self.radius * 3.14


@struct
class Square:
    side: f32

    def render(self) -> f32:
        return self.side * self.side


class TestProtocolDispatch(unittest.TestCase):
    def test_protocol_dispatch(self):
        @verify
        def draw_scene(obj: Renderable) -> f32:
            return obj.render()

        c = Circle(radius=10.0)
        s = Square(side=5.0)

        # First call: specialized for Circle
        self.assertAlmostEqual(draw_scene(c), 31.4, places=4)

        # Second call: specialized for Square
        self.assertAlmostEqual(draw_scene(s), 25.0, places=4)


if __name__ == "__main__":
    unittest.main()
