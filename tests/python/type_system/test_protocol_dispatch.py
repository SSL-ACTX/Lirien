import unittest
from typing import Protocol
from lirien import verify, f32, struct


class Renderable(Protocol):
    def render(self) -> f32: ...


class Shape(Protocol):
    def get_area(self) -> f32: ...
    def scale(self, factor: f32) -> None: ...


@struct
class Circle:
    radius: f32

    def render(self) -> f32:
        return self.radius * 3.14

    def get_area(self) -> f32:
        return 3.14 * self.radius * self.radius

    def scale(self, factor: f32) -> None:
        self.radius = self.radius * factor


@struct
class Square:
    side: f32

    def render(self) -> f32:
        return self.side * self.side

    def get_area(self) -> f32:
        return self.side * self.side

    def scale(self, factor: f32) -> None:
        self.side = self.side * factor


@struct
class Scene:
    obj1: Circle
    obj2: Square

    def total_area(self) -> f32:
        return self.obj1.get_area() + self.obj2.get_area()


class TestProtocolDispatch(unittest.TestCase):
    def test_protocol_dispatch(self):
        @verify
        def draw_scene(obj: Renderable) -> f32:
            return obj.render()

        c = Circle(radius=10.0)
        s = Square(side=5.0)

        # specialised for Circle
        self.assertAlmostEqual(draw_scene(c), 31.4, places=4)
        # specialised for Square
        self.assertAlmostEqual(draw_scene(s), 25.0, places=4)

    def test_complex_protocol(self):
        @verify
        def resize_and_get_area(obj: Shape, factor: f32) -> f32:
            obj.scale(factor)
            return obj.get_area()

        c = Circle(radius=2.0)
        # 2.0 * 2.0 = 4.0 radius -> area = 3.14 * 16 = 50.24
        self.assertAlmostEqual(resize_and_get_area(c, 2.0), 50.24, places=4)
        self.assertAlmostEqual(c.radius, 4.0)

    def test_nested_protocol_dispatch(self):
        # A protocol that uses another protocol or complex struct
        @verify
        def get_scene_area(s: Scene) -> f32:
            return s.total_area()

        scene = Scene(Circle(2.0), Square(3.0))
        # Circle area: 3.14 * 4 = 12.56
        # Square area: 9.0
        # Total: 21.56
        self.assertAlmostEqual(get_scene_area(scene), 21.56, places=4)


if __name__ == "__main__":
    unittest.main()
