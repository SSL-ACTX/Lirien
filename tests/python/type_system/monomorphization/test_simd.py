import unittest
from lila import verify, f32x4
from typing import TypeVar

T = TypeVar("T")


@verify
def add_anything(a: T, b: T) -> T:
    return a + b


class TestSimdMonomorphization(unittest.TestCase):
    def test_simd_monomorphization(self):
        v1 = f32x4(1.0, 2.0, 3.0, 4.0)
        v2 = f32x4(10.0, 20.0, 30.0, 40.0)
        res_simd = add_anything(v1, v2)

        self.assertEqual(res_simd.f0, 11.0)
        self.assertEqual(res_simd.f1, 22.0)
        self.assertEqual(res_simd.f2, 33.0)
        self.assertEqual(res_simd.f3, 44.0)


if __name__ == "__main__":
    unittest.main()
