import unittest
from lila import verify, i64, f64, f32x4, Buffer, Box
from typing import TypeVar
import numpy as np

T = TypeVar("T")


@verify
def add_anything(a: T, b: T) -> T:
    return a + b


@verify
def buffer_sum(data: Buffer[T]) -> T:
    res: T = 0
    for i in range(len(data)):
        res = res + data[i]
    return res


@verify
def unbox_add(boxed: Box[T], val: T) -> T:
    return boxed.value + val


class TestMonomorphization(unittest.TestCase):
    def test_basic_monomorphization(self):
        res_i64 = add_anything(10, 20)
        self.assertEqual(res_i64, 30)

        res_f64 = add_anything(1.5, 2.5)
        self.assertEqual(res_f64, 4.0)

    def test_simd_monomorphization(self):
        v1 = f32x4(1.0, 2.0, 3.0, 4.0)
        v2 = f32x4(10.0, 20.0, 30.0, 40.0)
        res_simd = add_anything(v1, v2)

        self.assertEqual(res_simd.f0, 11.0)
        self.assertEqual(res_simd.f1, 22.0)
        self.assertEqual(res_simd.f2, 33.0)
        self.assertEqual(res_simd.f3, 44.0)

    def test_buffer_monomorphization(self):
        # Test with i64 buffer
        data_i64 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        res_i64 = buffer_sum(data_i64)
        self.assertEqual(res_i64, 15)

        # Test with f64 buffer
        data_f64 = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        res_f64 = buffer_sum(data_f64)
        self.assertAlmostEqual(res_f64, 6.6)

    def test_box_monomorphization(self):
        # Test with i64 box
        b_i64 = Box[i64](100)
        res_i64 = unbox_add(b_i64, 50)
        self.assertEqual(res_i64, 150)

        # Test with f64 box
        b_f64 = Box[f64](1.5)
        res_f64 = unbox_add(b_f64, 2.5)
        self.assertEqual(res_f64, 4.0)


if __name__ == "__main__":
    unittest.main()
