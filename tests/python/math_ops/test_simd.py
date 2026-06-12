import unittest
import math
from lila import verify, f32x4, f64x2, i32x4, i64x2, i8x16, u8x16, i16x8, u16x8


class TestSIMD(unittest.TestCase):
    def test_i8x16_arithmetic(self):
        @verify
        def i8_ops(a: i8x16, b: i8x16) -> i8x16:
            return a + b

        va = i8x16(*(i for i in range(16)))
        vb = i8x16(10)
        res = i8_ops(va, vb)

        for i in range(16):
            self.assertEqual(res[i], i + 10)

    def test_u8x16_arithmetic(self):
        @verify
        def u8_ops(a: u8x16, b: u8x16) -> u8x16:
            return a - b

        va = u8x16(255)
        vb = u8x16(*(i for i in range(16)))
        res = u8_ops(va, vb)

        for i in range(16):
            self.assertEqual(res[i], 255 - i)

    def test_i16x8_arithmetic(self):
        @verify
        def i16_ops(a: i16x8, b: i16x8) -> i16x8:
            return a * b

        va = i16x8(*(i * 10 for i in range(8)))
        vb = i16x8(2)
        res = i16_ops(va, vb)

        for i in range(8):
            self.assertEqual(res[i], i * 20)

    def test_u16x8_arithmetic(self):
        @verify
        def u16_ops(a: u16x8, b: u16x8) -> u16x8:
            return a + b

        va = u16x8(*(i * 100 for i in range(8)))
        vb = u16x8(50)
        res = u16_ops(va, vb)

        for i in range(8):
            self.assertEqual(res[i], i * 100 + 50)

    def test_i8x16_auto_splat(self):
        @verify
        def splat_i8(a: i8x16, val: int) -> i8x16:
            return a + val

        va = i8x16(0)
        res = splat_i8(va, 42)

        for i in range(16):
            self.assertEqual(res[i], 42)

    def test_f32x4_arithmetic(self):
        @verify
        def f32_ops(a: f32x4, b: f32x4) -> f32x4:
            return (a + b) * (a - b)

        va = f32x4(10.0, 20.0, 30.0, 40.0)
        vb = f32x4(1.0, 2.0, 3.0, 4.0)

        # (a+b)*(a-b) = a^2 - b^2
        # [99.0, 396.0, 891.0, 1584.0]
        res = f32_ops(va, vb)

        self.assertAlmostEqual(res[0], 99.0)
        self.assertAlmostEqual(res[1], 396.0)
        self.assertAlmostEqual(res[2], 891.0)
        self.assertAlmostEqual(res[3], 1584.0)

    def test_f64x2_arithmetic(self):
        @verify
        def f64_ops(a: f64x2, b: f64x2) -> f64x2:
            return a * b + 5.0

        va = f64x2(2.0, 4.0)
        vb = f64x2(10.0, 20.0)
        res = f64_ops(va, vb)

        self.assertAlmostEqual(res[0], 25.0)
        self.assertAlmostEqual(res[1], 85.0)

    def test_i32x4_arithmetic(self):
        @verify
        def i32_ops(a: i32x4, b: i32x4) -> i32x4:
            return a + b

        va = i32x4(100, 200, 300, 400)
        vb = i32x4(1, 2, 3, 4)
        res = i32_ops(va, vb)

        self.assertEqual(res[0], 101)
        self.assertEqual(res[1], 202)
        self.assertEqual(res[2], 303)
        self.assertEqual(res[3], 404)

    def test_i64x2_arithmetic(self):
        @verify
        def i64_ops(a: i64x2, b: i64x2) -> i64x2:
            return a - b

        va = i64x2(1000, 2000)
        vb = i64x2(1, 2)
        res = i64_ops(va, vb)

        self.assertEqual(res[0], 999)
        self.assertEqual(res[1], 1998)

    def test_simd_splat(self):
        @verify
        def splat_test(a: f32x4, factor: float) -> f32x4:
            return a * factor

        va = f32x4(1.0, 2.0, 3.0, 4.0)
        res = splat_test(va, 10.0)

        self.assertAlmostEqual(res[0], 10.0)
        self.assertAlmostEqual(res[1], 20.0)
        self.assertAlmostEqual(res[2], 30.0)
        self.assertAlmostEqual(res[3], 40.0)

    def test_simd_lerp(self):
        @verify
        def lerp(a: f32x4, b: f32x4, t: float) -> f32x4:
            return a + (b - a) * t

        va = f32x4(0.0, 10.0, 20.0, 30.0)
        vb = f32x4(10.0, 20.0, 30.0, 40.0)
        res = lerp(va, vb, 0.5)

        self.assertAlmostEqual(res[0], 5.0)
        self.assertAlmostEqual(res[1], 15.0)
        self.assertAlmostEqual(res[2], 25.0)
        self.assertAlmostEqual(res[3], 35.0)

    def test_simd_abs_neg(self):
        @verify
        def abs_neg_test(a: f32x4) -> f32x4:
            return abs(-a)

        va = f32x4(-1.0, 2.0, -3.0, 4.0)
        res = abs_neg_test(va)
        # -va = [1.0, -2.0, 3.0, -4.0]
        # abs(-va) = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1], 2.0)
        self.assertEqual(res[2], 3.0)
        self.assertEqual(res[3], 4.0)

    def test_simd_min_max(self):
        @verify
        def min_max_test(a: i32x4, b: i32x4) -> i32x4:
            return min(a, b) + max(a, b)

        va = i32x4(10, 20, 30, 40)
        vb = i32x4(40, 30, 20, 10)
        res = min_max_test(va, vb)
        # min = [10, 20, 20, 10]
        # max = [40, 30, 30, 40]
        # sum = [50, 50, 50, 50]
        for i in range(4):
            self.assertEqual(res[i], 50)

    def test_simd_avg(self):
        @verify
        def avg_test(a: i32x4, b: i32x4) -> i32x4:
            return math.avg(a, b)

        va = i32x4(10, 20, 30, 40)
        vb = i32x4(11, 21, 31, 41)
        res = avg_test(va, vb)
        # (10 + 11 + 1) // 2 = 11 (avg_round)
        # (20 + 21 + 1) // 2 = 21
        for i in range(4):
            self.assertEqual(res[i], va[i] + 1)


if __name__ == "__main__":
    unittest.main()
