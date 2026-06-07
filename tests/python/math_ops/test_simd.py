import unittest
from lila import verify, f32x4, f64x2, i32x4, i64x2


class TestSIMD(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
