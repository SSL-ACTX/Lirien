import unittest
from lila import verify, i64, SizedArray, f32x4
from typing import Literal


class TestLoopUnrolling(unittest.TestCase):
    def test_basic_unroll(self):
        @verify
        def unrolled_loop(data: SizedArray[i64, 4], n: Literal[4]):
            for i in range(n):
                data[i] = i * 2

        # Use Lila's SizedArray constructor instead of raw ctypes
        data = SizedArray[i64, 4](0, 0, 0, 0)
        unrolled_loop(data, 4)

        self.assertEqual(data[0], 0)
        self.assertEqual(data[1], 2)
        self.assertEqual(data[2], 4)
        self.assertEqual(data[3], 6)

    def test_unroll_with_simd(self):
        @verify
        def simd_unroll(data: SizedArray[f32x4, 2], passes: Literal[2]):
            for i in range(passes):
                data[i] = data[i] * 2.0

        # Create SIMD data without manual ctypes structs
        data = SizedArray[f32x4, 2](f32x4(1.0), f32x4(1.0))

        simd_unroll(data, 2)

        for i in range(2):
            for j in range(4):
                self.assertEqual(data[i][j], 2.0)

    def test_unroll_with_break(self):
        @verify
        def unrolled_break(data: SizedArray[i64, 4], n: Literal[4]):
            for i in range(n):
                if i == 2:
                    break
                data[i] = i * 2

        data = SizedArray[i64, 4](0, 0, 0, 0)
        unrolled_break(data, 4)

        self.assertEqual(data[0], 0)
        self.assertEqual(data[1], 2)
        self.assertEqual(data[2], 0)  # Should not be updated
        self.assertEqual(data[3], 0)  # Should not be updated


if __name__ == "__main__":
    unittest.main()
