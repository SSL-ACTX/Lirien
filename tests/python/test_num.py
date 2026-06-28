import unittest
import math
from lirien import Tensor, f32, num


class TestLirienNum(unittest.TestCase):
    def test_transpose(self):
        a = Tensor.alloc((2, 3), f32)
        out = Tensor.alloc((3, 2), f32)

        # Fill a:
        # [[1.0, 2.0, 3.0],
        #  [4.0, 5.0, 6.0]]
        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[0, 2] = 3.0
        a[1, 0] = 4.0
        a[1, 1] = 5.0
        a[1, 2] = 6.0

        num.transpose(a, out)

        # out should be:
        # [[1.0, 4.0],
        #  [2.0, 5.0],
        #  [3.0, 6.0]]
        self.assertEqual(out[0, 0], 1.0)
        self.assertEqual(out[0, 1], 4.0)
        self.assertEqual(out[1, 0], 2.0)
        self.assertEqual(out[1, 1], 5.0)
        self.assertEqual(out[2, 0], 3.0)
        self.assertEqual(out[2, 1], 6.0)

    def test_relu(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.5
        a[0, 1] = -2.0
        a[1, 0] = 0.0
        a[1, 1] = -0.5

        num.relu(a, out)

        self.assertEqual(out[0, 0], 1.5)
        self.assertEqual(out[0, 1], 0.0)
        self.assertEqual(out[1, 0], 0.0)
        self.assertEqual(out[1, 1], 0.0)

    def test_leaky_relu(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.5
        a[0, 1] = -2.0
        a[1, 0] = 0.0
        a[1, 1] = -0.5

        num.leaky_relu(a, out, 0.1)

        self.assertEqual(out[0, 0], 1.5)
        self.assertAlmostEqual(out[0, 1], -0.2)
        self.assertEqual(out[1, 0], 0.0)
        self.assertAlmostEqual(out[1, 1], -0.05)

    def test_sigmoid(self):
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 0.0
        a[0, 1] = 1.0
        a[1, 0] = -1.0
        a[1, 1] = 10.0

        num.sigmoid(a, out)

        self.assertAlmostEqual(out[0, 0], 0.5)
        self.assertAlmostEqual(out[0, 1], 1.0 / (1.0 + math.exp(-1.0)))
        self.assertAlmostEqual(out[1, 0], 1.0 / (1.0 + math.exp(1.0)))
        self.assertAlmostEqual(out[1, 1], 1.0 / (1.0 + math.exp(-10.0)))

    def test_convolve1d(self):
        # M = 5, K = 3, M - K + 1 = 3
        signal = Tensor.alloc((5,), f32)
        kernel = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        for i in range(5):
            signal[i] = float(i + 1)  # [1, 2, 3, 4, 5]
        for i in range(3):
            kernel[i] = 1.0  # [1, 1, 1]

        num.convolve1d(signal, kernel, out)

        # out[0] = 1*1 + 2*1 + 3*1 = 6.0
        # out[1] = 2*1 + 3*1 + 4*1 = 9.0
        # out[2] = 3*1 + 4*1 + 5*1 = 12.0
        self.assertEqual(out[0], 6.0)
        self.assertEqual(out[1], 9.0)
        self.assertEqual(out[2], 12.0)

    def test_convolve2d(self):
        # H = 3, W = 3, KH = 2, KW = 2
        # H - KH + 1 = 2, W - KW + 1 = 2
        image = Tensor.alloc((3, 3), f32)
        kernel = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        for i in range(3):
            for j in range(3):
                image[i, j] = float(i * 3 + j + 1)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        for i in range(2):
            for j in range(2):
                kernel[i, j] = 1.0  # [[1, 1], [1, 1]]

        num.convolve2d(image, kernel, out)

        # out[0, 0] = 1 + 2 + 4 + 5 = 12.0
        # out[0, 1] = 2 + 3 + 5 + 6 = 16.0
        # out[1, 0] = 4 + 5 + 7 + 8 = 24.0
        # out[1, 1] = 5 + 6 + 8 + 9 = 28.0
        self.assertEqual(out[0, 0], 12.0)
        self.assertEqual(out[0, 1], 16.0)
        self.assertEqual(out[1, 0], 24.0)
        self.assertEqual(out[1, 1], 28.0)


if __name__ == "__main__":
    unittest.main()
