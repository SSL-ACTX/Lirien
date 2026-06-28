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

    def test_matmul(self):
        # M = 2, N = 3, K = 2
        a = Tensor.alloc((2, 3), f32)
        b = Tensor.alloc((3, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        # a = [[1, 2, 3],
        #      [4, 5, 6]]
        a[0, 0] = 1.0; a[0, 1] = 2.0; a[0, 2] = 3.0
        a[1, 0] = 4.0; a[1, 1] = 5.0; a[1, 2] = 6.0

        # b = [[7, 8],
        #      [9, 10],
        #      [11, 12]]
        b[0, 0] = 7.0; b[0, 1] = 8.0
        b[1, 0] = 9.0; b[1, 1] = 10.0
        b[2, 0] = 11.0; b[2, 1] = 12.0

        num.matmul(a, b, out)

        # out[0, 0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58.0
        # out[0, 1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64.0
        # out[1, 0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139.0
        # out[1, 1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154.0
        self.assertEqual(out[0, 0], 58.0)
        self.assertEqual(out[0, 1], 64.0)
        self.assertEqual(out[1, 0], 139.0)
        self.assertEqual(out[1, 1], 154.0)

    def test_softmax(self):
        # N = 3
        a = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0

        num.softmax(a, out)

        sum_exp = math.exp(1.0) + math.exp(2.0) + math.exp(3.0)
        self.assertAlmostEqual(out[0], math.exp(1.0) / sum_exp)
        self.assertAlmostEqual(out[1], math.exp(2.0) / sum_exp)
        self.assertAlmostEqual(out[2], math.exp(3.0) / sum_exp)

    def test_add(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        b = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0; a[0, 1] = 2.0
        a[1, 0] = 3.0; a[1, 1] = 4.0

        b[0, 0] = 5.0; b[0, 1] = 6.0
        b[1, 0] = 7.0; b[1, 1] = 8.0

        num.add(a, b, out)

        self.assertEqual(out[0, 0], 6.0)
        self.assertEqual(out[0, 1], 8.0)
        self.assertEqual(out[1, 0], 10.0)
        self.assertEqual(out[1, 1], 12.0)

    def test_sub(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        b = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 5.0; a[0, 1] = 6.0
        a[1, 0] = 7.0; a[1, 1] = 8.0

        b[0, 0] = 1.0; b[0, 1] = 2.0
        b[1, 0] = 3.0; b[1, 1] = 4.0

        num.sub(a, b, out)

        self.assertEqual(out[0, 0], 4.0)
        self.assertEqual(out[0, 1], 4.0)
        self.assertEqual(out[1, 0], 4.0)
        self.assertEqual(out[1, 1], 4.0)

    def test_mul(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        b = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0; a[0, 1] = 2.0
        a[1, 0] = 3.0; a[1, 1] = 4.0

        b[0, 0] = 5.0; b[0, 1] = 6.0
        b[1, 0] = 7.0; b[1, 1] = 8.0

        num.mul(a, b, out)

        self.assertEqual(out[0, 0], 5.0)
        self.assertEqual(out[0, 1], 12.0)
        self.assertEqual(out[1, 0], 21.0)
        self.assertEqual(out[1, 1], 32.0)

    def test_max_pool2d_2x2(self):
        # H = 4, W = 4, OH = 2, OW = 2
        image = Tensor.alloc((4, 4), f32)
        out = Tensor.alloc((2, 2), f32)

        val_list = [
            1.0, 2.0, 5.0, 6.0,
            3.0, 4.0, 7.0, 8.0,
            9.0, 10.0, 13.0, 14.0,
            11.0, 12.0, 15.0, 16.0
        ]
        for i in range(4):
            for j in range(4):
                image[i, j] = val_list[i * 4 + j]

        num.max_pool2d_2x2(image, out)

        self.assertEqual(out[0, 0], 4.0)
        self.assertEqual(out[0, 1], 8.0)
        self.assertEqual(out[1, 0], 12.0)
        self.assertEqual(out[1, 1], 16.0)


if __name__ == "__main__":
    unittest.main()
