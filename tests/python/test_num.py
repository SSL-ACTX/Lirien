import unittest
import math
from lirien import Tensor, f32, num, f32x4


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
        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[0, 2] = 3.0
        a[1, 0] = 4.0
        a[1, 1] = 5.0
        a[1, 2] = 6.0

        # b = [[7, 8],
        #      [9, 10],
        #      [11, 12]]
        b[0, 0] = 7.0
        b[0, 1] = 8.0
        b[1, 0] = 9.0
        b[1, 1] = 10.0
        b[2, 0] = 11.0
        b[2, 1] = 12.0

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

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[1, 0] = 3.0
        a[1, 1] = 4.0

        b[0, 0] = 5.0
        b[0, 1] = 6.0
        b[1, 0] = 7.0
        b[1, 1] = 8.0

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

        a[0, 0] = 5.0
        a[0, 1] = 6.0
        a[1, 0] = 7.0
        a[1, 1] = 8.0

        b[0, 0] = 1.0
        b[0, 1] = 2.0
        b[1, 0] = 3.0
        b[1, 1] = 4.0

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

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[1, 0] = 3.0
        a[1, 1] = 4.0

        b[0, 0] = 5.0
        b[0, 1] = 6.0
        b[1, 0] = 7.0
        b[1, 1] = 8.0

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
            1.0,
            2.0,
            5.0,
            6.0,
            3.0,
            4.0,
            7.0,
            8.0,
            9.0,
            10.0,
            13.0,
            14.0,
            11.0,
            12.0,
            15.0,
            16.0,
        ]
        for i in range(4):
            for j in range(4):
                image[i, j] = val_list[i * 4 + j]

        num.max_pool2d_2x2(image, out)

        self.assertEqual(out[0, 0], 4.0)
        self.assertEqual(out[0, 1], 8.0)
        self.assertEqual(out[1, 0], 12.0)
        self.assertEqual(out[1, 1], 16.0)

    def test_avg_pool2d_2x2(self):
        # H = 4, W = 4, OH = 2, OW = 2
        image = Tensor.alloc((4, 4), f32)
        out = Tensor.alloc((2, 2), f32)

        val_list = [
            1.0,
            2.0,
            5.0,
            6.0,
            3.0,
            4.0,
            7.0,
            8.0,
            9.0,
            10.0,
            13.0,
            14.0,
            11.0,
            12.0,
            15.0,
            16.0,
        ]
        for i in range(4):
            for j in range(4):
                image[i, j] = val_list[i * 4 + j]

        num.avg_pool2d_2x2(image, out)

        self.assertAlmostEqual(out[0, 0], 2.5)
        self.assertAlmostEqual(out[0, 1], 6.5)
        self.assertAlmostEqual(out[1, 0], 10.5)
        self.assertAlmostEqual(out[1, 1], 14.5)

    def test_clip(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = -1.5
        a[0, 1] = 0.5
        a[1, 0] = 2.5
        a[1, 1] = 1.0

        num.clip(a, out, 0.0, 2.0)

        self.assertEqual(out[0, 0], 0.0)
        self.assertEqual(out[0, 1], 0.5)
        self.assertEqual(out[1, 0], 2.0)
        self.assertEqual(out[1, 1], 1.0)

    def test_mean(self):
        # M = 4
        a = Tensor.alloc((4,), f32)
        out = Tensor.alloc((1,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        a[3] = 4.0

        num.mean(a, out, 4.0)

        self.assertAlmostEqual(out[0], 2.5)

    def test_scale(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[1, 0] = 3.0
        a[1, 1] = 4.0

        num.scale(a, out, 2.5)

        self.assertEqual(out[0, 0], 2.5)
        self.assertEqual(out[0, 1], 5.0)
        self.assertEqual(out[1, 0], 7.5)
        self.assertEqual(out[1, 1], 10.0)

    def test_bias_add(self):
        # M = 2, N = 3
        a = Tensor.alloc((2, 3), f32)
        bias = Tensor.alloc((3,), f32)
        out = Tensor.alloc((2, 3), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[0, 2] = 3.0
        a[1, 0] = 4.0
        a[1, 1] = 5.0
        a[1, 2] = 6.0

        bias[0] = 0.5
        bias[1] = 1.0
        bias[2] = 1.5

        num.bias_add(a, bias, out)

        self.assertEqual(out[0, 0], 1.5)
        self.assertEqual(out[0, 1], 3.0)
        self.assertEqual(out[0, 2], 4.5)
        self.assertEqual(out[1, 0], 4.5)
        self.assertEqual(out[1, 1], 6.0)
        self.assertEqual(out[1, 2], 7.5)

    def test_standardize(self):
        # M = 3
        a = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0

        num.standardize(a, out, 2.0, 1.0)

        self.assertAlmostEqual(out[0], -1.0)
        self.assertAlmostEqual(out[1], 0.0)
        self.assertAlmostEqual(out[2], 1.0)

    def test_matvec(self):
        # M = 2, N = 3
        matrix = Tensor.alloc((2, 3), f32)
        vector = Tensor.alloc((3,), f32)
        out = Tensor.alloc((2,), f32)

        matrix[0, 0] = 1.0
        matrix[0, 1] = 2.0
        matrix[0, 2] = 3.0
        matrix[1, 0] = 4.0
        matrix[1, 1] = 5.0
        matrix[1, 2] = 6.0

        vector[0] = 2.0
        vector[1] = 1.0
        vector[2] = 3.0

        num.matvec(matrix, vector, out)

        self.assertEqual(out[0], 13.0)
        self.assertEqual(out[1], 31.0)

    def test_outer(self):
        # M = 3, N = 2
        a = Tensor.alloc((3,), f32)
        b = Tensor.alloc((2,), f32)
        out = Tensor.alloc((3, 2), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        b[0] = 4.0
        b[1] = 5.0

        num.outer(a, b, out)

        self.assertEqual(out[0, 0], 4.0)
        self.assertEqual(out[0, 1], 5.0)
        self.assertEqual(out[1, 0], 8.0)
        self.assertEqual(out[1, 1], 10.0)
        self.assertEqual(out[2, 0], 12.0)
        self.assertEqual(out[2, 1], 15.0)

    def test_dot(self):
        # M = 3
        a = Tensor.alloc((3,), f32)
        b = Tensor.alloc((3,), f32)
        out = Tensor.alloc((1,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        b[0] = 4.0
        b[1] = 5.0
        b[2] = 6.0

        num.dot(a, b, out)

        self.assertEqual(out[0], 32.0)

    def test_l2_normalize(self):
        # M = 3
        a = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 3.0
        a[1] = 4.0
        a[2] = 0.0

        num.l2_normalize(a, out, 1e-9)

        self.assertAlmostEqual(out[0], 0.6)
        self.assertAlmostEqual(out[1], 0.8)
        self.assertAlmostEqual(out[2], 0.0)

    def test_l1_normalize(self):
        # M = 3
        a = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 1.0
        a[1] = -2.0
        a[2] = 1.0

        num.l1_normalize(a, out, 1e-9)

        self.assertAlmostEqual(out[0], 0.25)
        self.assertAlmostEqual(out[1], -0.5)
        self.assertAlmostEqual(out[2], 0.25)

    def test_cosine_similarity(self):
        # M = 3
        a = Tensor.alloc((3,), f32)
        b = Tensor.alloc((3,), f32)
        out = Tensor.alloc((1,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        b[0] = 2.0
        b[1] = 4.0
        b[2] = 6.0

        num.cosine_similarity(a, b, out, 1e-9)

        self.assertAlmostEqual(out[0], 1.0, places=5)

    def test_silu(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 0.0
        a[0, 1] = 1.0
        a[1, 0] = -1.0
        a[1, 1] = 2.0

        num.silu(a, out)

        self.assertAlmostEqual(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 1], 1.0 / (1.0 + math.exp(-1.0)), places=5)
        self.assertAlmostEqual(out[1, 0], -1.0 / (1.0 + math.exp(1.0)), places=5)
        self.assertAlmostEqual(out[1, 1], 2.0 / (1.0 + math.exp(-2.0)), places=5)

    def test_rms_norm(self):
        # M = 3
        a = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0

        num.rms_norm(a, out, 1e-9, 3.0)

        rms = math.sqrt(14.0 / 3.0)
        self.assertAlmostEqual(out[0], 1.0 / rms, places=5)
        self.assertAlmostEqual(out[1], 2.0 / rms, places=5)
        self.assertAlmostEqual(out[2], 3.0 / rms, places=5)

    def test_layer_norm(self):
        # M = 3
        a = Tensor.alloc((3,), f32)
        gamma = Tensor.alloc((3,), f32)
        beta = Tensor.alloc((3,), f32)
        out = Tensor.alloc((3,), f32)

        a[0] = 1.0
        a[1] = 2.0
        a[2] = 3.0
        gamma[0] = 1.0
        gamma[1] = 1.0
        gamma[2] = 1.0
        beta[0] = 0.0
        beta[1] = 0.0
        beta[2] = 0.0

        num.layer_norm(a, out, gamma, beta, 1e-9, 3.0)

        std_val = math.sqrt(2.0 / 3.0)
        self.assertAlmostEqual(out[0], -1.0 / std_val, places=5)
        self.assertAlmostEqual(out[1], 0.0, places=5)
        self.assertAlmostEqual(out[2], 1.0 / std_val, places=5)

    def test_hardsigmoid(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = -4.0
        a[0, 1] = 0.0
        a[1, 0] = 3.0
        a[1, 1] = -1.5

        num.hardsigmoid(a, out)

        self.assertAlmostEqual(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 1], 0.5)
        self.assertAlmostEqual(out[1, 0], 1.0)
        self.assertAlmostEqual(out[1, 1], 0.25)

    def test_hardswish(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = -4.0
        a[0, 1] = 0.0
        a[1, 0] = 3.0
        a[1, 1] = -1.5

        num.hardswish(a, out)

        self.assertAlmostEqual(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 1], 0.0)
        self.assertAlmostEqual(out[1, 0], 3.0)
        self.assertAlmostEqual(out[1, 1], -0.375)

    def test_elu(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = -1.0
        a[1, 0] = 0.0
        a[1, 1] = -2.0

        num.elu(a, out, 1.0)

        self.assertAlmostEqual(out[0, 0], 1.0)
        self.assertAlmostEqual(out[0, 1], math.exp(-1.0) - 1.0, places=5)
        self.assertAlmostEqual(out[1, 0], 0.0)
        self.assertAlmostEqual(out[1, 1], math.exp(-2.0) - 1.0, places=5)

    def test_selu(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        a[0, 0] = 1.0
        a[0, 1] = -1.0
        a[1, 0] = 0.0
        a[1, 1] = -2.0

        num.selu(a, out)

        scale = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717

        self.assertAlmostEqual(out[0, 0], scale * 1.0, places=5)
        self.assertAlmostEqual(
            out[0, 1], scale * alpha * (math.exp(-1.0) - 1.0), places=5
        )
        self.assertAlmostEqual(out[1, 0], 0.0, places=5)
        self.assertAlmostEqual(
            out[1, 1], scale * alpha * (math.exp(-2.0) - 1.0), places=5
        )

    def test_matvec_bias(self):
        # M = 2, N = 3
        matrix = Tensor.alloc((2, 3), f32)
        vector = Tensor.alloc((3,), f32)
        bias = Tensor.alloc((2,), f32)
        out = Tensor.alloc((2,), f32)

        matrix[0, 0] = 1.0
        matrix[0, 1] = 2.0
        matrix[0, 2] = 3.0
        matrix[1, 0] = 4.0
        matrix[1, 1] = 5.0
        matrix[1, 2] = 6.0

        vector[0] = 2.0
        vector[1] = 1.0
        vector[2] = 3.0
        bias[0] = 0.5
        bias[1] = -1.5

        num.matvec_bias(matrix, vector, bias, out)

        self.assertEqual(out[0], 13.5)
        self.assertEqual(out[1], 29.5)

    def test_sigmoid_cross_entropy(self):
        # M = 2, N = 2
        logits = Tensor.alloc((2, 2), f32)
        targets = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((2, 2), f32)

        logits[0, 0] = 0.0
        targets[0, 0] = 0.5
        logits[0, 1] = 1.0
        targets[0, 1] = 1.0
        logits[1, 0] = -2.0
        targets[1, 0] = 0.0
        logits[1, 1] = 10.0
        targets[1, 1] = 0.0

        num.sigmoid_cross_entropy(logits, targets, out)

        self.assertAlmostEqual(out[0, 0], 0.693147, places=5)
        self.assertAlmostEqual(out[0, 1], 0.3132617, places=5)
        self.assertAlmostEqual(out[1, 0], 0.126928, places=5)
        self.assertAlmostEqual(out[1, 1], 10.000045, places=5)

    def test_l2_loss(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32)
        b = Tensor.alloc((2, 2), f32)
        out = Tensor.alloc((1,), f32)

        a[0, 0] = 1.0
        a[0, 1] = 2.0
        a[1, 0] = 3.0
        a[1, 1] = 4.0

        b[0, 0] = 2.0
        b[0, 1] = 1.0
        b[1, 0] = 4.0
        b[1, 1] = 2.0

        num.l2_loss(a, b, out, 8.0)

        self.assertAlmostEqual(out[0], 0.875, places=5)

    def test_dot_simd(self):
        # M = 2
        a = Tensor.alloc((2,), f32x4)
        b = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((1,), f32)

        a[0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[1] = f32x4(5.0, 6.0, 7.0, 8.0)

        b[0] = f32x4(2.0, 1.0, 0.5, 0.25)
        b[1] = f32x4(0.0, 1.0, 2.0, 3.0)

        num.dot_simd(a, b, out)

        # a[0]*b[0] = [2.0, 2.0, 1.5, 1.0], sum = 6.5
        # a[1]*b[1] = [0.0, 6.0, 14.0, 24.0], sum = 44.0
        # Total sum = 6.5 + 44.0 = 50.5
        self.assertAlmostEqual(out[0], 50.5, places=5)

    def test_matvec_simd(self):
        # M = 2, N = 2
        matrix = Tensor.alloc((2, 2), f32x4)
        vector = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((2,), f32)

        matrix[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)
        matrix[0, 1] = f32x4(5.0, 6.0, 7.0, 8.0)
        matrix[1, 0] = f32x4(0.0, 1.0, 2.0, 3.0)
        matrix[1, 1] = f32x4(1.0, 1.0, 1.0, 1.0)

        vector[0] = f32x4(2.0, 1.0, 0.5, 0.25)
        vector[1] = f32x4(0.0, 1.0, 2.0, 3.0)

        num.matvec_simd(matrix, vector, out)

        # Row 0: matrix[0,0]*vector[0] + matrix[0,1]*vector[1]
        # [2.0, 2.0, 1.5, 1.0] + [0.0, 6.0, 14.0, 24.0] = [2.0, 8.0, 15.5, 25.0]
        # Sum = 50.5
        # Row 1: matrix[1,0]*vector[0] + matrix[1,1]*vector[1]
        # [0.0, 1.0, 1.0, 0.75] + [0.0, 1.0, 2.0, 3.0] = [0.0, 2.0, 3.0, 3.75]
        # Sum = 8.75
        self.assertAlmostEqual(out[0], 50.5, places=5)
        self.assertAlmostEqual(out[1], 8.75, places=5)

    def test_mse_simd(self):
        # M = 2
        a = Tensor.alloc((2,), f32x4)
        b = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((1,), f32)

        a[0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[1] = f32x4(5.0, 6.0, 7.0, 8.0)

        b[0] = f32x4(2.0, 1.0, 4.0, 2.0)
        b[1] = f32x4(4.0, 7.0, 5.0, 9.0)

        num.mse_simd(a, b, out)

        # diff0 = [-1.0, 1.0, -1.0, 2.0], diff0^2 = [1, 1, 1, 4] -> sum = 7
        # diff1 = [1.0, -1.0, 2.0, -1.0], diff1^2 = [1, 1, 4, 1] -> sum = 7
        # Total sum = 14.0
        self.assertAlmostEqual(out[0], 14.0, places=5)

    def test_mae_simd(self):
        # M = 2
        a = Tensor.alloc((2,), f32x4)
        b = Tensor.alloc((2,), f32x4)
        out = Tensor.alloc((1,), f32)

        a[0] = f32x4(1.0, 2.0, 3.0, 4.0)
        a[1] = f32x4(5.0, 6.0, 7.0, 8.0)

        b[0] = f32x4(2.0, 1.0, 4.0, 2.0)
        b[1] = f32x4(4.0, 7.0, 5.0, 9.0)

        num.mae_simd(a, b, out)

        # abs(diff0) = [1.0, 1.0, 1.0, 2.0] -> sum = 5.0
        # abs(diff1) = [1.0, 1.0, 2.0, 1.0] -> sum = 5.0
        # Total sum = 10.0
        self.assertAlmostEqual(out[0], 10.0, places=5)

    def test_add_simd(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32x4)
        b = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)
        b[0, 0] = f32x4(10.0, 20.0, 30.0, 40.0)

        num.add_simd(a, b, out)

        res = out[0, 0]
        self.assertEqual(res[0], 11.0)
        self.assertEqual(res[1], 22.0)
        self.assertEqual(res[2], 33.0)
        self.assertEqual(res[3], 44.0)

    def test_sub_simd(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32x4)
        b = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(10.0, 20.0, 30.0, 40.0)
        b[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)

        num.sub_simd(a, b, out)

        res = out[0, 0]
        self.assertEqual(res[0], 9.0)
        self.assertEqual(res[1], 18.0)
        self.assertEqual(res[2], 27.0)
        self.assertEqual(res[3], 36.0)

    def test_mul_simd(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32x4)
        b = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)
        b[0, 0] = f32x4(5.0, 6.0, 7.0, 8.0)

        num.mul_simd(a, b, out)

        res = out[0, 0]
        self.assertEqual(res[0], 5.0)
        self.assertEqual(res[1], 12.0)
        self.assertEqual(res[2], 21.0)
        self.assertEqual(res[3], 32.0)

    def test_scale_simd(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(1.0, 2.0, 3.0, 4.0)

        num.scale_simd(a, out, 5.0)

        res = out[0, 0]
        self.assertEqual(res[0], 5.0)
        self.assertEqual(res[1], 10.0)
        self.assertEqual(res[2], 15.0)
        self.assertEqual(res[3], 20.0)

    def test_relu_simd(self):
        # M = 2, N = 2
        a = Tensor.alloc((2, 2), f32x4)
        out = Tensor.alloc((2, 2), f32x4)

        a[0, 0] = f32x4(-1.5, 0.0, 2.5, -0.5)

        num.relu_simd(a, out)

        res = out[0, 0]
        self.assertEqual(res[0], 0.0)
        self.assertEqual(res[1], 0.0)
        self.assertEqual(res[2], 2.5)
        self.assertEqual(res[3], 0.0)


if __name__ == "__main__":
    unittest.main()
