import unittest
from lirien import verify, Tensor, f32


class TestTensorArithmetic(unittest.TestCase):
    def test_tensor_add(self):
        @verify
        def tensor_add(
            a: Tensor[f32, "M", "N"], b: Tensor[f32, "M", "N"]
        ) -> Tensor[f32, "M", "N"]:
            return a + b

        A = Tensor.alloc((2, 3), f32)
        B = Tensor.alloc((2, 3), f32)
        for i in range(2):
            for j in range(3):
                A[i, j] = float(i + j)
                B[i, j] = 1.0

        C = tensor_add(A, B)
        for i in range(2):
            for j in range(3):
                self.assertEqual(C[i, j], float(i + j + 1))

    def test_tensor_ops(self):
        @verify
        def tensor_sub(
            a: Tensor[f32, "M", "N"], b: Tensor[f32, "M", "N"]
        ) -> Tensor[f32, "M", "N"]:
            return a - b

        @verify
        def tensor_mul(
            a: Tensor[f32, "M", "N"], b: Tensor[f32, "M", "N"]
        ) -> Tensor[f32, "M", "N"]:
            return a * b

        @verify
        def tensor_div(
            a: Tensor[f32, "M", "N"], b: Tensor[f32, "M", "N"]
        ) -> Tensor[f32, "M", "N"]:
            return a / b

        A = Tensor.alloc((2, 2), f32)
        B = Tensor.alloc((2, 2), f32)
        A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 10.0, 20.0, 30.0, 40.0
        B[0, 0], B[0, 1], B[1, 0], B[1, 1] = 2.0, 2.0, 2.0, 2.0

        S = tensor_sub(A, B)
        self.assertEqual(S[0, 0], 8.0)
        self.assertEqual(S[1, 1], 38.0)

        M = tensor_mul(A, B)
        self.assertEqual(M[0, 0], 20.0)
        self.assertEqual(M[1, 1], 80.0)

        D = tensor_div(A, B)
        self.assertEqual(D[0, 0], 5.0)
        self.assertEqual(D[1, 1], 20.0)

    def test_tensor_scalar_mul(self):
        @verify
        def tensor_scalar_mul(
            a: Tensor[f32, "M", "N"], s: f32
        ) -> Tensor[f32, "M", "N"]:
            return a * s

        A = Tensor.alloc((2, 2), f32)
        A[0, 0] = 1.0
        A[0, 1] = 2.0
        A[1, 0] = 3.0
        A[1, 1] = 4.0

        B = tensor_scalar_mul(A, 2.0)
        self.assertEqual(B[0, 0], 2.0)
        self.assertEqual(B[0, 1], 4.0)
        self.assertEqual(B[1, 0], 6.0)
        self.assertEqual(B[1, 1], 8.0)


if __name__ == "__main__":
    unittest.main()
