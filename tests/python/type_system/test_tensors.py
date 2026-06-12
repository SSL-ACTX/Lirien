import unittest
from lila import verify, Tensor, f32, i64, Refined
from lila import VerificationError


class TestTensors(unittest.TestCase):
    def test_tensor_indexing(self):
        @verify(timeout=15000)
        def tensor_get(
            a: Tensor[f32, "M", "N"],
            i: Refined[i64, "0 <= v < M"],
            j: Refined[i64, "0 <= v < N"],
        ) -> f32:
            return a[i, j]

        @verify(timeout=15000)
        def tensor_set(
            a: Tensor[f32, "M", "N"],
            i: Refined[i64, "0 <= v < M"],
            j: Refined[i64, "0 <= v < N"],
            val: f32,
        ):
            a[i, j] = val

        A = Tensor.alloc((2, 3), f32)
        A[1, 2] = 42.0
        self.assertEqual(tensor_get(A, 1, 2), 42.0)

        tensor_set(A, 0, 1, 123.0)
        self.assertEqual(A[0, 1], 123.0)

    def test_tensor_indexing_out_of_bounds(self):
        # The verifier catches potential out-of-bounds access
        with self.assertRaisesRegex(
            VerificationError, "Potential out-of-bounds access"
        ):

            @verify
            def unsafe_get(a: Tensor[f32, "M", "N"], i: i64, j: i64) -> f32:
                return a[i, j]

    def test_valid_matmul(self):
        # This compiles through the verifier perfectly.
        @verify
        def valid_matmul(
            a: Tensor[f32, "M", "N"], b: Tensor[f32, "N", "K"]
        ) -> Tensor[f32, "M", "K"]:
            return a @ b

        self.assertTrue(getattr(valid_matmul, "__lila_jit__", False))

        # Test runtime execution and memory allocation
        A = Tensor.alloc((2, 3), f32)
        B = Tensor.alloc((3, 2), f32)

        # Fill A: [[1, 2, 3], [4, 5, 6]]
        A[0, 0] = 1.0
        A[0, 1] = 2.0
        A[0, 2] = 3.0
        A[1, 0] = 4.0
        A[1, 1] = 5.0
        A[1, 2] = 6.0

        # Fill B: [[7, 8], [9, 10], [11, 12]]
        B[0, 0] = 7.0
        B[0, 1] = 8.0
        B[1, 0] = 9.0
        B[1, 1] = 10.0
        B[2, 0] = 11.0
        B[2, 1] = 12.0

        C = valid_matmul(A, B)

        self.assertEqual(C.shape, (2, 2))
        self.assertEqual(C[0, 0], 58.0)  # 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        self.assertEqual(C[0, 1], 64.0)  # 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        self.assertEqual(C[1, 0], 139.0)  # 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        self.assertEqual(C[1, 1], 154.0)  # 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

    def test_invalid_matmul_inner_dim(self):
        # The verifier catches the dimension mismatch (N != P)
        with self.assertRaisesRegex(
            VerificationError,
            "Matrix multiplication dimension mismatch: inner dimensions must be equal",
        ):

            @verify
            def invalid_matmul(
                a: Tensor[f32, "M", "N"], b: Tensor[f32, "P", "K"]
            ) -> Tensor[f32, "M", "K"]:
                return a @ b

    def test_invalid_return_shape(self):
        # The verifier catches the return shape mismatch (M, K != M, P)
        with self.assertRaisesRegex(
            VerificationError, "Tensor shape mismatch in return: dimension 'P'"
        ):

            @verify
            def wrong_return(
                a: Tensor[f32, "M", "N"], b: Tensor[f32, "N", "K"]
            ) -> Tensor[f32, "M", "P"]:
                return a @ b


if __name__ == "__main__":
    unittest.main()
