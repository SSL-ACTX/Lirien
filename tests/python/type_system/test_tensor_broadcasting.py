import unittest
from lirien import verify, Tensor, f32


class TestTensorBroadcasting(unittest.TestCase):
    def test_tensor_broadcast_add(self):
        @verify
        def broadcast_add(
            a: Tensor[f32, "M", "N"], b: Tensor[f32, "N"]
        ) -> Tensor[f32, "M", "N"]:
            return a + b

        A = Tensor.alloc((2, 3), f32)
        B = Tensor.alloc((3,), f32)

        # A = [[1, 2, 3], [4, 5, 6]]
        A[0, 0], A[0, 1], A[0, 2] = 1.0, 2.0, 3.0
        A[1, 0], A[1, 1], A[1, 2] = 4.0, 5.0, 6.0

        # B = [10, 20, 30]
        B[0], B[1], B[2] = 10.0, 20.0, 30.0

        C = broadcast_add(A, B)

        # C should be [[11, 22, 33], [14, 25, 36]]
        self.assertEqual(C[0, 0], 11.0)
        self.assertEqual(C[0, 1], 22.0)
        self.assertEqual(C[0, 2], 33.0)
        self.assertEqual(C[1, 0], 14.0)
        self.assertEqual(C[1, 1], 25.0)
        self.assertEqual(C[1, 2], 36.0)

    def test_tensor_broadcast_col(self):
        @verify
        def broadcast_add_col(
            a: Tensor[f32, "M", "N"], b: Tensor[f32, "M", "1"]
        ) -> Tensor[f32, "M", "N"]:
            return a + b

        A = Tensor.alloc((2, 3), f32)
        B = Tensor.alloc((2, 1), f32)

        A[0, 0], A[0, 1], A[0, 2] = 1.0, 1.0, 1.0
        A[1, 0], A[1, 1], A[1, 2] = 2.0, 2.0, 2.0

        B[0, 0] = 10.0
        B[1, 0] = 20.0

        C = broadcast_add_col(A, B)

        self.assertEqual(C[0, 0], 11.0)
        self.assertEqual(C[0, 1], 11.0)
        self.assertEqual(C[0, 2], 11.0)
        self.assertEqual(C[1, 0], 22.0)
        self.assertEqual(C[1, 1], 22.0)
        self.assertEqual(C[1, 2], 22.0)


if __name__ == "__main__":
    unittest.main()
