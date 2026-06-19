import unittest
from lirien import verify, Tensor, f32


class TestTensorReductions(unittest.TestCase):
    def test_tensor_sum(self):
        @verify
        def tensor_sum(a: Tensor[f32, "M", "N"]) -> f32:
            return a.sum()

        A = Tensor.alloc((2, 3), f32)
        for i in range(2):
            for j in range(3):
                A[i, j] = 1.0

        # 2 * 3 = 6.0
        self.assertEqual(tensor_sum(A), 6.0)

    def test_tensor_max(self):
        @verify
        def tensor_max(a: Tensor[f32, "M", "N"]) -> f32:
            return a.max()

        A = Tensor.alloc((2, 2), f32)
        A[0, 0] = 1.0
        A[0, 1] = 5.0
        A[1, 0] = 3.0
        A[1, 1] = 2.0

        self.assertEqual(tensor_max(A), 5.0)

    def test_tensor_min(self):
        @verify
        def tensor_min(a: Tensor[f32, "M", "N"]) -> f32:
            return a.min()

        A = Tensor.alloc((2, 2), f32)
        A[0, 0] = 10.0
        A[0, 1] = 5.0
        A[1, 0] = 3.0
        A[1, 1] = 2.0

        self.assertEqual(tensor_min(A), 2.0)


if __name__ == "__main__":
    unittest.main()
