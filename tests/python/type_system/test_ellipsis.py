import unittest
from lirien import verify, Tensor, f32, f64, i64, Refined, SizedArray, Buffer


class TestEllipsis(unittest.TestCase):
    def test_buffer_inference(self):
        @verify
        def buffer_sum(buf: Buffer[...]) -> f32:
            s = 0.0
            for i in range(len(buf)):
                s += buf[i]
            return s

        import array

        # 'f' is float32
        data = array.array("f", [1.0, 2.0, 3.0, 4.0])
        mv = memoryview(data)

        self.assertEqual(buffer_sum(mv), 10.0)

        @verify
        def buffer_sum_i64(buf: Buffer[...]) -> i64:
            s = 0
            for i in range(len(buf)):
                s += buf[i]
            return s

        data_i = array.array("q", [10, 20, 30])  # 'q' is i64
        mv_i = memoryview(data_i)
        self.assertEqual(buffer_sum_i64(mv_i), 60)

    def test_sized_array_inference(self):
        @verify
        def sum_array(arr: SizedArray[i64, ...]) -> i64:
            s = 0
            for i in range(len(arr)):
                s += arr[i]
            return s

        data5 = SizedArray[i64, 5]([1, 2, 3, 4, 5])
        self.assertEqual(sum_array(data5), 15)

        data3 = SizedArray[i64, 3]([10, 20, 30])
        self.assertEqual(sum_array(data3), 60)

    def test_rank_polymorphism_tensor(self):
        @verify
        def get_first(a: Tensor[f32, ...]) -> f32:
            return a[0]

        @verify
        def get_first_2d(a: Tensor[f32, ...]) -> f32:
            return a[0, 0]

        A1 = Tensor.alloc((5,), f32)
        A1[0] = 42.0
        self.assertEqual(get_first(A1), 42.0)

        A2 = Tensor.alloc((3, 3), f32)
        A2[0, 0] = 123.0
        self.assertEqual(get_first_2d(A2), 123.0)

    def test_suffix_shape_matching(self):
        @verify
        def last_dim_match(a: Tensor[f32, ..., "N"], b: Tensor[f32, "N"]) -> f32:
            # a is rank 2 (M, N), b is rank 1 (N)
            return a[0, 0] + b[0]

        A = Tensor.alloc((2, 4), f32)
        B = Tensor.alloc((4,), f32)
        A[0, 0] = 10.0
        B[0] = 5.0
        self.assertEqual(last_dim_match(A, B), 15.0)

    def test_multi_ellipsis_expansion(self):
        @verify
        def multi_rank(a: Tensor[f32, ...], b: Tensor[f32, ...]) -> f32:
            return a[0] + b[0, 0]

        A = Tensor.alloc((2,), f32)
        B = Tensor.alloc((2, 2), f32)
        A[0] = 1.0
        B[0, 0] = 2.0
        self.assertEqual(multi_rank(A, B), 3.0)

    def test_refinement_inference(self):
        @verify
        def clamp_pos(x: i64) -> Refined[i64, ...]:
            if x < 1:
                return 1
            if x > 100:
                return 100
            return x

        self.assertEqual(clamp_pos(0), 1)
        self.assertEqual(clamp_pos(50), 50)
        self.assertEqual(clamp_pos(200), 100)

        @verify
        def uses_inference(x: i64) -> i64:
            y = clamp_pos(x)
            # Lirien should have inferred (and (>= {v} 1) (<= {v} 100))
            if y >= 1 and y <= 100:
                return y
            return -1  # Unreachable

        self.assertEqual(uses_inference(500), 100)
        self.assertEqual(uses_inference(-500), 1)

    def test_inferred_float_bounds(self):
        @verify
        def float_abs(x: f64) -> Refined[f64, ...]:
            if x < 0.0:
                return -x
            return x

        self.assertEqual(float_abs(-5.0), 5.0)
        self.assertEqual(float_abs(3.0), 3.0)


if __name__ == "__main__":
    unittest.main()
