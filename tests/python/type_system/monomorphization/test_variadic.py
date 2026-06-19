import unittest
from lirien import verify, i64, f32, Tensor
from typing import TypeVar, TypeVarTuple, Unpack

Shape = TypeVarTuple("Shape")
N = TypeVar("N")


@verify
def get_rank_poly(x: Tensor[f32, Unpack[Shape]]) -> i64:
    return len(Shape)


@verify
def get_first_dim(x: Tensor[f32, N, Unpack[Shape]]) -> i64:
    return N


class TestVariadicMonomorphization(unittest.TestCase):
    def test_variadic_typevars(self):
        # 1D
        t1 = Tensor.alloc((10,), f32)
        self.assertEqual(get_rank_poly(t1), 1)
        self.assertEqual(get_first_dim(t1), 10)

        # 3D
        t3 = Tensor.alloc((2, 3, 4), f32)
        self.assertEqual(get_rank_poly(t3), 3)
        self.assertEqual(get_first_dim(t3), 2)


if __name__ == "__main__":
    unittest.main()
