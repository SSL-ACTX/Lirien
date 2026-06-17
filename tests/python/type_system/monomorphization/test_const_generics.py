import unittest
from lila import verify, i64, f32, Tensor, SizedArray
from typing import TypeVar, Tuple

M = TypeVar("M")
N = TypeVar("N")


@verify
def get_tensor_dims(x: Tensor[f32, M, N]) -> Tuple[i64, i64]:
    return M, N


@verify
def SizedArray_sum(arr: SizedArray[i64, N]) -> i64:
    res: i64 = 0
    # N is used as a loop limit
    for i in range(N):
        res = res + arr[i]
    return res


class TestConstGenericsMonomorphization(unittest.TestCase):
    def test_const_generics_tensor(self):
        t1 = Tensor.alloc((10, 20), f32)
        m, n = get_tensor_dims(t1)
        self.assertEqual(m, 10)
        self.assertEqual(n, 20)

        t2 = Tensor.alloc((5, 5), f32)
        m, n = get_tensor_dims(t2)
        self.assertEqual(m, 5)
        self.assertEqual(n, 5)

    def test_const_generics_sized_array(self):
        # Create a SizedArray[i64, 4]
        arr = SizedArray[i64, 4](1, 2, 3, 4)
        res = SizedArray_sum(arr)
        self.assertEqual(res, 10)

        # Different size
        arr2 = SizedArray[i64, 2](10, 20)
        res2 = SizedArray_sum(arr2)
        self.assertEqual(res2, 30)


if __name__ == "__main__":
    unittest.main()
