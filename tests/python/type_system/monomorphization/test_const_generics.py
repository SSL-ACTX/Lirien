import unittest
from lila import verify, i64, f32, Tensor, SizedArray, TypeVar
from typing import Tuple

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


@verify
def pad_one(x: SizedArray[i64, N], out: SizedArray[i64, N + 1]) -> i64:
    for i in range(N):
        out[i] = x[i]
    out[N] = 100
    return N + 1


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

    def test_type_level_arithmetic(self):
        arr4 = SizedArray[i64, 4](1, 2, 3, 4)
        arr5 = SizedArray[i64, 5](0, 0, 0, 0, 0)
        res = pad_one(arr4, arr5)
        self.assertEqual(res, 5)
        self.assertEqual(arr5[0], 1)
        self.assertEqual(arr5[3], 4)
        self.assertEqual(arr5[4], 100)


if __name__ == "__main__":
    unittest.main()
