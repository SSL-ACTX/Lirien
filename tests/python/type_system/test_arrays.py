import unittest
from lirien import verify
from lirien.types import i64, SizedArray


@verify
def sum_array(arr: SizedArray[i64, 3]) -> i64:
    return arr[0] + arr[1] + arr[2]


@verify
def double_array(arr: SizedArray[i64, 2]):
    arr[0] = arr[0] * 2
    arr[1] = arr[1] * 2


class TestArrays(unittest.TestCase):
    def test_sized_array_basic(self):
        arr = SizedArray[i64, 3]([10, 20, 30])
        res = sum_array(arr)
        self.assertEqual(res, 60)

    def test_sized_array_mutation(self):
        arr = SizedArray[i64, 2]([5, 7])
        double_array(arr)
        self.assertEqual(arr[0], 10)
        self.assertEqual(arr[1], 14)

    def test_sized_array_oob_read(self):
        with self.assertRaisesRegex(Exception, "Potential out-of-bounds access"):

            @verify
            def read_oob(arr: SizedArray[i64, 3]) -> i64:
                return arr[3]

    def test_sized_array_oob_write(self):
        with self.assertRaisesRegex(Exception, "Potential out-of-bounds access"):

            @verify
            def write_oob(arr: SizedArray[i64, 2]):
                arr[2] = 10


if __name__ == "__main__":
    unittest.main()
