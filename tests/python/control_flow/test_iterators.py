import unittest
import array
from lila import verify, Buffer, i64, SizedArray


@verify
def sum_direct(buf: Buffer[i64]) -> i64:
    total = 0
    for x in buf:
        total = total + x
    return total


@verify
def sum_array_direct(arr: SizedArray[i64, 5]) -> i64:
    total = 0
    for x in arr:
        total = total + x
    return total


class TestIterators(unittest.TestCase):
    def test_buffer_iter(self):
        a = array.array("q", [1, 2, 3, 4, 5])
        self.assertEqual(sum_direct(a), 15)

    def test_array_iter(self):
        data = SizedArray[i64, 5]([10, 20, 30, 40, 50])
        self.assertEqual(sum_array_direct(data), 150)


if __name__ == "__main__":
    unittest.main()
