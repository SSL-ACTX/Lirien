import unittest
from lirien import verify, i64, SizedArray, TypeVar, Refined

N = TypeVar("N")


@verify
def get_slice_sum(
    arr: SizedArray[i64, N],
    start: Refined[i64, lambda x: (x >= 0) & (x <= N)],
    end: Refined[i64, lambda x: (x >= start) & (x <= N)],
) -> i64:
    # This should trigger ArraySlice IR instruction
    s = arr[start:end]
    res: i64 = 0
    # For now, we need to know the size of the slice to iterate over it
    for i in range(end - start):
        res = res + s[i]
    return res


@verify
def slice_open_end(
    arr: SizedArray[i64, 10], start: Refined[i64, lambda x: (x >= 0) & (x <= 10)]
) -> i64:
    s = arr[start:]
    res: i64 = 0
    for i in range(10 - start):
        res = res + s[i]
    return res


@verify
def slice_open_start(
    arr: SizedArray[i64, 10], end: Refined[i64, lambda x: (x >= 0) & (x <= 10)]
) -> i64:
    s = arr[:end]
    res: i64 = 0
    for i in range(end):
        res = res + s[i]
    return res


@verify
def nested_slice(arr: SizedArray[i64, 10]) -> i64:
    s1 = arr[2:8]  # [2, 3, 4, 5, 6, 7]
    s2 = s1[1:4]  # [3, 4, 5]
    return s2[0] + s2[1] + s2[2]


@verify
def slice_empty(arr: SizedArray[i64, 10]) -> i64:
    s = arr[5:5]
    return 0


class TestSlices(unittest.TestCase):
    def test_basic_slice(self):
        data = SizedArray[i64, 10]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # sum of [2, 3, 4, 5] = 14
        res = get_slice_sum(data, 2, 6)
        self.assertEqual(res, 14)

    def test_open_end(self):
        data = SizedArray[i64, 10]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # [7, 8, 9] = 24
        res = slice_open_end(data, 7)
        self.assertEqual(res, 24)

    def test_open_start(self):
        data = SizedArray[i64, 10]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # [0, 1, 2, 3] = 6
        res = slice_open_start(data, 4)
        self.assertEqual(res, 6)

    def test_nested_slice(self):
        data = SizedArray[i64, 10]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # nested_slice returns 3 + 4 + 5 = 12
        res = nested_slice(data)
        self.assertEqual(res, 12)

    def test_empty_slice(self):
        data = SizedArray[i64, 10]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        res = slice_empty(data)
        self.assertEqual(res, 0)


if __name__ == "__main__":
    unittest.main()
