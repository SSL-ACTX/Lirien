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


@verify
def slice_with_step_one(arr: SizedArray[i64, 10]) -> i64:
    s = arr[2:8:1]
    return s[0] + s[1]


@verify
def slice_with_step_two(arr: SizedArray[i64, 10]) -> i64:
    # arr[0:10:2] => elements at indices 0, 2, 4, 6, 8
    s = arr[0:10:2]
    # s[0]=arr[0], s[1]=arr[2], s[2]=arr[4]
    return s[0] + s[1] + s[2]


@verify
def slice_with_step_three(arr: SizedArray[i64, 9]) -> i64:
    # arr[0:9:3] => elements at indices 0, 3, 6
    s = arr[0:9:3]
    return s[0] + s[1] + s[2]


@verify
def slice_reverse_step_one(arr: SizedArray[i64, 10]) -> i64:
    # arr[9:4:-1] => elements at 9, 8, 7, 6, 5  (size=5)
    s = arr[9:4:-1]
    return s[0] + s[1] + s[2]


@verify
def slice_reverse_step_two(arr: SizedArray[i64, 10]) -> i64:
    # arr[8:0:-2] => elements at 8, 6, 4, 2  (size=4)
    s = arr[8:0:-2]
    return s[0] + s[1] + s[2] + s[3]


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

    def test_step_one(self):
        data = SizedArray[i64, 10]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # s[0]=arr[2]=2, s[1]=arr[3]=3, sum=5
        res = slice_with_step_one(data)
        self.assertEqual(res, 5)

    def test_step_two(self):
        data = SizedArray[i64, 10]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # s[0]=arr[0]=0, s[1]=arr[2]=2, s[2]=arr[4]=4, sum=6
        res = slice_with_step_two(data)
        self.assertEqual(res, 6)

    def test_step_three(self):
        data = SizedArray[i64, 9]([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # s[0]=arr[0]=0, s[1]=arr[3]=3, s[2]=arr[6]=6, sum=9
        res = slice_with_step_three(data)
        self.assertEqual(res, 9)

    def test_reverse_step_one(self):
        data = SizedArray[i64, 10]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # s[0]=arr[9]=9, s[1]=arr[8]=8, s[2]=arr[7]=7, sum=24
        res = slice_reverse_step_one(data)
        self.assertEqual(res, 24)

    def test_reverse_step_two(self):
        data = SizedArray[i64, 10]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # s[0]=arr[8]=8, s[1]=arr[6]=6, s[2]=arr[4]=4, s[3]=arr[2]=2, sum=20
        res = slice_reverse_step_two(data)
        self.assertEqual(res, 20)

    def test_zero_step_rejected(self):
        with self.assertRaises(Exception) as ctx:

            @verify
            def slice_zero_step(arr: SizedArray[i64, 10]) -> i64:
                s = arr[0:10:0]
                return s[0]

        self.assertIn("Slice step cannot be zero", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
