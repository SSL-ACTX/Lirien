import unittest
from lila import verify, i64, Buffer


@verify
def loop_no_hints(buf: Buffer[i64]) -> i64:
    n = len(buf)
    res = 0
    for i in range(n):
        res = res + buf[i]
    return res


@verify
def loop_decrement(buf: Buffer[i64]) -> i64:
    n = len(buf)
    res = 0
    i = n - 1
    while i >= 0:
        res = res + buf[i]
        i = i - 1
    return res


class TestLoopAutoInvariant(unittest.TestCase):
    def test_loop_no_hints(self):
        from array import array

        data = array("q", [1, 2, 3, 4, 5])
        res = loop_no_hints(data)
        self.assertEqual(res, 15)

    def test_loop_decrement(self):
        from array import array

        data = array("q", [1, 2, 3, 4, 5])
        res = loop_decrement(data)
        self.assertEqual(res, 15)


if __name__ == "__main__":
    unittest.main()
