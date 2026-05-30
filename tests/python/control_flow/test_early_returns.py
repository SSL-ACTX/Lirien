import unittest
from lila import verify, i64


class TestEarlyReturns(unittest.TestCase):
    def test_early_return_if(self):
        @verify
        def absolute_value(x: i64) -> i64:
            if x < 0:
                return -x
            return x

        self.assertEqual(absolute_value(-5), 5)
        self.assertEqual(absolute_value(10), 10)

    def test_early_return_loop(self):
        @verify
        def find_first_positive(n: i64) -> i64:
            for i in range(n):
                if i > 2:
                    return i
            return -1

        self.assertEqual(find_first_positive(5), 3)
        self.assertEqual(find_first_positive(2), -1)


if __name__ == "__main__":
    unittest.main()
