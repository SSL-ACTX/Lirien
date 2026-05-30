import unittest
from lila import verify, i64, f64


class TestWhileLoops(unittest.TestCase):
    def test_while_basic(self):
        @verify
        def sum_to_n(n: i64) -> i64:
            s = 0
            i = 0
            while i < n:
                s += i
                i += 1
            return s

        self.assertEqual(sum_to_n(10), 45)
        self.assertEqual(sum_to_n(0), 0)

    def test_while_break(self):
        @verify
        def find_first_gt(n: i64, limit: i64) -> i64:
            i = 0
            while i < n:
                if i > limit:
                    break
                i += 1
            return i

        self.assertEqual(find_first_gt(10, 5), 6)
        self.assertEqual(find_first_gt(5, 10), 5)

    def test_while_continue(self):
        @verify
        def sum_evens_while(n: i64) -> i64:
            s = 0
            i = 0
            while i < n:
                i += 1
                if i % 2 != 0:
                    continue
                s += i
            return s

        self.assertEqual(sum_evens_while(10), 30)  # 2 + 4 + 6 + 8 + 10 = 30

    def test_while_float(self):
        @verify
        def while_float(n: f64) -> f64:
            x = 0.0
            while x < n:
                x += 1.5
            return x

        self.assertAlmostEqual(while_float(10.0), 10.5)


if __name__ == "__main__":
    unittest.main()
