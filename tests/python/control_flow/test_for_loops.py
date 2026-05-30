import unittest
from lila import verify, i64, Buffer
import array


class TestForLoops(unittest.TestCase):
    def test_range_basic(self):
        @verify
        def sum_range(n: i64) -> i64:
            total: i64 = 0
            for i in range(n):
                total = total + i
            return total

        self.assertEqual(sum_range(5), 10)
        self.assertEqual(sum_range(10), 45)

    def test_range_start_stop(self):
        @verify
        def sum_range_start_stop(start: i64, stop: i64) -> i64:
            total: i64 = 0
            for i in range(start, stop):
                total = total + i
            return total

        self.assertEqual(sum_range_start_stop(2, 5), 9)  # 2 + 3 + 4

    def test_range_step(self):
        @verify
        def sum_odds(n: i64) -> i64:
            total: i64 = 0
            for i in range(1, n, 2):
                total = total + i
            return total

        self.assertEqual(sum_odds(5), 4)  # 1 + 3 = 4
        self.assertEqual(sum_odds(6), 9)  # 1 + 3 + 5 = 9

    def test_range_negative_step(self):
        @verify
        def sum_backwards(start: i64, stop: i64) -> i64:
            total: i64 = 0
            for i in range(start, stop, -1):
                total = total + i
            return total

        self.assertEqual(sum_backwards(5, 2), 12)  # 5 + 4 + 3 = 12

    def test_nested_loops(self):
        @verify
        def nested(n: i64) -> i64:
            count = 0
            for i in range(n):
                for j in range(n):
                    count = count + 1
            return count

        self.assertEqual(nested(3), 9)

    def test_enumerate_buffer(self):
        @verify
        def sum_enumerate(buf: Buffer[i64]) -> i64:
            total: i64 = 0
            for i, x in enumerate(buf):
                total = total + x + i
            return total

        buf = array.array("q", [10, 20, 30])
        # (10 + 0) + (20 + 1) + (30 + 2) = 10 + 21 + 32 = 63
        self.assertEqual(sum_enumerate(buf), 63)

    def test_direct_iter_buffer(self):
        @verify
        def sum_direct(buf: Buffer[i64]) -> i64:
            total: i64 = 0
            for x in buf:
                total = total + x
            return total

        buf = array.array("q", [1, 2, 3, 4, 5])
        self.assertEqual(sum_direct(buf), 15)

    def test_break(self):
        @verify
        def sum_until_limit(n: i64, limit: i64) -> i64:
            total: i64 = 0
            for i in range(n):
                if i >= limit:
                    break
                total = total + i
            return total

        self.assertEqual(sum_until_limit(10, 5), 10)  # 0+1+2+3+4 = 10
        self.assertEqual(sum_until_limit(5, 10), 10)  # 0+1+2+3+4 = 10

    def test_continue(self):
        @verify
        def sum_evens(n: i64) -> i64:
            total: i64 = 0
            for i in range(n):
                if i % 2 != 0:
                    continue
                total = total + i
            return total

        self.assertEqual(sum_evens(5), 6)  # 0+2+4 = 6


if __name__ == "__main__":
    unittest.main()
