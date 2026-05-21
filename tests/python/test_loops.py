import unittest
from lila import verify, f64, i64, Buffer
import numpy as np


@verify
def sum_buffer(buf: Buffer[f64]) -> f64:
    total = 0.0
    for x in buf:
        total += x
    return total


class TestLoops(unittest.TestCase):
    def test_sum_buffer(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        res = sum_buffer(data)
        self.assertEqual(res, 10.0)

    def test_nested_loops(self):
        @verify
        def nested(n: i64) -> i64:
            count = 0
            for i in range(n):
                for j in range(n):
                    count = count + 1
            return count

        self.assertEqual(nested(3), 9)

    def test_range_step(self):
        @verify
        def sum_step(n: i64) -> i64:
            total = 0
            for i in range(0, n, 2):
                total += i
            return total

        self.assertEqual(sum_step(10), 0 + 2 + 4 + 6 + 8)

    def test_enumerate(self):
        @verify
        def sum_enum(buf: Buffer[f64]) -> f64:
            total = 0.0
            for i, x in enumerate(buf):
                total += x + f64(i)
            return total

        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        # 1.0 + 0 + 2.0 + 1 + 3.0 + 2 = 9.0
        self.assertEqual(sum_enum(data), 9.0)

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
