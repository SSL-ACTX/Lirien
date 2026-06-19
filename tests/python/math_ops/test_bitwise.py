import unittest
from lirien import verify, i64, u64


@verify
def bitwise_ops(a: i64, b: i64) -> i64:
    res = a & b
    res = res | (a ^ b)
    res = res << 2
    res = res >> 1
    return ~res


@verify
def unsigned_ops(a: u64, b: u64) -> u64:
    return (a + b) & 0xFFFFFFFF


class TestBitwise(unittest.TestCase):
    def test_bitwise(self):
        # 10 & 7 = 2
        # 10 ^ 7 = 13
        # 2 | 13 = 15
        # 15 << 2 = 60
        # 60 >> 1 = 30
        # ~30 = -31
        res = bitwise_ops(10, 7)
        self.assertEqual(res, -31)

    def test_unsigned_overflow(self):
        # 2^64 - 1 + 1 should be 0 in BV theory
        max_u64 = 0xFFFFFFFFFFFFFFFF
        res = unsigned_ops(max_u64, 1)
        # (0) & 0xFFFFFFFF = 0
        self.assertEqual(res, 0)


if __name__ == "__main__":
    unittest.main()
