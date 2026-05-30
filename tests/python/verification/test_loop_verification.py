import unittest
from lila import verify, i64, Buffer


class TestLoopVerification(unittest.TestCase):
    def test_loop_safety(self):
        @verify
        def access_all(buf: Buffer[i64]) -> i64:
            total = 0
            # Lila should prove i is always in bounds
            for i in range(len(buf)):
                total = total + buf[i]
            return total

        import array

        buf = array.array("q", [1, 2, 3])
        self.assertEqual(access_all(buf), 6)

    def test_buffer_copy(self):
        @verify
        def buffer_copy(src: Buffer[i64], dst: Buffer[i64]) -> i64:
            # Prove this is safe even if src and dst are different sizes
            # as long as we only loop up to the smaller one.
            limit = len(src)
            if len(dst) < limit:
                limit = len(dst)

            for i in range(limit):
                dst[i] = src[i]

            return limit

        import array

        s = array.array("q", [1, 2, 3])
        d = array.array("q", [0, 0, 0, 0])
        self.assertEqual(buffer_copy(s, d), 3)
        self.assertEqual(d[0], 1)
        self.assertEqual(d[1], 2)
        self.assertEqual(d[2], 3)


if __name__ == "__main__":
    unittest.main()
