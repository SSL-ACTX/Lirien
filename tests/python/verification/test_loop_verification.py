import unittest
from lirien import verify, i64, Buffer, VerificationError


class TestLoopVerification(unittest.TestCase):
    def test_loop_safety(self):
        @verify
        def access_all(buf: Buffer[i64]) -> i64:
            total = 0
            # Lirien should prove i is always in bounds
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

    def test_safe_dynamic_loop(self):
        @verify
        def safe_dynamic_loop(buf: Buffer[i64], limit: i64) -> i64:
            if limit <= len(buf):
                idx = 0
                for i in range(limit):
                    buf[idx] = 42
                    idx = idx + 1
                return idx
            return 0

        import array

        b = array.array("q", [0, 0])
        self.assertEqual(safe_dynamic_loop(b, 2), 2)
        self.assertEqual(b[0], 42)
        self.assertEqual(b[1], 42)

    def test_unsafe_dynamic_loop(self):
        with self.assertRaises(VerificationError):

            @verify
            def unsafe_dynamic_loop(buf: Buffer[i64], limit: i64) -> i64:
                idx = 0
                for i in range(limit):
                    buf[idx] = 42
                    idx = idx + 1
                return idx


if __name__ == "__main__":
    unittest.main()
