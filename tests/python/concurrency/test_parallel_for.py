import unittest
import numpy as np
from lila import verify, parallel_for, Buffer, f64, Hand, i64, struct, VerificationError


@struct
class Stats:
    count: i64


@verify
def update_h(h: Hand[i64], v: i64) -> None:
    h.val = v


@verify
def update_s(s: Hand[Stats], v: i64) -> None:
    s.count = v


@verify
def read_buf(b: Buffer[f64], i: i64) -> f64:
    if i >= 0:
        if i < len(b):
            return b[i]
    return 0.0


class TestParallelFor(unittest.TestCase):
    def test_parallel_read_simple(self):
        """Test parallel read using direct buffer access."""

        @verify
        def parallel_read(vec: Buffer[f64]) -> f64:
            parallel_for(range(len(vec)), lambda i: vec[i])
            return 0.0

        vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        parallel_read(vec)

    def test_parallel_read_call(self):
        """Test parallel read calling another verified function."""

        @verify
        def parallel_read_call(b: Buffer[f64]) -> None:
            parallel_for(range(len(b)), lambda i: read_buf(b, i))

        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        parallel_read_call(b)

    def test_parallel_write_fail(self):
        """Verify that parallel mutation of a captured Hand is blocked."""
        with self.assertRaises(VerificationError) as cm:

            @verify
            def parallel_mut(h: Hand[i64]) -> None:
                parallel_for(range(10), lambda i: update_h(h, 10))

        self.assertIn("Possible data-race", str(cm.exception))

    def test_parallel_shared_write_fail(self):
        """Verify that parallel mutation of a shared struct field is blocked."""
        with self.assertRaises(VerificationError) as cm:

            @verify
            def parallel_shared_mut(s: Hand[Stats]) -> None:
                parallel_for(range(10), lambda i: update_s(s, 10))

        self.assertIn("Possible data-race", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
