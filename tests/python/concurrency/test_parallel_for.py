import unittest
import numpy as np
from lila import verify, parallel_for, Buffer, f64, i64, struct, VerificationError


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


if __name__ == "__main__":
    unittest.main()
