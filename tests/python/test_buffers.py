import unittest
import array

try:
    import numpy as np
except ImportError:
    np = None
from lila import verify, Buffer, i64, u8, f64, SizedArray, Refined

Idx5 = Refined[i64, lambda x: (x >= 0) & (x < 5)]


@verify
def sum_i64_buffer(data: Buffer[i64]) -> i64:
    total = 0
    for i in range(len(data)):
        total = total + data[i]
    return total


@verify
def fill_u8_buffer(data: Buffer[u8], val: u8) -> None:
    for i in range(len(data)):
        data[i] = val


@verify
def scale_f64_buffer(data: Buffer[f64], factor: f64) -> None:
    for i in range(len(data)):
        data[i] = data[i] * factor


class TestBuffers(unittest.TestCase):
    def test_array_protocol(self):
        a = array.array("q", [1, 2, 3, 4, 5])  # 'q' is signed 64-bit
        res = sum_i64_buffer(a)
        self.assertEqual(res, 15)

    def test_bytearray_protocol(self):
        b = bytearray(10)
        fill_u8_buffer(b, 123)
        self.assertTrue(all(x == 123 for x in b))

    def test_memoryview_protocol(self):
        b = bytearray(10)
        mv = memoryview(b)
        fill_u8_buffer(mv, 42)
        self.assertTrue(all(x == 42 for x in b))

    def test_numpy_interop(self):
        if np is None:
            self.skipTest("NumPy not installed")
        a = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        scale_f64_buffer(a, 2.0)
        self.assertTrue(np.allclose(a, [2.2, 4.4, 6.6]))

    def test_sized_array_bounds(self):
        @verify
        def get_val(arr: SizedArray[i64, 5], idx: Idx5) -> i64:
            return arr[idx]

        data = SizedArray[i64, 5]([10, 20, 30, 40, 50])
        self.assertEqual(get_val(data, 2), 30)

    def test_unsafe_bounds_fail(self):
        from lila.compiler import VerificationError

        with self.assertRaises(VerificationError):

            @verify
            def get_val_unsafe(arr: SizedArray[i64, 5], idx: i64) -> i64:
                return arr[idx]


if __name__ == "__main__":
    unittest.main()
