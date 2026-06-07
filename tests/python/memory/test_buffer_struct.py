import unittest
from lila import struct, i64, Buffer, verify


@struct
class Point3D:
    x: i64
    y: i64
    z: i64


@verify
def sum_points(data: Buffer[Point3D]) -> i64:
    total = 0
    for i in range(len(data)):
        total = total + data[i].x + data[i].y + data[i].z
    return total


class TestBufferStruct(unittest.TestCase):
    def test_buffer_struct(self):
        # We need a way to create an array of Point3D.
        import ctypes

        ArrayType = Point3D.__lila_ctypes__ * 10
        arr = ArrayType()
        for i in range(10):
            arr[i].x = 1
            arr[i].y = 2
            arr[i].z = 3

        res = sum_points(arr)
        self.assertEqual(res, 60)


if __name__ == "__main__":
    unittest.main()
