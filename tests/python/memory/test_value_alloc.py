import unittest
from lirien import value, i64, Buffer, verify


@value
class Point3D:
    x: i64
    y: i64
    z: i64


@value
class Particle:
    position: Point3D
    velocity: Point3D
    mass: i64


@verify
def sum_points(data: Buffer[Point3D]) -> i64:
    total = 0
    for i in range(len(data)):
        total = total + data[i].x + data[i].y + data[i].z
    return total


@verify
def update_particles(data: Buffer[Particle], time_step: i64) -> None:
    for i in range(len(data)):
        data[i].position.x = data[i].position.x + (data[i].velocity.x * time_step)
        data[i].position.y = data[i].position.y + (data[i].velocity.y * time_step)
        data[i].position.z = data[i].position.z + (data[i].velocity.z * time_step)


class TestBufferValue(unittest.TestCase):
    def test_buffer_alloc(self):
        arr = Buffer[Point3D].alloc(10)

        for i in range(10):
            arr[i].x = 1
            arr[i].y = 2
            arr[i].z = 3

        res = sum_points(arr)
        self.assertEqual(res, 60)

    def test_nested_value_types(self):
        arr = Buffer[Particle].alloc(5)

        for i in range(5):
            arr[i].position.x = 10
            arr[i].position.y = 20
            arr[i].position.z = 30
            arr[i].velocity.x = 1
            arr[i].velocity.y = 2
            arr[i].velocity.z = 3
            arr[i].mass = 100

        update_particles(arr, 5)

        # Verify the updates
        self.assertEqual(arr[0].position.x, 15)
        self.assertEqual(arr[0].position.y, 30)
        self.assertEqual(arr[0].position.z, 45)

        self.assertEqual(arr[4].position.x, 15)
        self.assertEqual(arr[4].position.y, 30)
        self.assertEqual(arr[4].position.z, 45)


if __name__ == "__main__":
    unittest.main()
