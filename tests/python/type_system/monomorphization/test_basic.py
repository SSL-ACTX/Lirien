import unittest
from lirien import verify
from typing import TypeVar

T = TypeVar("T")


@verify
def add_anything(a: T, b: T) -> T:
    return a + b


class TestBasicMonomorphization(unittest.TestCase):
    def test_basic_monomorphization(self):
        res_i64 = add_anything(10, 20)
        self.assertEqual(res_i64, 30)

        res_f64 = add_anything(1.5, 2.5)
        self.assertEqual(res_f64, 4.0)


if __name__ == "__main__":
    unittest.main()
