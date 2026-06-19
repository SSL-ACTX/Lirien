import unittest
from typing import Tuple
from lirien import verify, i64


@verify
def get_point() -> Tuple[i64, i64]:
    return 10, 20


@verify
def test_unpack() -> i64:
    x, y = get_point()
    return x + y


@verify
def test_nested_unpack() -> i64:
    # Lirien should support nested tuple construction
    # and unpacking if the type system allows it.
    # Nested tuples require explicit type annotations.
    p = (1, (2, 3))
    x, inner = p
    y, z = inner
    return x + y + z


class TestTupleUnpacking(unittest.TestCase):
    def test_basic_unpack(self):
        self.assertEqual(test_unpack(), 30)

    def test_nested_unpack(self):
        self.assertEqual(test_nested_unpack(), 6)


if __name__ == "__main__":
    unittest.main()
