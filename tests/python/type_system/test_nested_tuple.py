import unittest
from typing import Tuple
from lila import verify, i64


@verify
def add_nested_tuples(
    t1: Tuple[Tuple[i64, i64], i64], t2: Tuple[Tuple[i64, i64], i64]
) -> Tuple[Tuple[i64, i64], i64]:
    # Lila IR currently supports basic arithmetic on tuple elements via destructuring in Python
    # But wait, Lila DSL might not support nested destructuring well yet.
    # Let's use simple access if supported, or just construct new ones.
    x1, y1 = t1[0]
    z1 = t1[1]
    x2, y2 = t2[0]
    z2 = t2[1]
    return ((x1 + x2, y1 + y2), z1 + z2)


@verify
def identity_nested(t: Tuple[Tuple[i64, i64], i64]) -> Tuple[Tuple[i64, i64], i64]:
    return t


@verify
def add_tuples_2(t1: Tuple[i64, i64], t2: Tuple[i64, i64]) -> Tuple[i64, i64]:
    return (t1[0] + t2[0], t1[1] + t2[1])


class TestNestedTuple(unittest.TestCase):
    def test_tuples_2(self):
        t1 = (1, 2)
        t2 = (10, 20)
        res = add_tuples_2(t1, t2)
        self.assertEqual(res, (11, 22))

    def test_identity_nested(self):
        t = ((1, 2), 3)
        res = identity_nested(t)
        self.assertEqual(res, t)
        self.assertIsInstance(res, tuple)
        self.assertIsInstance(res[0], tuple)

    def test_add_nested(self):
        t1 = ((1, 2), 3)
        t2 = ((10, 20), 30)
        res = add_nested_tuples(t1, t2)
        self.assertEqual(res, ((11, 22), 33))


if __name__ == "__main__":
    unittest.main()
