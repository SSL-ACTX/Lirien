import unittest
from lirien import verify, i64


@verify
def add_nested_tuples(
    t1: tuple[tuple[i64, i64], i64], t2: tuple[tuple[i64, i64], i64]
) -> tuple[tuple[i64, i64], i64]:
    ((x1, y1), z1) = t1
    ((x2, y2), z2) = t2
    return ((x1 + x2, y1 + y2), z1 + z2)


@verify
def identity_nested(t: tuple[tuple[i64, i64], i64]) -> tuple[tuple[i64, i64], i64]:
    return t


@verify
def add_tuples_2(t1: tuple[i64, i64], t2: tuple[i64, i64]) -> tuple[i64, i64]:
    return (t1[0] + t2[0], t1[1] + t2[1])


@verify
def nested_destructuring(t: tuple[tuple[i64, i64], i64]) -> i64:
    ((x, y), z) = t
    return x + y + z


class TestNestedTuple(unittest.TestCase):
    def test_nested_destructuring(self):
        res = nested_destructuring(((1, 2), 3))
        self.assertEqual(res, 6)

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
