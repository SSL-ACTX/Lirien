import unittest
from lila import verify, i64, f64
from typing import Callable, TypeVar

T = TypeVar("T")


@verify
def inner_hof(f: Callable[[T], T], x: T) -> T:
    return f(x)


@verify
def outer_hof(f: Callable[[T], T], x: T) -> T:
    return inner_hof(f, x)


@verify
def inc_i64(x: i64) -> i64:
    return x + 1


@verify
def inc_f64(x: f64) -> f64:
    return x + 1.0


class TestHOFStability(unittest.TestCase):
    def test_recursive_generic_hof(self):
        @verify
        def recursive_apply(f: Callable[[i64], i64], x: i64, n: i64) -> i64:
            if n <= 0:
                return x
            return recursive_apply(f, f(x), n - 1)

        self.assertEqual(recursive_apply(inc_i64, 0, 10), 10)


if __name__ == "__main__":
    unittest.main()
