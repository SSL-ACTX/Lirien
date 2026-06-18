import unittest
from lila import verify, struct, adt, i64, f64
from typing import TypeVar, Generic

T = TypeVar("T")


@struct
class BoxedVal(Generic[T]):
    value: T


@adt
class Opt(Generic[T]):
    Some: T
    None_: None


@verify
def get_boxed_i64(b: BoxedVal[i64]) -> i64:
    return b.value


@verify
def get_boxed_f64(b: BoxedVal[f64]) -> f64:
    return b.value


@verify
def unwrap_opt_i64(o: Opt[i64]) -> i64:
    match o:
        case Opt_i64.Some(val):
            return val
        case Opt_i64.None_:
            return -1


@verify
def unwrap_opt_f64(o: Opt[f64]) -> f64:
    match o:
        case Opt_f64.Some(val):
            return val
        case Opt_f64.None_:
            return -1.0


class TestGenericData(unittest.TestCase):
    def test_struct_specialization(self):
        b1 = BoxedVal[i64](10)
        b2 = BoxedVal[f64](20.5)

        self.assertEqual(get_boxed_i64(b1), 10)
        self.assertEqual(get_boxed_f64(b2), 20.5)

    def test_adt_specialization(self):
        o1 = Opt[i64].Some(42)
        o2 = Opt[f64].Some(3.14)
        o3 = Opt[i64].None_()

        self.assertEqual(unwrap_opt_i64(o1), 42)
        self.assertAlmostEqual(unwrap_opt_f64(o2), 3.14, places=2)
        self.assertEqual(unwrap_opt_i64(o3), -1)


if __name__ == "__main__":
    unittest.main()
