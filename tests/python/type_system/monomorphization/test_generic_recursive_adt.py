import unittest
from lila import verify, adt, i64, Box
from typing import TypeVar, Generic

T = TypeVar("T")


@adt
class List(Generic[T]):
    Cons: (T, Box["List[T]"])
    Nil: None


@verify
def get_head(l: List[i64]) -> i64:
    match l:
        case List_i64.Cons(val, next_node):
            return val
        case List_i64.Nil:
            return -1


@verify
def get_length(l: List[i64]) -> i64:
    match l:
        case List_i64.Cons(_, next_node):
            return 1 + get_length(next_node.val)
        case List_i64.Nil:
            return 0


class TestGenericRecursiveADT(unittest.TestCase):
    def test_list_i64(self):
        nil = List[i64].Nil()
        l1 = List[i64].Cons(42, nil)
        l2 = List[i64].Cons(1, l1)
        l3 = List[i64].Cons(2, l2)

        self.assertEqual(get_head(l3), 2)
        self.assertEqual(get_length(l3), 3)
        self.assertEqual(get_length(nil), 0)


if __name__ == "__main__":
    unittest.main()
