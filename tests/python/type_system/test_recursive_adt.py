import unittest
from lila import verify, adt, i64, Box


@adt
class Node:
    Cons: (i64, Box["Node"])
    Nil: None


class TestRecursiveADT(unittest.TestCase):
    def test_list_basic(self):
        @verify
        def get_head(n: Node) -> i64:
            match n:
                case Node.Cons(val, next):
                    return val
                case Node.Nil:
                    return 0

        # Construct a small list
        nil = Node.Nil()
        list1 = Node.Cons(10, nil)  # Nil should be automatically boxed if type says so

        self.assertEqual(get_head(list1), 10)
        self.assertEqual(get_head(nil), 0)

    def test_list_recursive(self):
        @verify
        def get_length(n: Node) -> i64:
            match n:
                case Node.Cons(val, next):
                    return 1 + get_length(next)
                case Node.Nil:
                    return 0

        nil = Node.Nil()
        list1 = Node.Cons(1, nil)
        list2 = Node.Cons(2, list1)
        list3 = Node.Cons(3, list2)

        self.assertEqual(get_length(list3), 3)


if __name__ == "__main__":
    unittest.main()
