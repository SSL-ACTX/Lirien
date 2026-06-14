import unittest
from lila import verify, struct, Box, i64, VerificationError
from typing import Optional


@struct
class Node:
    val: i64
    next: Optional[Box["Node"]]


class TestNullSafety(unittest.TestCase):
    def test_recursive_linked_list(self):
        @verify
        def sum_list(n: Optional[Box[Node]]) -> i64:
            if n is None:
                return 0
            # Automatic dereferencing of 'n' and 'n.next'
            return n.val + sum_list(n.next)

        # 1 -> 2 -> 3 -> None
        n3 = Node(val=3, next=None)
        n2 = Node(val=2, next=Box(n3))
        n1 = Node(val=1, next=Box(n2))

        self.assertEqual(sum_list(Box(n1)), 6)
        self.assertEqual(sum_list(None), 0)

    def test_null_dereference_fails(self):
        with self.assertRaises(VerificationError) as cm:

            @verify
            def unsafe_get(n: Optional[Box[Node]]) -> i64:
                # Dereferencing without 'is None' check
                return n.val

        self.assertIn("Potential null pointer dereference", str(cm.exception))

    def test_union_syntax(self):
        # Test 'Box[Node] | None' syntax (Python 3.10+)
        import sys

        if sys.version_info >= (3, 10):

            @verify
            def sum_list_union(n: Box[Node] | None) -> i64:
                if n is None:
                    return 0
                return n.val

            self.assertEqual(sum_list_union(Box(Node(42, None))), 42)
            self.assertEqual(sum_list_union(None), 0)

    def test_nested_box_unwrap(self):
        @verify
        def get_deep_val(n: Box[Box[Node]]) -> i64:
            # Should handle multiple layers of Box
            return (
                n.val.val
            )  # first .val unwraps outer Box, second .val accesses Node.val

        node = Node(val=1337, next=None)
        deep_box = Box(Box(node))
        self.assertEqual(get_deep_val(deep_box), 1337)

    def test_mixed_optional_refined(self):
        # Optional[Box[i64]] is also supported for primitives
        @verify
        def get_val_or_zero(x: Optional[Box[i64]]) -> i64:
            if x is not None:
                return x.val
            return 0

        self.assertEqual(get_val_or_zero(Box(42)), 42)
        self.assertEqual(get_val_or_zero(None), 0)


if __name__ == "__main__":
    unittest.main()
