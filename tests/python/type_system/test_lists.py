import unittest
from lirien import verify, List, i64, Refined, VerificationError


class TestLists(unittest.TestCase):
    def test_list_basic(self):
        @verify
        def build_and_get_list() -> List[i64]:
            l = List[i64]()
            l.append(42)
            l.append(100)
            return l

        lst = build_and_get_list()
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], 42)
        self.assertEqual(lst[1], 100)

    def test_list_len_and_indexing(self):
        @verify
        def list_ops(l: List[i64]) -> i64:
            if len(l) > 0:
                l[0] = 77
                return len(l) + l[0]
            return 0

        lst = List[i64]()
        lst.append(10)
        res = list_ops(lst)
        self.assertEqual(res, 78)  # len=1 + 77 = 78
        self.assertEqual(lst[0], 77)

    def test_list_bounds_checking_load(self):
        with self.assertRaisesRegex(VerificationError, "Potential out-of-bounds"):

            @verify
            def load_unsafe(l: List[i64], idx: i64) -> i64:
                return l[idx]

    def test_list_bounds_checking_store(self):
        with self.assertRaisesRegex(VerificationError, "Potential out-of-bounds"):

            @verify
            def store_unsafe(l: List[i64], idx: i64):
                l[idx] = 42

    def test_list_bounds_checking_safe(self):
        IdxVal = Refined[i64, lambda x: (x >= 0) & (x < 1)]

        @verify
        def load_safe(l: List[i64], idx: IdxVal) -> i64:
            if len(l) > idx:
                return l[idx]
            return 0

        lst = List[i64]()
        lst.append(99)
        self.assertEqual(load_safe(lst, 0), 99)

    def test_list_of_structs(self):
        from lirien import struct

        @struct
        class Point:
            x: i64
            y: i64

        @verify
        def struct_list_ops() -> List[Point]:
            l = List[Point]()
            l.append(Point(10, 20))
            l.append(Point(30, 40))
            # Mutate one element
            if len(l) > 0:
                l[0] = Point(15, 25)
            return l

        lst = struct_list_ops()
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0].x, 15)
        self.assertEqual(lst[0].y, 25)
        self.assertEqual(lst[1].x, 30)
        self.assertEqual(lst[1].y, 40)

    def test_list_of_floats(self):
        from lirien import f64

        @verify
        def float_list_ops() -> List[f64]:
            l = List[f64]()
            l.append(1.5)
            l.append(2.5)
            if len(l) > 1:
                l[1] = 3.5
            return l

        lst = float_list_ops()
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], 1.5)
        self.assertEqual(lst[1], 3.5)

    def test_list_pep585_alias(self):
        @verify
        def build_and_get_list_pep585() -> list[i64]:
            l = list[i64]()
            l.append(42)
            l.append(100)
            return l

        lst = build_and_get_list_pep585()
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], 42)
        self.assertEqual(lst[1], 100)

    def test_list_comprehension_range(self):
        @verify
        def comp_range(n: i64) -> list[i64]:
            return [x * 2 for x in range(n)]

        lst = comp_range(5)
        self.assertEqual(len(lst), 5)
        self.assertEqual(lst[0], 0)
        self.assertEqual(lst[1], 2)
        self.assertEqual(lst[2], 4)
        self.assertEqual(lst[3], 6)
        self.assertEqual(lst[4], 8)

    def test_list_comprehension_list(self):
        @verify
        def comp_list(l: list[i64]) -> list[i64]:
            return [x + 1 for x in l]

        l = List[i64]()
        l.append(10)
        l.append(20)
        lst = comp_list(l)
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], 11)
        self.assertEqual(lst[1], 21)

    def test_list_comprehension_filter(self):
        @verify
        def comp_filter(l: list[i64]) -> list[i64]:
            return [x for x in l if x > 10]

        l = List[i64]()
        l.append(5)
        l.append(15)
        l.append(8)
        l.append(20)
        lst = comp_filter(l)
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], 15)
        self.assertEqual(lst[1], 20)


if __name__ == "__main__":
    unittest.main()
