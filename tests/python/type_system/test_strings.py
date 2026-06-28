import unittest
from lirien import verify, i64, Refined, VerificationError


class TestStrings(unittest.TestCase):
    def test_string_const_and_len(self):
        @verify
        def get_const_len() -> i64:
            s = "hello"
            return len(s)

        @verify
        def get_param_len(s: str) -> i64:
            return len(s)

        self.assertEqual(get_const_len(), 5)
        self.assertEqual(get_param_len("world"), 5)
        self.assertEqual(get_param_len(""), 0)

    def test_string_concat(self):
        @verify
        def concat_strings(s1: str, s2: str) -> str:
            return s1 + s2

        self.assertEqual(concat_strings("hello ", "world"), "hello world")
        self.assertEqual(concat_strings("", ""), "")

    def test_string_eq_ne(self):
        @verify
        def compare_eq(s1: str, s2: str) -> bool:
            return s1 == s2

        @verify
        def compare_ne(s1: str, s2: str) -> bool:
            return s1 != s2

        self.assertTrue(compare_eq("abc", "abc"))
        self.assertFalse(compare_eq("abc", "def"))
        self.assertTrue(compare_ne("abc", "def"))
        self.assertFalse(compare_ne("abc", "abc"))

    def test_string_indexing_basic(self):
        @verify
        def get_first_char(s: str) -> str:
            if len(s) > 0:
                return s[0]
            return ""

        self.assertEqual(get_first_char("hello"), "h")
        self.assertEqual(get_first_char(""), "")

    def test_string_indexing_unsafe(self):
        with self.assertRaisesRegex(VerificationError, "Potential out-of-bounds"):

            @verify
            def get_char_unsafe(s: str, idx: i64) -> str:
                return s[idx]

    def test_string_indexing_safe(self):
        IdxVal = Refined[i64, lambda x: (x >= 0) & (x < 3)]

        @verify
        def get_char_safe(s: str, idx: IdxVal) -> str:
            if len(s) > idx:
                return s[idx]
            return ""

        self.assertEqual(get_char_safe("hello", 1), "e")
        self.assertEqual(get_char_safe("abc", 2), "c")

    def test_string_slicing(self):
        @verify
        def slice_string(s: str, start: i64, end: i64) -> str:
            return s[start:end]

        @verify
        def slice_default_end(s: str, start: i64) -> str:
            return s[start:]

        self.assertEqual(slice_string("hello world", 0, 5), "hello")
        self.assertEqual(slice_string("hello world", 6, 11), "world")
        self.assertEqual(slice_default_end("hello world", 6), "world")


if __name__ == "__main__":
    unittest.main()
