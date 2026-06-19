import unittest
from lirien import verify, enum, struct, i64


@struct
class SomePayload:
    val: i64


@struct
class Empty:
    pass


@enum
class Option:
    Some: SomePayload
    Empty: Empty


class TestMatchStatements(unittest.TestCase):
    def test_match_basic(self):
        @verify
        def unwrap_or_zero(opt: Option) -> i64:
            match opt:
                case Option.Some(s):
                    return s.val
                case Option.Empty:
                    return 0

        self.assertEqual(unwrap_or_zero(Option.Some(10)), 10)
        self.assertEqual(unwrap_or_zero(Option.Empty()), 0)


if __name__ == "__main__":
    unittest.main()
