import unittest
from lirien import verify, struct, enum, i64


@struct
class SomePayload:
    val: i64


@struct
class Empty:
    pass


@enum
class Option:
    Some: SomePayload
    NoneVariant: Empty


@verify
def create_some(x: i64) -> i64:
    s = SomePayload(x)
    opt = Option.Some(s)
    if opt.is_Some():
        return opt.as_Some().val
    return -1


@verify
def create_none() -> i64:
    opt = Option.NoneVariant(Empty())
    if opt.is_NoneVariant():
        return 0
    return 1


@verify
def check_is_variant(x: i64) -> i64:
    s = SomePayload(x)
    opt = Option.Some(s)
    res = 0
    if opt.is_Some():
        res = res + 1
    if not opt.is_NoneVariant():
        res = res + 2
    return res


class TestEnums(unittest.TestCase):
    def test_enum_creation_and_extraction(self):
        self.assertEqual(create_some(42), 42)
        self.assertEqual(create_some(100), 100)

    def test_none_variant(self):
        self.assertEqual(create_none(), 0)

    def test_is_variant(self):
        self.assertEqual(check_is_variant(10), 3)

    def test_enum_repr(self):
        s = SomePayload(42)
        opt_some = Option.Some(s)
        opt_none = Option.NoneVariant(Empty())
        self.assertEqual(repr(opt_some), "Option.Some(SomePayload(val=42))")
        self.assertEqual(repr(opt_none), "Option.NoneVariant(Empty())")

    def test_enum_eq(self):
        s1 = SomePayload(42)
        s2 = SomePayload(42)
        s3 = SomePayload(100)
        self.assertEqual(Option.Some(s1), Option.Some(s2))
        self.assertNotEqual(Option.Some(s1), Option.Some(s3))
        self.assertNotEqual(Option.Some(s1), Option.NoneVariant(Empty()))


if __name__ == "__main__":
    unittest.main()
