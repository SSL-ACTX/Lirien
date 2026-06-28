import unittest
from lirien import verify, i64


class TestExceptions(unittest.TestCase):
    def test_raise_and_catch_in_python(self):
        @verify
        def raise_val_error(x: i64) -> i64:
            if x < 0:
                raise ValueError
            return x

        self.assertEqual(raise_val_error(10), 10)
        with self.assertRaises(ValueError):
            raise_val_error(-5)

    def test_exception_propagation(self):
        @verify
        def raise_val_error(x: i64) -> i64:
            if x < 0:
                raise ValueError
            return x

        @verify
        def call_raise(x: i64) -> i64:
            return raise_val_error(x)

        self.assertEqual(call_raise(10), 10)
        with self.assertRaises(ValueError):
            call_raise(-5)

    def test_try_except_catch(self):
        @verify
        def raise_val_error(x: i64) -> i64:
            if x < 0:
                raise ValueError
            return x

        @verify
        def catch_val_error(x: i64) -> i64:
            try:
                raise_val_error(x)
            except ValueError:
                return 42
            return x

        self.assertEqual(catch_val_error(10), 10)
        self.assertEqual(catch_val_error(-5), 42)

    def test_try_except_multiple(self):
        @verify
        def catch_multiple(x: i64) -> i64:
            try:
                if x < 0:
                    raise ValueError
                elif x == 0:
                    raise TypeError
                else:
                    raise IndexError
            except ValueError:
                return 100
            except TypeError:
                return 200
            except IndexError:
                return 300
            return 0

        self.assertEqual(catch_multiple(-5), 100)
        self.assertEqual(catch_multiple(0), 200)
        self.assertEqual(catch_multiple(5), 300)


if __name__ == "__main__":
    unittest.main()
