import unittest
from lila import verify, f32x4, f64, i64
from typing import overload


@overload
def process(data: f32x4) -> f32x4: ...


@overload
def process(data: f64) -> f64: ...


@verify
def process(data):
    return data * 2.0


class TestOverloading(unittest.TestCase):
    def test_basic_overloading(self):
        # Test SIMD overload
        v = f32x4(1.0, 2.0, 3.0, 4.0)
        res_v = process(v)
        self.assertEqual(res_v.f0, 2.0)
        self.assertEqual(res_v.f1, 4.0)
        self.assertEqual(res_v.f2, 6.0)
        self.assertEqual(res_v.f3, 8.0)

        # Test float overload
        res_f = process(1.5)
        self.assertEqual(res_f, 3.0)

    def test_overload_selection_priority(self):
        # If we have overloads for both i64 and f64
        @overload
        def scale(x: i64) -> i64: ...

        @overload
        def scale(x: f64) -> f64: ...

        @verify
        def scale(x):
            return x + x

        self.assertEqual(scale(5), 10)
        self.assertEqual(scale(1.5), 3.0)

    def test_multi_arg_overloading(self):
        @overload
        def combine(a: i64, b: i64) -> i64: ...

        @overload
        def combine(a: f64, b: f64) -> f64: ...

        @verify
        def combine(a, b):
            return a * b

        self.assertEqual(combine(10, 20), 200)
        self.assertAlmostEqual(combine(2.5, 4.0), 10.0)

    def test_method_overloading(self):
        from lila import struct, f32

        @struct
        class Math:
            factor: f32

            @overload
            def apply(self, val: i64) -> i64: ...

            @overload
            def apply(self, val: f32) -> f32: ...

            @verify
            def apply(self, val):
                # factor is f32, so i64 * f32 results in f32 usually,
                # but let's just do something simple.
                return val + val

        m = Math(2.0)
        self.assertEqual(m.apply(5), 10)
        self.assertAlmostEqual(m.apply(1.5), 3.0)

    def test_mixed_overloading(self):
        @overload
        def convert(x: i64) -> f64: ...

        @overload
        def convert(x: f64) -> i64: ...

        @verify
        def convert(x):
            # Lila's IR builder will insert appropriate casts
            # based on the matched overload's return annotation.
            return x

        self.assertIsInstance(convert(10), float)
        self.assertEqual(convert(10), 10.0)
        self.assertIsInstance(convert(3.14), int)
        self.assertEqual(convert(3.14), 3)

    def test_order_dependency(self):
        # The first matching overload should be chosen.
        @overload
        def first_match(x: i64) -> i64: ...

        @overload
        def first_match(x: f64) -> f64: ...

        @verify
        def first_match(x):
            if isinstance(x, int):
                return 1
            else:
                return 2

        self.assertEqual(first_match(10), 1)
        self.assertEqual(first_match(1.5), 2)

    def test_no_match(self):
        @overload
        def only_int(x: i64) -> i64: ...

        @verify
        def only_int(x):
            return x

        with self.assertRaises(TypeError) as cm:
            only_int(1.5)
        self.assertIn("No matching Lila overload found", str(cm.exception))

    def test_buffer_vs_scalar(self):
        from lila import Buffer
        import ctypes

        @overload
        def handle(data: i64) -> i64: ...

        @overload
        def handle(data: Buffer[i64]) -> i64: ...

        @verify
        def handle(data):
            if isinstance(data, int):
                return data
            else:
                if len(data) > 0:
                    return data[0]
                return 0

        self.assertEqual(handle(42), 42)
        buf = (ctypes.c_int64 * 1)(123)
        self.assertEqual(handle(buf), 123)

    def test_complex_mixed_combinations(self):
        @overload
        def mixed(a: i64, b: f64) -> f64: ...

        @overload
        def mixed(a: f64, b: i64) -> f64: ...

        @verify
        def mixed(a, b):
            return a + b

        self.assertEqual(mixed(10, 2.5), 12.5)
        self.assertEqual(mixed(2.5, 10), 12.5)

    def test_overload_with_refined_types(self):
        from typing import Annotated
        from lila import i64

        @overload
        def check(x: Annotated[i64, lambda x: x > 0]) -> i64: ...

        @overload
        def check(x: Annotated[i64, lambda x: x <= 0]) -> i64: ...

        @verify
        def check(x):
            return x

        # Note: Currently Lila matches overloads based on base type names.
        # This test ensures that even with refinements, it picks the base i64 match correctly.
        self.assertEqual(check(10), 10)
        self.assertEqual(check(-10), -10)


if __name__ == "__main__":
    unittest.main()
