import unittest
from lirien import verify, i64, Closure


@verify
def make_adder_lambda(x: i64) -> Closure[[i64], i64]:
    return lambda y: x + y


@verify
def make_adder_def(x: i64) -> Closure[[i64], i64]:
    def adder(y: i64) -> i64:
        return x + y

    return adder


@verify
def make_multiplier_and_adder(a: i64, b: i64) -> Closure[[i64], i64]:
    # Capture multiple variables
    def calc(x: i64) -> i64:
        return x * a + b

    return calc


@verify
def make_nested_adder(x: i64) -> Closure[[i64], Closure[[i64], i64]]:
    # Nested closures: Closure creating another closure
    def middle(y: i64) -> Closure[[i64], i64]:
        def inner(z: i64) -> i64:
            return x + y + z

        return inner

    return middle


@verify
def compose(f: Closure[[i64], i64], g: Closure[[i64], i64]) -> Closure[[i64], i64]:
    # Function composition
    return lambda x: f(g(x))


class TestClosures(unittest.TestCase):
    def test_lambda_closure(self):
        adder = make_adder_lambda(5)
        self.assertEqual(adder(10), 15)
        self.assertEqual(adder(-2), 3)

    def test_nested_def_closure(self):
        adder = make_adder_def(5)
        self.assertEqual(adder(10), 15)
        self.assertEqual(adder(-2), 3)

    def test_multiple_captures(self):
        calc = make_multiplier_and_adder(10, 3)
        self.assertEqual(calc(5), 53)
        self.assertEqual(calc(0), 3)

    def test_deeply_nested_defs(self):
        # make_nested_adder(1) returns a closure that captures x=1
        add_to_1 = make_nested_adder(1)
        # add_to_1(10) returns a closure that captures x=1, y=10
        add_to_11 = add_to_1(10)
        # add_to_11(100) returns 1 + 10 + 100 = 111
        self.assertEqual(add_to_11(100), 111)

    def test_composition(self):
        add5 = make_adder_lambda(5)
        mul10_add3 = make_multiplier_and_adder(10, 3)

        # (x * 10 + 3) + 5
        composed = compose(add5, mul10_add3)
        self.assertEqual(composed(2), 28)

        # (x + 5) * 10 + 3
        composed2 = compose(mul10_add3, add5)
        self.assertEqual(composed2(2), 73)

    def test_closure_independence(self):
        # Ensure different closure instances have independent state
        add5 = make_adder_def(5)
        add10 = make_adder_def(10)

        self.assertEqual(add5(1), 6)
        self.assertEqual(add10(1), 11)


if __name__ == "__main__":
    unittest.main()
