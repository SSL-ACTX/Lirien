import unittest
from lila import verify, i64, f64, Result, Ok, Err

Result_i64 = Result[i64, i64]
Result_f64 = Result[f64, i64]


@verify
def divide(a: i64, b: i64) -> Result_i64:
    if b == 0:
        return Err(0)
    return Ok(a // b)


@verify
def safe_div_sum(a: i64, b: i64) -> i64:
    res = divide(a, b)
    match res:
        case Result_i64.Ok(val):
            return val
        case Result_i64.Err(err):
            return -1
    return -2


@verify
def divide_f64(a: f64, b: f64) -> Result_f64:
    if b == 0.0:
        return Err(1)
    return Ok(a / b)


@verify
def process_f64(a: f64, b: f64) -> f64:
    res = divide_f64(a, b)
    match res:
        case Result_f64.Ok(val):
            return val
        case Result_f64.Err(_):
            return -999.0
    return -1.0


class TestResult(unittest.TestCase):
    def test_basic_result(self):
        # 10 // 2 = 5
        res = safe_div_sum(10, 2)
        self.assertEqual(res, 5)

        # 10 // 0 = Err(0) -> -1
        res2 = safe_div_sum(10, 0)
        self.assertEqual(res2, -1)

    def test_f64_result(self):
        res = process_f64(10.0, 2.0)
        self.assertEqual(res, 5.0)

        res2 = process_f64(10.0, 0.0)
        self.assertEqual(res2, -999.0)


if __name__ == "__main__":
    unittest.main()
