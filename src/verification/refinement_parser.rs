use std::ops::Neg;
use z3::ast::{Array, Ast, Bool, Float, Int, Real, RoundingMode, BV};

pub fn parse_refinement(refinement: &str, v: &Int, v_bv: Option<&BV>) -> Result<Bool, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    parse_bool_expr(&refinement, Some(v), None, None, None, v_bv)
}

pub fn parse_float_refinement(refinement: &str, v: &Float) -> Result<Bool, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    parse_bool_expr(&refinement, None, None, Some(v), None, None)
}

pub fn parse_real_refinement(refinement: &str, v: &Real) -> Result<Bool, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    parse_bool_expr(&refinement, None, Some(v), None, None, None)
}

pub fn parse_array_refinement(refinement: &str, v: &Array, is_float: bool) -> Result<Bool, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    if is_float {
        parse_bool_expr(
            &refinement,
            None,
            None,
            Some(&Float::new_const_double("DUMMY")),
            Some(v),
            None,
        )
    } else {
        parse_bool_expr(
            &refinement,
            Some(&Int::new_const("DUMMY")),
            None,
            None,
            Some(v),
            Some(&BV::new_const("DUMMY", 64)),
        )
    }
}

fn parse_bool_expr(
    sexpr: &str,
    v_int: Option<&Int>,
    v_real: Option<&Real>,
    v_float: Option<&Float>,
    v_arr: Option<&Array>,
    v_bv: Option<&BV>,
) -> Result<Bool, String> {
    let sexpr = sexpr.trim();
    if sexpr == "true" {
        return Ok(Bool::from_bool(true));
    }
    if sexpr == "false" {
        return Ok(Bool::from_bool(false));
    }

    if !sexpr.starts_with('(') {
        return Err(format!("Invalid boolean sexpr: {}", sexpr));
    }
    let inner = &sexpr[1..sexpr.len() - 1];
    let parts = split_sexpr_parts(inner);
    if parts.is_empty() {
        return Err("Empty sexpr".to_string());
    }

    match parts[0] {
        "and" | "&" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_bool_expr(part, v_int, v_real, v_float, v_arr, v_bv)?);
            }
            let refs: Vec<&Bool> = sub_exprs.iter().collect();
            Ok(Bool::and(&refs))
        }
        "or" | "|" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_bool_expr(part, v_int, v_real, v_float, v_arr, v_bv)?);
            }
            let refs: Vec<&Bool> = sub_exprs.iter().collect();
            Ok(Bool::or(&refs))
        }
        "xor" | "^" => {
            if parts.len() != 3 {
                return Err("xor expects 2 arguments".to_string());
            }
            let lhs = parse_bool_expr(parts[1], v_int, v_real, v_float, v_arr, v_bv)?;
            let rhs = parse_bool_expr(parts[2], v_int, v_real, v_float, v_arr, v_bv)?;
            Ok(lhs.xor(&rhs))
        }
        "not" | "~" => {
            if parts.len() != 2 {
                return Err("not expects exactly one argument".to_string());
            }
            Ok(parse_bool_expr(parts[1], v_int, v_real, v_float, v_arr, v_bv)?.not())
        }
        "ite" => {
            if parts.len() != 4 {
                return Err("ite (if-then-else) expects 3 arguments".to_string());
            }
            let cond = parse_bool_expr(parts[1], v_int, v_real, v_float, v_arr, v_bv)?;
            let then = parse_bool_expr(parts[2], v_int, v_real, v_float, v_arr, v_bv)?;
            let orelse = parse_bool_expr(parts[3], v_int, v_real, v_float, v_arr, v_bv)?;
            Ok(cond.ite(&then, &orelse))
        }
        "=" | "!=" | "<" | "<=" | ">" | ">=" => {
            if parts.len() != 3 {
                return Err(format!("Invalid comparison: {:?}", parts));
            }
            if v_float.is_some() {
                let lhs = parse_float_expr(parts[1], v_float, v_arr)?;
                let rhs = parse_float_expr(parts[2], v_float, v_arr)?;
                match parts[0] {
                    "=" => Ok(lhs.eq(&rhs)),
                    "!=" => Ok(lhs.eq(&rhs).not()),
                    "<" => Ok(lhs.lt(&rhs)),
                    "<=" => Ok(lhs.le(&rhs)),
                    ">" => Ok(lhs.gt(&rhs)),
                    ">=" => Ok(lhs.ge(&rhs)),
                    _ => unreachable!(),
                }
            } else if v_real.is_some() {
                let lhs = parse_real_expr(parts[1], v_real, v_arr)?;
                let rhs = parse_real_expr(parts[2], v_real, v_arr)?;
                match parts[0] {
                    "=" => Ok(lhs.eq(&rhs)),
                    "!=" => Ok(lhs.eq(&rhs).not()),
                    "<" => Ok(lhs.lt(&rhs)),
                    "<=" => Ok(lhs.le(&rhs)),
                    ">" => Ok(lhs.gt(&rhs)),
                    ">=" => Ok(lhs.ge(&rhs)),
                    _ => unreachable!(),
                }
            } else {
                let lhs = parse_int_expr(parts[1], v_int, v_arr, v_bv)?;
                let rhs = parse_int_expr(parts[2], v_int, v_arr, v_bv)?;
                match parts[0] {
                    "=" => Ok(lhs.eq(&rhs)),
                    "!=" => Ok(lhs.eq(&rhs).not()),
                    "<" => Ok(lhs.lt(&rhs)),
                    "<=" => Ok(lhs.le(&rhs)),
                    ">" => Ok(lhs.gt(&rhs)),
                    ">=" => Ok(lhs.ge(&rhs)),
                    _ => unreachable!(),
                }
            }
        }
        _ => Err(format!("Unknown boolean operator: {}", parts[0])),
    }
}

fn parse_int_expr(
    expr: &str,
    v_int: Option<&Int>,
    v_arr: Option<&Array>,
    v_bv: Option<&BV>,
) -> Result<Int, String> {
    let expr = expr.trim();
    if expr == "VALUE_PLACEHOLDER" {
        return v_int
            .cloned()
            .ok_or_else(|| "VALUE_PLACEHOLDER used but no Int value available".to_string());
    }
    if let Ok(val) = expr.parse::<i64>() {
        return Ok(Int::from_i64(val));
    }

    if !expr.starts_with('(') {
        return Ok(Int::new_const(expr));
    }

    let inner = &expr[1..expr.len() - 1];
    let parts = split_sexpr_parts(inner);
    if parts.is_empty() {
        return Err("Empty arithmetic sexpr".to_string());
    }

    match parts[0] {
        "+" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_int_expr(part, v_int, v_arr, v_bv)?);
            }
            let refs: Vec<&Int> = sub_exprs.iter().collect();
            Ok(Int::add(&refs))
        }
        "-" => {
            if parts.len() == 2 {
                Ok(parse_int_expr(parts[1], v_int, v_arr, v_bv)?.unary_minus())
            } else if parts.len() == 3 {
                let lhs = parse_int_expr(parts[1], v_int, v_arr, v_bv)?;
                let rhs = parse_int_expr(parts[2], v_int, v_arr, v_bv)?;
                Ok(Int::sub(&[&lhs, &rhs]))
            } else {
                Err("- expects 1 or 2 arguments".to_string())
            }
        }
        "*" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_int_expr(part, v_int, v_arr, v_bv)?);
            }
            let refs: Vec<&Int> = sub_exprs.iter().collect();
            Ok(Int::mul(&refs))
        }
        "/" | "div" => {
            if parts.len() != 3 {
                return Err("div expects 2 arguments".to_string());
            }
            let lhs = parse_int_expr(parts[1], v_int, v_arr, v_bv)?;
            let rhs = parse_int_expr(parts[2], v_int, v_arr, v_bv)?;
            Ok(lhs.div(&rhs))
        }
        "%" | "mod" => {
            if parts.len() != 3 {
                return Err("mod expects 2 arguments".to_string());
            }
            let lhs = parse_int_expr(parts[1], v_int, v_arr, v_bv)?;
            let rhs = parse_int_expr(parts[2], v_int, v_arr, v_bv)?;
            Ok(lhs.rem(&rhs))
        }
        "select" => {
            if parts.len() != 3 {
                return Err("select expects 2 arguments".to_string());
            }
            let arr = if parts[1] == "VALUE_PLACEHOLDER" {
                v_arr.cloned().ok_or_else(|| {
                    "VALUE_PLACEHOLDER used in select but no Array available".to_string()
                })?
            } else {
                // If it's not a placeholder, assume it's a named array constant
                v_arr.cloned().ok_or_else(|| {
                    format!(
                        "Array '{}' used in select but no Array context available",
                        parts[1]
                    )
                })?
            };
            let idx = parse_int_expr(parts[2], v_int, v_arr, v_bv)?;
            let res = arr.select(&idx);
            if let Some(i) = res.as_int() {
                Ok(i)
            } else if let Some(bv) = res.as_bv() {
                Ok(bv.to_int(true))
            } else {
                Err("select did not return an int or bitvector".to_string())
            }
        }
        "ite" => {
            if parts.len() != 4 {
                return Err("ite (if-then-else) expects 3 arguments".to_string());
            }
            let cond = parse_bool_expr(parts[1], v_int, None, None, v_arr, v_bv)?;
            let then = parse_int_expr(parts[2], v_int, v_arr, v_bv)?;
            let orelse = parse_int_expr(parts[3], v_int, v_arr, v_bv)?;
            Ok(cond.ite(&then, &orelse))
        }
        "&" | "|" | "^" | "<<" | ">>" | "~" => {
            // Handle bitwise by converting to BV, performing op, and converting back to Int
            if v_bv.is_none() && v_int.is_none() {
                return Err("Bitwise op used but no value available".to_string());
            }
            let lhs_bv = parse_bv_expr(expr, v_bv, v_int)?;
            Ok(lhs_bv.to_int(true))
        }
        _ => Err(format!("Unknown arithmetic operator: {}", parts[0])),
    }
}

fn parse_bv_expr(expr: &str, v_bv: Option<&BV>, v_int: Option<&Int>) -> Result<BV, String> {
    let expr = expr.trim();
    if expr == "VALUE_PLACEHOLDER" {
        return v_bv
            .cloned()
            .ok_or_else(|| "VALUE_PLACEHOLDER used but no BV available".to_string());
    }
    if let Ok(val) = expr.parse::<i64>() {
        return Ok(BV::from_i64(val, 64));
    }

    if !expr.starts_with('(') {
        return Ok(BV::new_const(expr, 64));
    }

    let inner = &expr[1..expr.len() - 1];
    let parts = split_sexpr_parts(inner);
    if parts.is_empty() {
        return Err("Empty bitwise sexpr".to_string());
    }

    match parts[0] {
        "&" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int)?;
            Ok(lhs.bvand(&rhs))
        }
        "|" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int)?;
            Ok(lhs.bvor(&rhs))
        }
        "^" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int)?;
            Ok(lhs.bvxor(&rhs))
        }
        "~" => {
            let operand = parse_bv_expr(parts[1], v_bv, v_int)?;
            Ok(operand.bvnot())
        }
        "<<" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int)?;
            Ok(lhs.bvshl(&rhs))
        }
        ">>" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int)?;
            Ok(lhs.bvashr(&rhs))
        }
        // Arithmetic ops in BV context
        "+" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int)?;
            Ok(lhs.bvadd(&rhs))
        }
        "-" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int)?;
            Ok(lhs.bvsub(&rhs))
        }
        "VALUE_PLACEHOLDER" => v_bv
            .cloned()
            .ok_or_else(|| "No BV value available".to_string()),
        _ => {
            // Fallback: try parsing as Int and convert to BV
            let int_val = parse_int_expr(expr, v_int, None, v_bv)?;
            Ok(BV::from_int(&int_val, 64))
        }
    }
}

fn parse_real_expr(
    expr: &str,
    v_real: Option<&Real>,
    v_arr: Option<&Array>,
) -> Result<Real, String> {
    let expr = expr.trim();
    if expr == "VALUE_PLACEHOLDER" {
        return v_real
            .cloned()
            .ok_or_else(|| "VALUE_PLACEHOLDER used but no Real value available".to_string());
    }
    if let Ok(val) = expr.parse::<f64>() {
        let numer = (val * 1000.0) as i64;
        return Ok(Real::from_rational(numer, 1000));
    }

    if !expr.starts_with('(') {
        return Ok(Real::new_const(expr));
    }

    let inner = &expr[1..expr.len() - 1];
    let parts = split_sexpr_parts(inner);
    if parts.is_empty() {
        return Err("Empty real arithmetic sexpr".to_string());
    }

    match parts[0] {
        "+" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_real_expr(part, v_real, v_arr)?);
            }
            let refs: Vec<&Real> = sub_exprs.iter().collect();
            Ok(Real::add(&refs))
        }
        "-" => {
            if parts.len() == 2 {
                Ok(parse_real_expr(parts[1], v_real, v_arr)?.unary_minus())
            } else if parts.len() == 3 {
                let lhs = parse_real_expr(parts[1], v_real, v_arr)?;
                let rhs = parse_real_expr(parts[2], v_real, v_arr)?;
                Ok(Real::sub(&[&lhs, &rhs]))
            } else {
                Err("- expects 1 or 2 arguments".to_string())
            }
        }
        "*" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_real_expr(part, v_real, v_arr)?);
            }
            let refs: Vec<&Real> = sub_exprs.iter().collect();
            Ok(Real::mul(&refs))
        }
        "/" | "div" => {
            if parts.len() != 3 {
                return Err("div expects 2 arguments".to_string());
            }
            let lhs = parse_real_expr(parts[1], v_real, v_arr)?;
            let rhs = parse_real_expr(parts[2], v_real, v_arr)?;
            Ok(lhs.div(&rhs))
        }
        "ite" => {
            if parts.len() != 4 {
                return Err("ite (if-then-else) expects 3 arguments".to_string());
            }
            let cond = parse_bool_expr(parts[1], None, v_real, None, v_arr, None)?;
            let then = parse_real_expr(parts[2], v_real, v_arr)?;
            let orelse = parse_real_expr(parts[3], v_real, v_arr)?;
            Ok(cond.ite(&then, &orelse))
        }
        "select" => {
            if parts.len() != 3 {
                return Err("select expects 2 arguments".to_string());
            }
            let arr = if parts[1] == "VALUE_PLACEHOLDER" {
                v_arr.cloned().ok_or_else(|| {
                    "VALUE_PLACEHOLDER used in select but no Array available".to_string()
                })?
            } else {
                v_arr.cloned().ok_or_else(|| {
                    format!(
                        "Array '{}' used in select but no Array context available",
                        parts[1]
                    )
                })?
            };
            let idx = parse_int_expr(parts[2], None, v_arr, None)?;
            let res = arr.select(&idx);
            res.as_real()
                .ok_or_else(|| "select did not return a real".to_string())
        }
        _ => Err(format!("Unknown real arithmetic operator: {}", parts[0])),
    }
}

fn parse_float_expr(
    expr: &str,
    v_float: Option<&Float>,
    v_arr: Option<&Array>,
) -> Result<Float, String> {
    let expr = expr.trim();
    if expr == "VALUE_PLACEHOLDER" {
        return v_float
            .cloned()
            .ok_or_else(|| "VALUE_PLACEHOLDER used but no Float value available".to_string());
    }
    if let Ok(val) = expr.parse::<f64>() {
        return Ok(Float::from_f64(val));
    }

    if !expr.starts_with('(') {
        return Ok(Float::new_const_double(expr));
    }

    let inner = &expr[1..expr.len() - 1];
    let parts = split_sexpr_parts(inner);
    if parts.is_empty() {
        return Err("Empty float arithmetic sexpr".to_string());
    }

    let rm = RoundingMode::round_towards_zero();

    match parts[0] {
        "+" => {
            if parts.len() != 3 {
                return Err("+ expects 2 arguments".to_string());
            }
            let lhs = parse_float_expr(parts[1], v_float, v_arr)?;
            let rhs = parse_float_expr(parts[2], v_float, v_arr)?;
            Ok(rm.add(&lhs, &rhs))
        }
        "-" => {
            if parts.len() == 2 {
                Ok(parse_float_expr(parts[1], v_float, v_arr)?.neg())
            } else if parts.len() == 3 {
                let lhs = parse_float_expr(parts[1], v_float, v_arr)?;
                let rhs = parse_float_expr(parts[2], v_float, v_arr)?;
                Ok(rm.sub(&lhs, &rhs))
            } else {
                Err("- expects 1 or 2 arguments".to_string())
            }
        }
        "*" => {
            if parts.len() != 3 {
                return Err("* expects 2 arguments".to_string());
            }
            let lhs = parse_float_expr(parts[1], v_float, v_arr)?;
            let rhs = parse_float_expr(parts[2], v_float, v_arr)?;
            Ok(rm.mul(&lhs, &rhs))
        }
        "/" | "div" => {
            if parts.len() != 3 {
                return Err("div expects 2 arguments".to_string());
            }
            let lhs = parse_float_expr(parts[1], v_float, v_arr)?;
            let rhs = parse_float_expr(parts[2], v_float, v_arr)?;
            Ok(rm.div(&lhs, &rhs))
        }
        "ite" => {
            if parts.len() != 4 {
                return Err("ite (if-then-else) expects 3 arguments".to_string());
            }
            let cond = parse_bool_expr(parts[1], None, None, v_float, v_arr, None)?;
            let then = parse_float_expr(parts[2], v_float, v_arr)?;
            let orelse = parse_float_expr(parts[3], v_float, v_arr)?;
            Ok(cond.ite(&then, &orelse))
        }
        "select" => {
            if parts.len() != 3 {
                return Err("select expects 2 arguments".to_string());
            }
            let arr = if parts[1] == "VALUE_PLACEHOLDER" {
                v_arr.cloned().ok_or_else(|| {
                    "VALUE_PLACEHOLDER used in select but no Array available".to_string()
                })?
            } else {
                v_arr.cloned().ok_or_else(|| {
                    format!(
                        "Array '{}' used in select but no Array context available",
                        parts[1]
                    )
                })?
            };
            let idx = parse_int_expr(parts[2], None, v_arr, None)?;
            let res = arr.select(&idx);
            res.as_float()
                .ok_or_else(|| "select did not return a float".to_string())
        }
        _ => Err(format!("Unknown float arithmetic operator: {}", parts[0])),
    }
}

fn split_sexpr_parts(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut current_start = 0;
    let mut depth = 0;
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => depth -= 1,
            ' ' if depth == 0 => {
                let part = s[current_start..i].trim();
                if !part.is_empty() {
                    parts.push(part);
                }
                current_start = i + 1;
            }
            _ => {}
        }
    }
    let last_part = s[current_start..].trim();
    if !last_part.is_empty() {
        parts.push(last_part);
    }
    parts
}
