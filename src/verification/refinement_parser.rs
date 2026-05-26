use z3::ast::{Array, Ast, Bool, Int, Real};
use z3::Context;

pub fn parse_refinement<'ctx>(
    ctx: &'ctx Context,
    refinement: &str,
    v: &Int<'ctx>,
) -> Result<Bool<'ctx>, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    parse_bool_expr(ctx, &refinement, Some(v), None, None)
}

pub fn parse_real_refinement<'ctx>(
    ctx: &'ctx Context,
    refinement: &str,
    v: &Real<'ctx>,
) -> Result<Bool<'ctx>, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    parse_bool_expr(ctx, &refinement, None, Some(v), None)
}

pub fn parse_array_refinement<'ctx>(
    ctx: &'ctx Context,
    refinement: &str,
    v: &Array<'ctx>,
    is_real: bool,
) -> Result<Bool<'ctx>, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    if is_real {
        parse_bool_expr(
            ctx,
            &refinement,
            None,
            Some(&Real::new_const(ctx, "DUMMY")),
            Some(v),
        )
    } else {
        parse_bool_expr(
            ctx,
            &refinement,
            Some(&Int::new_const(ctx, "DUMMY")),
            None,
            Some(v),
        )
    }
}

fn parse_bool_expr<'ctx>(
    ctx: &'ctx Context,
    sexpr: &str,
    v_int: Option<&Int<'ctx>>,
    v_real: Option<&Real<'ctx>>,
    v_arr: Option<&Array<'ctx>>,
) -> Result<Bool<'ctx>, String> {
    let sexpr = sexpr.trim();
    if !sexpr.starts_with('(') {
        return Err(format!("Invalid boolean sexpr: {}", sexpr));
    }
    let inner = &sexpr[1..sexpr.len() - 1];
    let parts = split_sexpr_parts(inner);
    if parts.is_empty() {
        return Err("Empty sexpr".to_string());
    }

    match parts[0] {
        "and" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_bool_expr(ctx, part, v_int, v_real, v_arr)?);
            }
            let refs: Vec<&Bool<'ctx>> = sub_exprs.iter().collect();
            Ok(Bool::and(ctx, &refs))
        }
        "or" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_bool_expr(ctx, part, v_int, v_real, v_arr)?);
            }
            let refs: Vec<&Bool<'ctx>> = sub_exprs.iter().collect();
            Ok(Bool::or(ctx, &refs))
        }
        "not" => {
            if parts.len() != 2 {
                return Err("not expects exactly one argument".to_string());
            }
            Ok(parse_bool_expr(ctx, parts[1], v_int, v_real, v_arr)?.not())
        }
        "=" | "!=" | "<" | "<=" | ">" | ">=" => {
            if parts.len() != 3 {
                return Err(format!("Invalid comparison: {:?}", parts));
            }
            if v_real.is_some() {
                let lhs = parse_real_expr(ctx, parts[1], v_real, v_arr)?;
                let rhs = parse_real_expr(ctx, parts[2], v_real, v_arr)?;
                match parts[0] {
                    "=" => Ok(lhs._eq(&rhs)),
                    "!=" => Ok(lhs._eq(&rhs).not()),
                    "<" => Ok(lhs.lt(&rhs)),
                    "<=" => Ok(lhs.le(&rhs)),
                    ">" => Ok(lhs.gt(&rhs)),
                    ">=" => Ok(lhs.ge(&rhs)),
                    _ => unreachable!(),
                }
            } else {
                let lhs = parse_int_expr(ctx, parts[1], v_int, v_arr)?;
                let rhs = parse_int_expr(ctx, parts[2], v_int, v_arr)?;
                match parts[0] {
                    "=" => Ok(lhs._eq(&rhs)),
                    "!=" => Ok(lhs._eq(&rhs).not()),
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

fn parse_int_expr<'ctx>(
    ctx: &'ctx Context,
    expr: &str,
    v_int: Option<&Int<'ctx>>,
    v_arr: Option<&Array<'ctx>>,
) -> Result<Int<'ctx>, String> {
    let expr = expr.trim();
    if expr == "VALUE_PLACEHOLDER" {
        return v_int
            .cloned()
            .ok_or_else(|| "VALUE_PLACEHOLDER used but no Int value available".to_string());
    }
    if let Ok(val) = expr.parse::<i64>() {
        return Ok(Int::from_i64(ctx, val));
    }

    if !expr.starts_with('(') {
        return Ok(Int::new_const(ctx, expr));
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
                sub_exprs.push(parse_int_expr(ctx, part, v_int, v_arr)?);
            }
            let refs: Vec<&Int<'ctx>> = sub_exprs.iter().collect();
            Ok(Int::add(ctx, &refs))
        }
        "-" => {
            if parts.len() == 2 {
                Ok(parse_int_expr(ctx, parts[1], v_int, v_arr)?.unary_minus())
            } else if parts.len() == 3 {
                let lhs = parse_int_expr(ctx, parts[1], v_int, v_arr)?;
                let rhs = parse_int_expr(ctx, parts[2], v_int, v_arr)?;
                Ok(Int::sub(ctx, &[&lhs, &rhs]))
            } else {
                Err("- expects 1 or 2 arguments".to_string())
            }
        }
        "*" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_int_expr(ctx, part, v_int, v_arr)?);
            }
            let refs: Vec<&Int<'ctx>> = sub_exprs.iter().collect();
            Ok(Int::mul(ctx, &refs))
        }
        "/" | "div" => {
            if parts.len() != 3 {
                return Err("div expects 2 arguments".to_string());
            }
            let lhs = parse_int_expr(ctx, parts[1], v_int, v_arr)?;
            let rhs = parse_int_expr(ctx, parts[2], v_int, v_arr)?;
            Ok(lhs.div(&rhs))
        }
        "%" | "mod" => {
            if parts.len() != 3 {
                return Err("mod expects 2 arguments".to_string());
            }
            let lhs = parse_int_expr(ctx, parts[1], v_int, v_arr)?;
            let rhs = parse_int_expr(ctx, parts[2], v_int, v_arr)?;
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
                return Err("Only select from placeholder supported for now".to_string());
            };
            let idx = parse_int_expr(ctx, parts[2], v_int, v_arr)?;
            let res = arr.select(&idx);
            if let Some(i) = res.as_int() {
                Ok(i)
            } else if let Some(bv) = res.as_bv() {
                Ok(bv.to_int(true))
            } else {
                Err("select did not return an int or bitvector".to_string())
            }
        }
        _ => Err(format!("Unknown arithmetic operator: {}", parts[0])),
    }
}

fn parse_real_expr<'ctx>(
    ctx: &'ctx Context,
    expr: &str,
    v_real: Option<&Real<'ctx>>,
    v_arr: Option<&Array<'ctx>>,
) -> Result<Real<'ctx>, String> {
    let expr = expr.trim();
    if expr == "VALUE_PLACEHOLDER" {
        return v_real
            .cloned()
            .ok_or_else(|| "VALUE_PLACEHOLDER used but no Real value available".to_string());
    }
    if let Ok(val) = expr.parse::<f64>() {
        let numer = (val * 1000.0) as i32;
        return Ok(Real::from_real(ctx, numer, 1000));
    }

    if !expr.starts_with('(') {
        return Ok(Real::new_const(ctx, expr));
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
                sub_exprs.push(parse_real_expr(ctx, part, v_real, v_arr)?);
            }
            let refs: Vec<&Real<'ctx>> = sub_exprs.iter().collect();
            Ok(Real::add(ctx, &refs))
        }
        "-" => {
            if parts.len() == 2 {
                Ok(parse_real_expr(ctx, parts[1], v_real, v_arr)?.unary_minus())
            } else if parts.len() == 3 {
                let lhs = parse_real_expr(ctx, parts[1], v_real, v_arr)?;
                let rhs = parse_real_expr(ctx, parts[2], v_real, v_arr)?;
                Ok(Real::sub(ctx, &[&lhs, &rhs]))
            } else {
                Err("- expects 1 or 2 arguments".to_string())
            }
        }
        "*" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_real_expr(ctx, part, v_real, v_arr)?);
            }
            let refs: Vec<&Real<'ctx>> = sub_exprs.iter().collect();
            Ok(Real::mul(ctx, &refs))
        }
        "/" | "div" => {
            if parts.len() != 3 {
                return Err("div expects 2 arguments".to_string());
            }
            let lhs = parse_real_expr(ctx, parts[1], v_real, v_arr)?;
            let rhs = parse_real_expr(ctx, parts[2], v_real, v_arr)?;
            Ok(lhs.div(&rhs))
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
                return Err("Only select from placeholder supported for now".to_string());
            };
            let idx = parse_int_expr(ctx, parts[2], None, v_arr)?;
            let res = arr.select(&idx);
            res.as_real()
                .ok_or_else(|| "select did not return a real".to_string())
        }
        _ => Err(format!("Unknown real arithmetic operator: {}", parts[0])),
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
