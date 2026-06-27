use crate::refinement::resolver::Resolver;
use crate::refinement::utils::split_sexpr_parts;
use z3::ast::{Array, Int, BV};

pub(crate) fn parse_int_expr(
    expr: &str,
    v_int: Option<&Int>,
    v_arr: Option<&Array>,
    v_bv: Option<&BV>,
    resolver: Option<&Resolver>,
) -> Result<Int, String> {
    let expr = expr.trim();
    if expr == "VALUE_PLACEHOLDER" {
        return v_int
            .cloned()
            .ok_or_else(|| "VALUE_PLACEHOLDER used but no Int value available".to_string());
    }
    if let Some(r) = resolver {
        if let Some(i) = r.resolve_int(expr) {
            return Ok(i);
        }
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
                sub_exprs.push(parse_int_expr(part, v_int, v_arr, v_bv, resolver)?);
            }
            let refs: Vec<&Int> = sub_exprs.iter().collect();
            Ok(Int::add(&refs))
        }
        "-" => {
            if parts.len() == 2 {
                Ok(parse_int_expr(parts[1], v_int, v_arr, v_bv, resolver)?.unary_minus())
            } else if parts.len() == 3 {
                let lhs = parse_int_expr(parts[1], v_int, v_arr, v_bv, resolver)?;
                let rhs = parse_int_expr(parts[2], v_int, v_arr, v_bv, resolver)?;
                Ok(Int::sub(&[&lhs, &rhs]))
            } else {
                Err("- expects 1 or 2 arguments".to_string())
            }
        }
        "*" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_int_expr(part, v_int, v_arr, v_bv, resolver)?);
            }
            let refs: Vec<&Int> = sub_exprs.iter().collect();
            Ok(Int::mul(&refs))
        }
        "/" | "div" => {
            if parts.len() != 3 {
                return Err("div expects 2 arguments".to_string());
            }
            let lhs = parse_int_expr(parts[1], v_int, v_arr, v_bv, resolver)?;
            let rhs = parse_int_expr(parts[2], v_int, v_arr, v_bv, resolver)?;
            Ok(lhs.div(&rhs))
        }
        "%" | "mod" => {
            if parts.len() != 3 {
                return Err("mod expects 2 arguments".to_string());
            }
            let lhs = parse_int_expr(parts[1], v_int, v_arr, v_bv, resolver)?;
            let rhs = parse_int_expr(parts[2], v_int, v_arr, v_bv, resolver)?;
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
            let idx = parse_int_expr(parts[2], v_int, v_arr, v_bv, resolver)?;
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
            let cond = super::booleans::parse_bool_expr(
                parts[1], v_int, None, None, v_arr, v_bv, resolver,
            )?;
            let then = parse_int_expr(parts[2], v_int, v_arr, v_bv, resolver)?;
            let orelse = parse_int_expr(parts[3], v_int, v_arr, v_bv, resolver)?;
            Ok(cond.ite(&then, &orelse))
        }
        "&" | "|" | "^" | "<<" | ">>" | "~" => {
            // Handle bitwise by converting to BV, performing op, and converting back to Int
            if v_bv.is_none() && v_int.is_none() && resolver.is_none() {
                return Err("Bitwise op used but no value available".to_string());
            }
            let lhs_bv = super::bitvectors::parse_bv_expr(expr, v_bv, v_int, resolver)?;
            Ok(lhs_bv.to_int(true))
        }
        _ => Err(format!("Unknown arithmetic operator: {}", parts[0])),
    }
}
