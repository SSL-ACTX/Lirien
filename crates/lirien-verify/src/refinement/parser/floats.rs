use crate::refinement::resolver::Resolver;
use crate::refinement::utils::split_sexpr_parts;
use crate::refinement::utils::unify_floats;
use std::ops::Neg;
use z3::ast::{Array, Float, RoundingMode};

pub(crate) fn parse_float_expr(
    expr: &str,
    v_float: Option<&Float>,
    v_arr: Option<&Array>,
    resolver: Option<&Resolver>,
) -> Result<Float, String> {
    let expr = expr.trim();
    if expr == "VALUE_PLACEHOLDER" {
        return v_float
            .cloned()
            .ok_or_else(|| "VALUE_PLACEHOLDER used but no Float value available".to_string());
    }
    if let Some(r) = resolver {
        if let Some(f) = r.resolve_float(expr) {
            return Ok(f);
        }
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
            let lhs = parse_float_expr(parts[1], v_float, v_arr, resolver)?;
            let rhs = parse_float_expr(parts[2], v_float, v_arr, resolver)?;
            let (l, r) = unify_floats(&lhs, &rhs);
            Ok(rm.add(&l, &r))
        }
        "-" => {
            if parts.len() == 2 {
                Ok(parse_float_expr(parts[1], v_float, v_arr, resolver)?.neg())
            } else if parts.len() == 3 {
                let lhs = parse_float_expr(parts[1], v_float, v_arr, resolver)?;
                let rhs = parse_float_expr(parts[2], v_float, v_arr, resolver)?;
                let (l, r) = unify_floats(&lhs, &rhs);
                Ok(rm.sub(&l, &r))
            } else {
                Err("- expects 1 or 2 arguments".to_string())
            }
        }
        "*" => {
            if parts.len() != 3 {
                return Err("* expects 2 arguments".to_string());
            }
            let lhs = parse_float_expr(parts[1], v_float, v_arr, resolver)?;
            let rhs = parse_float_expr(parts[2], v_float, v_arr, resolver)?;
            let (l, r) = unify_floats(&lhs, &rhs);
            Ok(rm.mul(&l, &r))
        }
        "/" | "div" => {
            if parts.len() != 3 {
                return Err("div expects 2 arguments".to_string());
            }
            let lhs = parse_float_expr(parts[1], v_float, v_arr, resolver)?;
            let rhs = parse_float_expr(parts[2], v_float, v_arr, resolver)?;
            let (l, r) = unify_floats(&lhs, &rhs);
            Ok(rm.div(&l, &r))
        }
        "ite" => {
            if parts.len() != 4 {
                return Err("ite (if-then-else) expects 3 arguments".to_string());
            }
            let cond = super::booleans::parse_bool_expr(parts[1], None, None, v_float, v_arr, None, resolver)?;
            let then = parse_float_expr(parts[2], v_float, v_arr, resolver)?;
            let orelse = parse_float_expr(parts[3], v_float, v_arr, resolver)?;
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
            let idx = super::integers::parse_int_expr(parts[2], None, v_arr, None, resolver)?;
            let res = arr.select(&idx);
            res.as_float()
                .ok_or_else(|| "select did not return a float".to_string())
        }
        _ => Err(format!("Unknown float arithmetic operator: {}", parts[0])),
    }
}
