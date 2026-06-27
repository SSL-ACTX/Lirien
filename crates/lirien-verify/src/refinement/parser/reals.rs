use crate::refinement::resolver::Resolver;
use crate::refinement::utils::split_sexpr_parts;
use z3::ast::{Array, Real};

pub(crate) fn parse_real_expr(
    expr: &str,
    v_real: Option<&Real>,
    v_arr: Option<&Array>,
    resolver: Option<&Resolver>,
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
                sub_exprs.push(parse_real_expr(part, v_real, v_arr, resolver)?);
            }
            let refs: Vec<&Real> = sub_exprs.iter().collect();
            Ok(Real::add(&refs))
        }
        "-" => {
            if parts.len() == 2 {
                Ok(parse_real_expr(parts[1], v_real, v_arr, resolver)?.unary_minus())
            } else if parts.len() == 3 {
                let lhs = parse_real_expr(parts[1], v_real, v_arr, resolver)?;
                let rhs = parse_real_expr(parts[2], v_real, v_arr, resolver)?;
                Ok(Real::sub(&[&lhs, &rhs]))
            } else {
                Err("- expects 1 or 2 arguments".to_string())
            }
        }
        "*" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_real_expr(part, v_real, v_arr, resolver)?);
            }
            let refs: Vec<&Real> = sub_exprs.iter().collect();
            Ok(Real::mul(&refs))
        }
        "/" | "div" => {
            if parts.len() != 3 {
                return Err("div expects 2 arguments".to_string());
            }
            let lhs = parse_real_expr(parts[1], v_real, v_arr, resolver)?;
            let rhs = parse_real_expr(parts[2], v_real, v_arr, resolver)?;
            Ok(lhs.div(&rhs))
        }
        "ite" => {
            if parts.len() != 4 {
                return Err("ite (if-then-else) expects 3 arguments".to_string());
            }
            let cond = super::booleans::parse_bool_expr(
                parts[1], None, v_real, None, v_arr, None, resolver,
            )?;
            let then = parse_real_expr(parts[2], v_real, v_arr, resolver)?;
            let orelse = parse_real_expr(parts[3], v_real, v_arr, resolver)?;
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
            res.as_real()
                .ok_or_else(|| "select did not return a real".to_string())
        }
        _ => Err(format!("Unknown real arithmetic operator: {}", parts[0])),
    }
}
