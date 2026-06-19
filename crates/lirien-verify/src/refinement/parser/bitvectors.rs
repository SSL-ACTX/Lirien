use crate::refinement::resolver::Resolver;
use crate::refinement::utils::split_sexpr_parts;
use z3::ast::{Int, BV};

pub(crate) fn parse_bv_expr(
    expr: &str,
    v_bv: Option<&BV>,
    v_int: Option<&Int>,
    resolver: Option<&Resolver>,
) -> Result<BV, String> {
    let expr = expr.trim();
    if expr == "VALUE_PLACEHOLDER" {
        return v_bv
            .cloned()
            .ok_or_else(|| "VALUE_PLACEHOLDER used but no BV available".to_string());
    }
    if let Some(r) = resolver {
        if let Some(bv) = r.resolve_bv(expr) {
            return Ok(bv);
        }
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
            let lhs = parse_bv_expr(parts[1], v_bv, v_int, resolver)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int, resolver)?;
            Ok(lhs.bvand(&rhs))
        }
        "|" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int, resolver)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int, resolver)?;
            Ok(lhs.bvor(&rhs))
        }
        "^" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int, resolver)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int, resolver)?;
            Ok(lhs.bvxor(&rhs))
        }
        "~" => {
            let operand = parse_bv_expr(parts[1], v_bv, v_int, resolver)?;
            Ok(operand.bvnot())
        }
        "<<" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int, resolver)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int, resolver)?;
            Ok(lhs.bvshl(&rhs))
        }
        ">>" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int, resolver)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int, resolver)?;
            Ok(lhs.bvashr(&rhs))
        }
        // Arithmetic ops in BV context
        "+" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int, resolver)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int, resolver)?;
            Ok(lhs.bvadd(&rhs))
        }
        "-" => {
            let lhs = parse_bv_expr(parts[1], v_bv, v_int, resolver)?;
            let rhs = parse_bv_expr(parts[2], v_bv, v_int, resolver)?;
            Ok(lhs.bvsub(&rhs))
        }
        "VALUE_PLACEHOLDER" => v_bv
            .cloned()
            .ok_or_else(|| "No BV value available".to_string()),
        _ => {
            // Fallback: try parsing as Int and convert to BV
            let int_val = super::integers::parse_int_expr(expr, v_int, None, v_bv, resolver)?;
            Ok(BV::from_int(&int_val, 64))
        }
    }
}
