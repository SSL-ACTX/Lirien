use z3::ast::{Ast, Bool, Float};
use z3_sys;

pub fn unify_floats(a: &Float, b: &Float) -> (Float, Float) {
    let sort_a = a.get_sort();
    let sort_b = b.get_sort();
    if sort_a == sort_b {
        return (a.clone(), b.clone());
    }

    let ctx = a.get_ctx();
    unsafe {
        let context = ctx.get_z3_context();
        let ebits_a = z3_sys::Z3_fpa_get_ebits(context, sort_a.get_z3_sort());
        let ebits_b = z3_sys::Z3_fpa_get_ebits(context, sort_b.get_z3_sort());

        if ebits_a > ebits_b {
            // Promote b to sort of a
            let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(context)
                .expect("Rounding mode failed");
            let promoted = z3_sys::Z3_mk_fpa_to_fp_float(
                context,
                rm,
                b.get_z3_ast(),
                sort_a.get_z3_sort(),
            );
            (a.clone(), Float::wrap(ctx, promoted.expect("Promotion failed")))
        } else {
            // Promote a to sort of b
            let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(context)
                .expect("Rounding mode failed");
            let promoted = z3_sys::Z3_mk_fpa_to_fp_float(
                context,
                rm,
                a.get_z3_ast(),
                sort_b.get_z3_sort(),
            );
            (Float::wrap(ctx, promoted.expect("Promotion failed")), b.clone())
        }
    }
}

pub fn float_eq(a: &Float, b: &Float) -> Bool {
    let (lhs, rhs) = unify_floats(a, b);
    let ctx = lhs.get_ctx();
    unsafe {
        let ast = z3_sys::Z3_mk_eq(
            ctx.get_z3_context(),
            lhs.get_z3_ast(),
            rhs.get_z3_ast(),
        );
        Bool::wrap(ctx, ast.expect("Z3_mk_eq failed"))
    }
}

pub fn float_lt(a: &Float, b: &Float) -> Bool {
    let (lhs, rhs) = unify_floats(a, b);
    let ctx = lhs.get_ctx();
    unsafe {
        let ast = z3_sys::Z3_mk_fpa_lt(
            ctx.get_z3_context(),
            lhs.get_z3_ast(),
            rhs.get_z3_ast(),
        );
        Bool::wrap(ctx, ast.expect("Z3_mk_fpa_lt failed"))
    }
}

pub fn float_le(a: &Float, b: &Float) -> Bool {
    let (lhs, rhs) = unify_floats(a, b);
    let ctx = lhs.get_ctx();
    unsafe {
        let ast = z3_sys::Z3_mk_fpa_leq(
            ctx.get_z3_context(),
            lhs.get_z3_ast(),
            rhs.get_z3_ast(),
        );
        Bool::wrap(ctx, ast.expect("Z3_mk_fpa_leq failed"))
    }
}

pub fn float_gt(a: &Float, b: &Float) -> Bool {
    let (lhs, rhs) = unify_floats(a, b);
    let ctx = lhs.get_ctx();
    unsafe {
        let ast = z3_sys::Z3_mk_fpa_gt(
            ctx.get_z3_context(),
            lhs.get_z3_ast(),
            rhs.get_z3_ast(),
        );
        Bool::wrap(ctx, ast.expect("Z3_mk_fpa_gt failed"))
    }
}

pub fn float_ge(a: &Float, b: &Float) -> Bool {
    let (lhs, rhs) = unify_floats(a, b);
    let ctx = lhs.get_ctx();
    unsafe {
        let ast = z3_sys::Z3_mk_fpa_geq(
            ctx.get_z3_context(),
            lhs.get_z3_ast(),
            rhs.get_z3_ast(),
        );
        Bool::wrap(ctx, ast.expect("Z3_mk_fpa_geq failed"))
    }
}

pub fn split_sexpr_parts(s: &str) -> Vec<&str> {
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
