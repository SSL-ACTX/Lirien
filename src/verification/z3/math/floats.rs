use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind};
use z3::ast::{Ast, Bool, Real};
use z3::SatResult;

pub fn translate<'ctx>(
    ctx: &mut TranslationContext<'ctx>,
    inst: &Instruction,
    path_cond: &Bool<'ctx>,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::ConstFloat(dest, val) => {
            if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                let (numer, denom) = f64_to_rational(*val);
                let z3_val = Real::from_real(ctx.ctx, numer, denom);
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&z3_val)));
            }
        }
        InstructionKind::FAdd(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_reals.get(dest),
                ctx.z3_reals.get(lhs),
                ctx.z3_reals.get(rhs),
            ) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&Real::add(ctx.ctx, &[z3_l, z3_r]))));
            }
        }
        InstructionKind::FSub(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_reals.get(dest),
                ctx.z3_reals.get(lhs),
                ctx.z3_reals.get(rhs),
            ) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&Real::sub(ctx.ctx, &[z3_l, z3_r]))));
            }
        }
        InstructionKind::FMul(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_reals.get(dest),
                ctx.z3_reals.get(lhs),
                ctx.z3_reals.get(rhs),
            ) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&Real::mul(ctx.ctx, &[z3_l, z3_r]))));
            }
        }
        InstructionKind::FDiv(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_reals.get(dest),
                ctx.z3_reals.get(lhs),
                ctx.z3_reals.get(rhs),
            ) {
                let zero = Real::from_real(ctx.ctx, 0, 1);
                let is_zero = z3_r._eq(&zero);

                ctx.solver.push();
                ctx.solver.assert(path_cond);
                ctx.solver.assert(&is_zero);
                if ctx.solver.check() == SatResult::Sat {
                    let loc_info = inst
                        .location
                        .map(|l| format!(" at {}", l))
                        .unwrap_or_default();
                    return Err(format!(
                        "Potential float division by zero at v{}{}",
                        dest.0, loc_info
                    ));
                }
                ctx.solver.pop(1);

                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&z3_l.div(z3_r))));
            }
        }
        InstructionKind::FSqrt(dest, _src)
        | InstructionKind::FSin(dest, _src)
        | InstructionKind::FCos(dest, _src) => {
            // These are now handled in transcendental.rs but we keep dispatch logic consistent.
            // If they reach here, they should have been handled by transcendental::translate.
            // This is just to satisfy the compiler about unused variables if we were to handle them here.
            let _z3_dest = ctx.z3_reals.get(dest);
        }
        InstructionKind::FPow(dest, lhs, rhs) => {
            if let (Some(_z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_reals.get(dest),
                ctx.z3_reals.get(lhs),
                ctx.z3_reals.get(rhs),
            ) {
                let zero = Real::from_real(ctx.ctx, 0, 1);
                ctx.solver.push();
                ctx.solver.assert(path_cond);
                let is_base_zero = z3_l._eq(&zero);
                let is_exp_nonpositive = z3_r.le(&zero);
                let is_base_negative = z3_l.lt(&zero);
                let domain_err = Bool::or(
                    ctx.ctx,
                    &[
                        &Bool::and(ctx.ctx, &[&is_base_zero, &is_exp_nonpositive]),
                        &is_base_negative,
                    ],
                );
                ctx.solver.assert(&domain_err);
                if ctx.solver.check() == SatResult::Sat {
                    return Err(format!("Potential domain error in fpow at v{}", dest.0));
                }
                ctx.solver.pop(1);
            }
        }
        _ => {}
    }
    Ok(())
}

fn f64_to_rational(val: f64) -> (i32, i32) {
    if val.is_nan() || val.is_infinite() {
        return (0, 1);
    }
    let precision = 1000000;
    let numer = (val * precision as f64).round() as i64;
    if numer > i32::MAX as i64 {
        (
            i32::MAX,
            (precision as i64 * i32::MAX as i64 / numer) as i32,
        )
    } else if numer < i32::MIN as i64 {
        (
            i32::MIN,
            (precision as i64 * i32::MIN as i64 / numer).abs() as i32,
        )
    } else {
        (numer as i32, precision)
    }
}
