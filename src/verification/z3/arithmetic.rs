use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind};
use z3::ast::{Ast, Bool, Int, Real};
use z3::SatResult;

pub fn translate<'ctx>(
    ctx: &mut TranslationContext<'ctx>,
    inst: &Instruction,
    path_cond: &Bool<'ctx>,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::ConstInt(dest, val) => {
            if let Some(z3_dest) = ctx.z3_ints.get(dest) {
                let z3_val = Int::from_i64(ctx.ctx, *val);
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&z3_val)));
            }
        }
        InstructionKind::ConstFloat(dest, val) => {
            if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                let (numer, denom) = f64_to_rational(*val);
                let z3_val = Real::from_real(ctx.ctx, numer, denom);
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&z3_val)));
            }
        }
        InstructionKind::Add(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_ints.get(lhs),
                ctx.z3_ints.get(rhs),
            ) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&Int::add(ctx.ctx, &[z3_l, z3_r]))));
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
        InstructionKind::Sub(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_ints.get(lhs),
                ctx.z3_ints.get(rhs),
            ) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&Int::sub(ctx.ctx, &[z3_l, z3_r]))));
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
        InstructionKind::Mul(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_ints.get(lhs),
                ctx.z3_ints.get(rhs),
            ) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&Int::mul(ctx.ctx, &[z3_l, z3_r]))));
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
        InstructionKind::SDiv(dest, lhs, rhs) | InstructionKind::SRem(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_ints.get(lhs),
                ctx.z3_ints.get(rhs),
            ) {
                let zero = Int::from_i64(ctx.ctx, 0);
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
                        "Potential division by zero at v{}{}",
                        dest.0, loc_info
                    ));
                }
                ctx.solver.pop(1);

                if let InstructionKind::SDiv(_, _, _) = &inst.kind {
                    ctx.solver
                        .assert(&path_cond.implies(&z3_dest._eq(&z3_l.div(z3_r))));
                } else {
                    ctx.solver
                        .assert(&path_cond.implies(&z3_dest._eq(&z3_l.rem(z3_r))));
                }
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
        InstructionKind::UDiv(dest, lhs, rhs) | InstructionKind::URem(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_ints.get(lhs),
                ctx.z3_ints.get(rhs),
            ) {
                if let InstructionKind::UDiv(_, _, _) = &inst.kind {
                    ctx.solver
                        .assert(&path_cond.implies(&z3_dest._eq(&z3_l.div(z3_r))));
                } else {
                    ctx.solver
                        .assert(&path_cond.implies(&z3_dest._eq(&z3_l.rem(z3_r))));
                }
            }
        }
        InstructionKind::Eq(dest, lhs, rhs)
        | InstructionKind::Ne(dest, lhs, rhs)
        | InstructionKind::SLt(dest, lhs, rhs)
        | InstructionKind::SLe(dest, lhs, rhs)
        | InstructionKind::SGt(dest, lhs, rhs)
        | InstructionKind::SGe(dest, lhs, rhs)
        | InstructionKind::ULt(dest, lhs, rhs)
        | InstructionKind::ULe(dest, lhs, rhs)
        | InstructionKind::UGt(dest, lhs, rhs)
        | InstructionKind::UGe(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(l), Some(r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_ints.get(lhs),
                ctx.z3_ints.get(rhs),
            ) {
                let is_true = match &inst.kind {
                    InstructionKind::Eq(_, _, _) => l._eq(r),
                    InstructionKind::Ne(_, _, _) => l._eq(r).not(),
                    InstructionKind::SLt(_, _, _) | InstructionKind::ULt(_, _, _) => l.lt(r),
                    InstructionKind::SLe(_, _, _) | InstructionKind::ULe(_, _, _) => l.le(r),
                    InstructionKind::SGt(_, _, _) | InstructionKind::UGt(_, _, _) => l.gt(r),
                    InstructionKind::SGe(_, _, _) | InstructionKind::UGe(_, _, _) => l.ge(r),
                    _ => unreachable!(),
                };
                let val = Bool::ite(
                    &is_true,
                    &Int::from_i64(ctx.ctx, 1),
                    &Int::from_i64(ctx.ctx, 0),
                );
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&val)));
            } else if let (Some(z3_dest), Some(l), Some(r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_reals.get(lhs),
                ctx.z3_reals.get(rhs),
            ) {
                let is_true = match &inst.kind {
                    InstructionKind::Eq(_, _, _) => l._eq(r),
                    InstructionKind::Ne(_, _, _) => l._eq(r).not(),
                    InstructionKind::SLt(_, _, _) | InstructionKind::FLt(_, _, _) => l.lt(r),
                    InstructionKind::SLe(_, _, _) | InstructionKind::FLe(_, _, _) => l.le(r),
                    InstructionKind::SGt(_, _, _) | InstructionKind::FGt(_, _, _) => l.gt(r),
                    InstructionKind::SGe(_, _, _) | InstructionKind::FGe(_, _, _) => l.ge(r),
                    _ => unreachable!(),
                };
                let val = Bool::ite(
                    &is_true,
                    &Int::from_i64(ctx.ctx, 1),
                    &Int::from_i64(ctx.ctx, 0),
                );
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&val)));
            }
        }
        InstructionKind::FLt(dest, lhs, rhs)
        | InstructionKind::FLe(dest, lhs, rhs)
        | InstructionKind::FGt(dest, lhs, rhs)
        | InstructionKind::FGe(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(l), Some(r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_reals.get(lhs),
                ctx.z3_reals.get(rhs),
            ) {
                let is_true = match &inst.kind {
                    InstructionKind::FLt(_, _, _) => l.lt(r),
                    InstructionKind::FLe(_, _, _) => l.le(r),
                    InstructionKind::FGt(_, _, _) => l.gt(r),
                    InstructionKind::FGe(_, _, _) => l.ge(r),
                    _ => unreachable!(),
                };
                let val = Bool::ite(
                    &is_true,
                    &Int::from_i64(ctx.ctx, 1),
                    &Int::from_i64(ctx.ctx, 0),
                );
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&val)));
            }
        }
        InstructionKind::And(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_ints.get(lhs),
                ctx.z3_ints.get(rhs),
            ) {
                let zero = Int::from_i64(ctx.ctx, 0);
                let both_nonzero =
                    Bool::and(ctx.ctx, &[&z3_l._eq(&zero).not(), &z3_r._eq(&zero).not()]);
                let val = Bool::ite(
                    &both_nonzero,
                    &Int::from_i64(ctx.ctx, 1),
                    &Int::from_i64(ctx.ctx, 0),
                );
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&val)));
            }
        }
        InstructionKind::Or(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_ints.get(dest),
                ctx.z3_ints.get(lhs),
                ctx.z3_ints.get(rhs),
            ) {
                let zero = Int::from_i64(ctx.ctx, 0);
                let either_nonzero =
                    Bool::or(ctx.ctx, &[&z3_l._eq(&zero).not(), &z3_r._eq(&zero).not()]);
                let val = Bool::ite(
                    &either_nonzero,
                    &Int::from_i64(ctx.ctx, 1),
                    &Int::from_i64(ctx.ctx, 0),
                );
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&val)));
            }
        }
        InstructionKind::Not(dest, src) => {
            if let (Some(z3_dest), Some(z3_src)) = (ctx.z3_ints.get(dest), ctx.z3_ints.get(src)) {
                let zero = Int::from_i64(ctx.ctx, 0);
                let is_zero = z3_src._eq(&zero);
                let val = Bool::ite(
                    &is_zero,
                    &Int::from_i64(ctx.ctx, 1),
                    &Int::from_i64(ctx.ctx, 0),
                );
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&val)));
            }
        }
        InstructionKind::FSqrt(dest, _src)
        | InstructionKind::FSin(dest, _src)
        | InstructionKind::FCos(dest, _src) => {
            if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                match &inst.kind {
                    InstructionKind::FSin(_, _) | InstructionKind::FCos(_, _) => {
                        let neg_one = Real::from_real(ctx.ctx, -1, 1);
                        let one = Real::from_real(ctx.ctx, 1, 1);
                        ctx.solver.assert(&path_cond.implies(&z3_dest.ge(&neg_one)));
                        ctx.solver.assert(&path_cond.implies(&z3_dest.le(&one)));
                    }
                    InstructionKind::FSqrt(_, s_val) => {
                        if let Some(z3_src) = ctx.z3_reals.get(s_val) {
                            let zero = Real::from_real(ctx.ctx, 0, 1);

                            ctx.solver.push();
                            ctx.solver.assert(path_cond);
                            ctx.solver.assert(&z3_src.lt(&zero));
                            if ctx.solver.check() == SatResult::Sat {
                                return Err(format!(
                                    "Potential sqrt of negative number at v{}",
                                    dest.0
                                ));
                            }
                            ctx.solver.pop(1);
                        }
                    }
                    _ => unreachable!(),
                }
            }
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

        InstructionKind::IToF(dest, src, _) => {
            if let (Some(d), Some(s)) = (ctx.z3_reals.get(dest), ctx.z3_ints.get(src)) {
                ctx.solver.assert(&path_cond.implies(&d._eq(&s.to_real())));
            }
        }
        InstructionKind::FToI(dest, src, _) => {
            if let (Some(d), Some(s)) = (ctx.z3_ints.get(dest), ctx.z3_reals.get(src)) {
                ctx.solver.assert(&path_cond.implies(&d._eq(&s.to_int())));
            }
        }

        InstructionKind::Shl(dest, _, _)
        | InstructionKind::LShr(dest, _, _)
        | InstructionKind::AShr(dest, _, _)
        | InstructionKind::Xor(dest, _, _) => {
            let _z3_dest = ctx.z3_ints.get(dest);
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
