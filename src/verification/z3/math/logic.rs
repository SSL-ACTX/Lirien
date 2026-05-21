use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind};
use z3::ast::{Ast, Bool, Int};

pub fn translate<'ctx>(
    ctx: &mut TranslationContext<'ctx>,
    inst: &Instruction,
    path_cond: &Bool<'ctx>,
) -> Result<(), String> {
    match &inst.kind {
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
        _ => {}
    }
    Ok(())
}
