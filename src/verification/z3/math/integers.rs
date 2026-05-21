use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind};
use z3::ast::{Ast, Bool, Int};
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
        InstructionKind::Shl(dest, _, _)
        | InstructionKind::LShr(dest, _, _)
        | InstructionKind::AShr(dest, _, _)
        | InstructionKind::Xor(dest, _, _) => {
            // Unconstrained bitwise for now
            let _z3_dest = ctx.z3_ints.get(dest);
        }
        _ => {}
    }
    Ok(())
}
