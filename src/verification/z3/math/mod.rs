use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind};
use z3::ast::{Ast, Bool};

pub mod floats;
pub mod integers;
pub mod logic;
pub mod transcendental;

pub fn translate<'ctx>(
    ctx: &mut TranslationContext<'ctx>,
    inst: &Instruction,
    path_cond: &Bool<'ctx>,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::Add(_, _, _)
        | InstructionKind::Sub(_, _, _)
        | InstructionKind::Mul(_, _, _)
        | InstructionKind::SDiv(_, _, _)
        | InstructionKind::UDiv(_, _, _)
        | InstructionKind::SRem(_, _, _)
        | InstructionKind::URem(_, _, _)
        | InstructionKind::ConstInt(_, _)
        | InstructionKind::Shl(_, _, _)
        | InstructionKind::LShr(_, _, _)
        | InstructionKind::AShr(_, _, _)
        | InstructionKind::Xor(_, _, _) => integers::translate(ctx, inst, path_cond),

        InstructionKind::FAdd(_, _, _)
        | InstructionKind::FSub(_, _, _)
        | InstructionKind::FMul(_, _, _)
        | InstructionKind::FDiv(_, _, _)
        | InstructionKind::ConstFloat(_, _)
        | InstructionKind::FPow(_, _, _) => floats::translate(ctx, inst, path_cond),

        InstructionKind::FSqrt(_, _)
        | InstructionKind::FSin(_, _)
        | InstructionKind::FCos(_, _)
        | InstructionKind::FTan(_, _)
        | InstructionKind::FAsin(_, _)
        | InstructionKind::FAcos(_, _)
        | InstructionKind::FAtan(_, _)
        | InstructionKind::FAtan2(_, _, _)
        | InstructionKind::FSinh(_, _)
        | InstructionKind::FCosh(_, _)
        | InstructionKind::FTanh(_, _)
        | InstructionKind::FAsinh(_, _)
        | InstructionKind::FAcosh(_, _)
        | InstructionKind::FAtanh(_, _)
        | InstructionKind::FExp(_, _)
        | InstructionKind::FExp2(_, _)
        | InstructionKind::FLog(_, _)
        | InstructionKind::FLog2(_, _)
        | InstructionKind::FLog10(_, _) => transcendental::translate(ctx, inst, path_cond),

        InstructionKind::Eq(_, _, _)
        | InstructionKind::Ne(_, _, _)
        | InstructionKind::SLt(_, _, _)
        | InstructionKind::SLe(_, _, _)
        | InstructionKind::SGt(_, _, _)
        | InstructionKind::SGe(_, _, _)
        | InstructionKind::ULt(_, _, _)
        | InstructionKind::ULe(_, _, _)
        | InstructionKind::UGt(_, _, _)
        | InstructionKind::UGe(_, _, _)
        | InstructionKind::FLt(_, _, _)
        | InstructionKind::FLe(_, _, _)
        | InstructionKind::FGt(_, _, _)
        | InstructionKind::FGe(_, _, _)
        | InstructionKind::And(_, _, _)
        | InstructionKind::Or(_, _, _)
        | InstructionKind::Not(_, _) => logic::translate(ctx, inst, path_cond),

        InstructionKind::IToF(dest, src, _) => {
            if let (Some(d), Some(s)) = (ctx.z3_reals.get(dest), ctx.z3_ints.get(src)) {
                ctx.solver.assert(&path_cond.implies(&d._eq(&s.to_real())));
            }
            Ok(())
        }
        InstructionKind::FToI(dest, src, _) => {
            if let (Some(d), Some(s)) = (ctx.z3_ints.get(dest), ctx.z3_reals.get(src)) {
                ctx.solver.assert(&path_cond.implies(&d._eq(&s.to_int())));
            }
            Ok(())
        }

        _ => Ok(()),
    }
}
