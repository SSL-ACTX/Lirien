use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind};
use z3::ast::{Bool, Real};
use z3::SatResult;

pub fn translate<'ctx>(
    ctx: &mut TranslationContext<'ctx>,
    inst: &Instruction,
    path_cond: &Bool<'ctx>,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::FSqrt(dest, src) => {
            if let (Some(z3_dest), Some(z3_src)) = (ctx.z3_reals.get(dest), ctx.z3_reals.get(src)) {
                let zero = Real::from_real(ctx.ctx, 0, 1);

                // Domain check: x >= 0
                ctx.solver.push();
                ctx.solver.assert(path_cond);
                ctx.solver.assert(&z3_src.lt(&zero));
                if ctx.solver.check() == SatResult::Sat {
                    return Err(format!("Potential sqrt of negative number at v{}", dest.0));
                }
                ctx.solver.pop(1);

                // Range lila: sqrt(x) >= 0
                ctx.solver.assert(&path_cond.implies(&z3_dest.ge(&zero)));
            }
        }
        InstructionKind::FSin(dest, _) | InstructionKind::FCos(dest, _) => {
            if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                let neg_one = Real::from_real(ctx.ctx, -1, 1);
                let one = Real::from_real(ctx.ctx, 1, 1);
                // Range lila: -1 <= sin/cos(x) <= 1
                ctx.solver.assert(&path_cond.implies(&z3_dest.ge(&neg_one)));
                ctx.solver.assert(&path_cond.implies(&z3_dest.le(&one)));
            }
        }
        InstructionKind::FLog(dest, src)
        | InstructionKind::FLog2(dest, src)
        | InstructionKind::FLog10(dest, src) => {
            if let (Some(_z3_dest), Some(z3_src)) = (ctx.z3_reals.get(dest), ctx.z3_reals.get(src))
            {
                let zero = Real::from_real(ctx.ctx, 0, 1);

                // Domain check: x > 0
                ctx.solver.push();
                ctx.solver.assert(path_cond);
                ctx.solver.assert(&z3_src.le(&zero));
                if ctx.solver.check() == SatResult::Sat {
                    return Err(format!(
                        "Potential non-positive argument to log at v{}",
                        dest.0
                    ));
                }
                ctx.solver.pop(1);
            }
        }
        InstructionKind::FAsin(dest, src) | InstructionKind::FAcos(dest, src) => {
            if let (Some(_z3_dest), Some(z3_src)) = (ctx.z3_reals.get(dest), ctx.z3_reals.get(src))
            {
                let neg_one = Real::from_real(ctx.ctx, -1, 1);
                let one = Real::from_real(ctx.ctx, 1, 1);

                // Domain check: -1 <= x <= 1
                ctx.solver.push();
                ctx.solver.assert(path_cond);
                let out_of_bounds = Bool::or(ctx.ctx, &[&z3_src.lt(&neg_one), &z3_src.gt(&one)]);
                ctx.solver.assert(&out_of_bounds);
                if ctx.solver.check() == SatResult::Sat {
                    return Err(format!(
                        "Potential out-of-domain argument to asin/acos at v{}",
                        dest.0
                    ));
                }
                ctx.solver.pop(1);
            }
        }
        InstructionKind::FAcosh(dest, src) => {
            if let (Some(_z3_dest), Some(z3_src)) = (ctx.z3_reals.get(dest), ctx.z3_reals.get(src))
            {
                let one = Real::from_real(ctx.ctx, 1, 1);

                // Domain check: x >= 1
                ctx.solver.push();
                ctx.solver.assert(path_cond);
                ctx.solver.assert(&z3_src.lt(&one));
                if ctx.solver.check() == SatResult::Sat {
                    return Err(format!("Potential argument < 1 to acosh at v{}", dest.0));
                }
                ctx.solver.pop(1);
            }
        }
        InstructionKind::FAtanh(dest, src) => {
            if let (Some(_z3_dest), Some(z3_src)) = (ctx.z3_reals.get(dest), ctx.z3_reals.get(src))
            {
                let neg_one = Real::from_real(ctx.ctx, -1, 1);
                let one = Real::from_real(ctx.ctx, 1, 1);

                // Domain check: -1 < x < 1
                ctx.solver.push();
                ctx.solver.assert(path_cond);
                let out_of_bounds = Bool::or(ctx.ctx, &[&z3_src.le(&neg_one), &z3_src.ge(&one)]);
                ctx.solver.assert(&out_of_bounds);
                if ctx.solver.check() == SatResult::Sat {
                    return Err(format!(
                        "Potential out-of-domain argument to atanh at v{}",
                        dest.0
                    ));
                }
                ctx.solver.pop(1);
            }
        }
        InstructionKind::FExp(dest, _) | InstructionKind::FExp2(dest, _) => {
            if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                let zero = Real::from_real(ctx.ctx, 0, 1);
                // Range lila: exp(x) > 0
                ctx.solver.assert(&path_cond.implies(&z3_dest.gt(&zero)));
            }
        }
        InstructionKind::FCosh(dest, _) => {
            if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                let one = Real::from_real(ctx.ctx, 1, 1);
                // Range lila: cosh(x) >= 1
                ctx.solver.assert(&path_cond.implies(&z3_dest.ge(&one)));
            }
        }
        InstructionKind::FTanh(dest, _) => {
            if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                let neg_one = Real::from_real(ctx.ctx, -1, 1);
                let one = Real::from_real(ctx.ctx, 1, 1);
                // Range lila: -1 < tanh(x) < 1
                ctx.solver.assert(&path_cond.implies(&z3_dest.gt(&neg_one)));
                ctx.solver.assert(&path_cond.implies(&z3_dest.lt(&one)));
            }
        }
        _ => {}
    }
    Ok(())
}
