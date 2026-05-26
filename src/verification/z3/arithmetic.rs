use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind, Type};
use z3::ast::{Ast, Bool, Float, RoundingMode, BV};
use z3::SatResult;
use z3_sys::*;

pub fn translate(
    ctx: &mut TranslationContext,
    inst: &Instruction,
    path_cond: &Bool,
) -> Result<(), String> {
    let rm = RoundingMode::round_towards_zero();

    match &inst.kind {
        InstructionKind::ConstInt(dest, val) => {
            if let Some(z3_dest) = ctx.z3_bvs.get(dest) {
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);
                let z3_val = BV::from_i64(*val, bit_width);
                ctx.solver.assert(path_cond.implies(z3_dest.eq(&z3_val)));
            }
        }
        InstructionKind::ConstFloat(dest, val) => {
            if let Some(z3_dest) = ctx.z3_floats.get(dest) {
                let ty = ctx.func.get_type(*dest);
                let z3_val = if matches!(ty, Type::F32) {
                    Float::from_f32(*val as f32)
                } else {
                    Float::from_f64(*val)
                };
                ctx.solver.assert(path_cond.implies(z3_dest.eq(&z3_val)));
            }
        }
        InstructionKind::Add(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_l.bvadd(z3_r))));
            }
        }
        InstructionKind::FAdd(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(rm.add(z3_l, z3_r))));
            }
        }
        InstructionKind::Sub(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_l.bvsub(z3_r))));
            }
        }
        InstructionKind::FSub(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(rm.sub(z3_l, z3_r))));
            }
        }
        InstructionKind::Mul(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_l.bvmul(z3_r))));
            }
        }
        InstructionKind::FMul(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(rm.mul(z3_l, z3_r))));
            }
        }
        InstructionKind::SDiv(dest, lhs, rhs) | InstructionKind::SRem(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let bit_width = ctx.func.get_type(*rhs).int_bit_width().unwrap_or(64);
                let zero = BV::from_i64(0, bit_width);
                let is_zero = z3_r.eq(&zero);

                ctx.solver.push();
                ctx.solver.assert(path_cond);
                ctx.solver.assert(&is_zero);
                if ctx.solver.check() != SatResult::Unsat {
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
                        .assert(path_cond.implies(z3_dest.eq(z3_l.bvsdiv(z3_r))));
                } else {
                    ctx.solver
                        .assert(path_cond.implies(z3_dest.eq(z3_l.bvsrem(z3_r))));
                }
            }
        }
        InstructionKind::FDiv(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let ty = ctx.func.get_type(*rhs);
                let zero = if matches!(ty, Type::F32) {
                    Float::from_f32(0.0)
                } else {
                    Float::from_f64(0.0)
                };

                ctx.solver.push();
                ctx.solver.assert(path_cond);
                ctx.solver.assert(z3_r.eq(&zero));
                if ctx.solver.check() != SatResult::Unsat {
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
                    .assert(path_cond.implies(z3_dest.eq(rm.div(z3_l, z3_r))));
            }
        }
        InstructionKind::UDiv(dest, lhs, rhs) | InstructionKind::URem(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                if let InstructionKind::UDiv(_, _, _) = &inst.kind {
                    ctx.solver
                        .assert(path_cond.implies(z3_dest.eq(z3_l.bvudiv(z3_r))));
                } else {
                    ctx.solver
                        .assert(path_cond.implies(z3_dest.eq(z3_l.bvurem(z3_r))));
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
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let is_true = match &inst.kind {
                    InstructionKind::Eq(_, _, _) => l.eq(r),
                    InstructionKind::Ne(_, _, _) => l.eq(r).not(),
                    InstructionKind::SLt(_, _, _) => l.bvslt(r),
                    InstructionKind::SLe(_, _, _) => l.bvsle(r),
                    InstructionKind::SGt(_, _, _) => l.bvsgt(r),
                    InstructionKind::SGe(_, _, _) => l.bvsge(r),
                    InstructionKind::ULt(_, _, _) => l.bvult(r),
                    InstructionKind::ULe(_, _, _) => l.bvule(r),
                    InstructionKind::UGt(_, _, _) => l.bvugt(r),
                    InstructionKind::UGe(_, _, _) => l.bvuge(r),
                    _ => unreachable!(),
                };
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);
                let one = BV::from_i64(1, bit_width);
                let zero = BV::from_i64(0, bit_width);
                let val = Bool::ite(&is_true, &one, &zero);
                ctx.solver.assert(path_cond.implies(z3_dest.eq(&val)));
            } else if let (Some(z3_dest), Some(l), Some(r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let is_true = match &inst.kind {
                    InstructionKind::Eq(_, _, _) => l.eq(r),
                    InstructionKind::Ne(_, _, _) => l.eq(r).not(),
                    InstructionKind::SLt(_, _, _) | InstructionKind::FLt(_, _, _) => l.lt(r),
                    InstructionKind::SLe(_, _, _) | InstructionKind::FLe(_, _, _) => l.le(r),
                    InstructionKind::SGt(_, _, _) | InstructionKind::FGt(_, _, _) => l.gt(r),
                    InstructionKind::SGe(_, _, _) | InstructionKind::FGe(_, _, _) => l.ge(r),
                    _ => unreachable!(),
                };
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);
                let one = BV::from_i64(1, bit_width);
                let zero = BV::from_i64(0, bit_width);
                let val = Bool::ite(&is_true, &one, &zero);
                ctx.solver.assert(path_cond.implies(z3_dest.eq(&val)));
            }
        }
        InstructionKind::FLt(dest, lhs, rhs)
        | InstructionKind::FLe(dest, lhs, rhs)
        | InstructionKind::FGt(dest, lhs, rhs)
        | InstructionKind::FGe(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(l), Some(r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let is_true = match &inst.kind {
                    InstructionKind::FLt(_, _, _) => l.lt(r),
                    InstructionKind::FLe(_, _, _) => l.le(r),
                    InstructionKind::FGt(_, _, _) => l.gt(r),
                    InstructionKind::FGe(_, _, _) => l.ge(r),
                    _ => unreachable!(),
                };
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);
                let one = BV::from_i64(1, bit_width);
                let zero = BV::from_i64(0, bit_width);
                let val = Bool::ite(&is_true, &one, &zero);
                ctx.solver.assert(path_cond.implies(z3_dest.eq(&val)));
            }
        }
        InstructionKind::And(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let ty = ctx.func.get_type(*dest);
                if matches!(ty, Type::Bool) {
                    let one = BV::from_i64(1, 1);
                    let both_true = Bool::and(&[&z3_l.eq(&one), &z3_r.eq(&one)]);
                    let val = Bool::ite(&both_true, &one, &BV::from_i64(0, 1));
                    ctx.solver.assert(path_cond.implies(z3_dest.eq(&val)));
                } else {
                    ctx.solver
                        .assert(path_cond.implies(z3_dest.eq(z3_l.bvand(z3_r))));
                }
            }
        }
        InstructionKind::Or(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let ty = ctx.func.get_type(*dest);
                if matches!(ty, Type::Bool) {
                    let one = BV::from_i64(1, 1);
                    let either_true = Bool::or(&[&z3_l.eq(&one), &z3_r.eq(&one)]);
                    let val = Bool::ite(&either_true, &one, &BV::from_i64(0, 1));
                    ctx.solver.assert(path_cond.implies(z3_dest.eq(&val)));
                } else {
                    ctx.solver
                        .assert(path_cond.implies(z3_dest.eq(z3_l.bvor(z3_r))));
                }
            }
        }
        InstructionKind::Xor(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_l.bvxor(z3_r))));
            }
        }
        InstructionKind::Not(dest, src) => {
            if let (Some(z3_dest), Some(z3_src)) = (ctx.z3_bvs.get(dest), ctx.z3_bvs.get(src)) {
                let ty = ctx.func.get_type(*dest);
                if matches!(ty, Type::Bool) {
                    let one = BV::from_i64(1, 1);
                    let is_false = z3_src.eq(BV::from_i64(0, 1));
                    let val = Bool::ite(&is_false, &one, &BV::from_i64(0, 1));
                    ctx.solver.assert(path_cond.implies(z3_dest.eq(&val)));
                } else {
                    ctx.solver
                        .assert(path_cond.implies(z3_dest.eq(z3_src.bvnot())));
                }
            }
        }
        InstructionKind::Shl(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_l.bvshl(z3_r))));
            }
        }
        InstructionKind::LShr(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_l.bvlshr(z3_r))));
            }
        }
        InstructionKind::AShr(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_l.bvashr(z3_r))));
            }
        }
        InstructionKind::FSqrt(dest, _src)
        | InstructionKind::FSin(dest, _src)
        | InstructionKind::FCos(dest, _src) => {
            if let Some(_z3_dest) = ctx.z3_floats.get(dest) {
                match &inst.kind {
                    InstructionKind::FSin(_, _) | InstructionKind::FCos(_, _) => {}
                    InstructionKind::FSqrt(_, s_val) => {
                        if let Some(z3_src) = ctx.z3_floats.get(s_val) {
                            let ty = ctx.func.get_type(*s_val);
                            let zero = if matches!(ty, Type::F32) {
                                Float::from_f32(0.0)
                            } else {
                                Float::from_f64(0.0)
                            };

                            ctx.solver.push();
                            ctx.solver.assert(path_cond);
                            ctx.solver.assert(z3_src.lt(&zero));
                            if ctx.solver.check() != SatResult::Unsat {
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
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let ty = ctx.func.get_type(*lhs);
                let zero = if matches!(ty, Type::F32) {
                    Float::from_f32(0.0)
                } else {
                    Float::from_f64(0.0)
                };

                ctx.solver.push();
                ctx.solver.assert(path_cond);

                let is_base_zero = z3_l.eq(&zero);
                let is_exp_nonpositive = z3_r.le(&zero);
                let is_base_negative = z3_l.lt(&zero);

                let domain_err = Bool::or(&[
                    &Bool::and(&[&is_base_zero, &is_exp_nonpositive]),
                    &is_base_negative,
                ]);

                ctx.solver.assert(&domain_err);
                if ctx.solver.check() != SatResult::Unsat {
                    return Err(format!("Potential domain error in fpow at v{}", dest.0));
                }
                ctx.solver.pop(1);
            }
        }

        InstructionKind::IToF(dest, src, _) => {
            if let (Some(d), Some(s)) = (ctx.z3_floats.get(dest), ctx.z3_bvs.get(src)) {
                let is_signed = !matches!(
                    ctx.func.get_type(*src),
                    Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::Bool
                );

                unsafe {
                    let conv = if is_signed {
                        Z3_mk_fpa_to_fp_signed(
                            ctx.ctx.get_z3_context(),
                            rm.get_z3_ast(),
                            s.get_z3_ast(),
                            d.get_sort().get_z3_sort(),
                        )
                    } else {
                        Z3_mk_fpa_to_fp_unsigned(
                            ctx.ctx.get_z3_context(),
                            rm.get_z3_ast(),
                            s.get_z3_ast(),
                            d.get_sort().get_z3_sort(),
                        )
                    };
                    let res = Float::wrap(ctx.ctx, conv.unwrap());
                    ctx.solver.assert(path_cond.implies(d.eq(&res)));
                }
            }
        }
        InstructionKind::FToI(dest, src, _) => {
            if let (Some(d), Some(s)) = (ctx.z3_bvs.get(dest), ctx.z3_floats.get(src)) {
                let is_signed = !matches!(
                    ctx.func.get_type(*dest),
                    Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::Bool
                );
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);

                unsafe {
                    let conv = if is_signed {
                        Z3_mk_fpa_to_sbv(
                            ctx.ctx.get_z3_context(),
                            rm.get_z3_ast(),
                            s.get_z3_ast(),
                            bit_width,
                        )
                    } else {
                        Z3_mk_fpa_to_ubv(
                            ctx.ctx.get_z3_context(),
                            rm.get_z3_ast(),
                            s.get_z3_ast(),
                            bit_width,
                        )
                    };
                    let res = BV::wrap(ctx.ctx, conv.unwrap());
                    ctx.solver.assert(path_cond.implies(d.eq(&res)));
                }
            }
        }

        _ => {}
    }
    Ok(())
}
