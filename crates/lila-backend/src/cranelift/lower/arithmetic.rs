use super::{get_val, CodegenContext, LoweringError};
use cranelift::prelude::*;
use cranelift_module::Module;
use lila_ir::ir::{InstructionKind, Type};

pub fn lower<M: Module>(ctx: &mut CodegenContext<M>, kind: &InstructionKind) -> Result<(), LoweringError> {
    macro_rules! bin_op {
        ($dest:expr, $lhs:expr, $rhs:expr, $op:ident) => {{
            let l = get_val(&ctx.values, $lhs);
            let r = get_val(&ctx.values, $rhs);
            let res = ctx.builder.ins().$op(l, r);
            ctx.values.insert(*$dest, res);
        }};
    }

    fn lower_cmp<M: Module>(
        ctx: &mut CodegenContext<M>,
        dest: &lila_ir::ir::Value,
        lhs: &lila_ir::ir::Value,
        rhs: &lila_ir::ir::Value,
        int_cc: IntCC,
        float_cc: FloatCC,
    ) {
        let l = get_val(&ctx.values, lhs);
        let r = get_val(&ctx.values, rhs);
        let l_ty = ctx.builder.func.dfg.value_type(l);
        let res = if l_ty.is_float() {
            ctx.builder.ins().fcmp(float_cc, l, r)
        } else {
            ctx.builder.ins().icmp(int_cc, l, r)
        };
        let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
        let res_final = ctx.builder.ins().bmask(res_ty, res);
        ctx.values.insert(*dest, res_final);
    }

    match kind {
        InstructionKind::Add(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, iadd),
        InstructionKind::Sub(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, isub),
        InstructionKind::Mul(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, imul),
        InstructionKind::SDiv(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, sdiv),
        InstructionKind::UDiv(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, udiv),
        InstructionKind::SRem(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, srem),
        InstructionKind::URem(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, urem),
        InstructionKind::And(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, band),
        InstructionKind::Or(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, bor),
        InstructionKind::Xor(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, bxor),
        InstructionKind::Shl(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, ishl),
        InstructionKind::LShr(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, ushr),
        InstructionKind::AShr(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, sshr),

        InstructionKind::FAdd(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, fadd),
        InstructionKind::FSub(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, fsub),
        InstructionKind::FMul(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, fmul),
        InstructionKind::FDiv(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, fdiv),

        InstructionKind::Abs(dest, src) => {
            let s = get_val(&ctx.values, src);
            let ssa_ty = ctx.ssa_func.get_type(*src);
            let res = if ssa_ty.is_float() {
                ctx.builder.ins().fabs(s)
            } else {
                ctx.builder.ins().iabs(s)
            };
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Neg(dest, src) => {
            let s = get_val(&ctx.values, src);
            let ssa_ty = ctx.ssa_func.get_type(*src);
            let res = if ssa_ty.is_float() {
                ctx.builder.ins().fneg(s)
            } else {
                ctx.builder.ins().ineg(s)
            };
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Min(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let ssa_ty = ctx.ssa_func.get_type(*lhs);
            let res = if ssa_ty.is_float() {
                ctx.builder.ins().fmin(l, r)
            } else {
                if ssa_ty.is_signed() {
                    ctx.builder.ins().smin(l, r)
                } else {
                    ctx.builder.ins().umin(l, r)
                }
            };
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Max(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let ssa_ty = ctx.ssa_func.get_type(*lhs);
            let res = if ssa_ty.is_float() {
                ctx.builder.ins().fmax(l, r)
            } else {
                if ssa_ty.is_signed() {
                    ctx.builder.ins().smax(l, r)
                } else {
                    ctx.builder.ins().umax(l, r)
                }
            };
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Avg(dest, lhs, rhs) => bin_op!(dest, lhs, rhs, avg_round),
        InstructionKind::FSqrt(dest, src) => {
            let s = get_val(&ctx.values, src);
            let res = ctx.builder.ins().sqrt(s);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::SIMDSplat(dest, scalar) => {
            let s = get_val(&ctx.values, scalar);
            let ty = ctx.ssa_func.get_type(*dest);
            let cl_ty = super::translate_type(&ty);
            let lane_ty = cl_ty.lane_type();
            let narrowed_s = if ctx.builder.func.dfg.value_type(s) != lane_ty {
                ctx.builder.ins().ireduce(lane_ty, s)
            } else {
                s
            };
            let res = ctx.builder.ins().splat(cl_ty, narrowed_s);
            ctx.values.insert(*dest, res);
        }

        InstructionKind::SIMDExtractLane(dest, vec, lane) => {
            let v = get_val(&ctx.values, vec);
            let res = ctx.builder.ins().extractlane(v, *lane as u8);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::SIMDInsertLane(dest, vec, val, lane) => {
            let v = get_val(&ctx.values, vec);
            let s = get_val(&ctx.values, val);
            let res = ctx.builder.ins().insertlane(v, s, *lane as u8);
            ctx.values.insert(*dest, res);
        }

        InstructionKind::Not(dest, src) => {
            let s = get_val(&ctx.values, src);
            let res = ctx.builder.ins().bnot(s);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Eq(dest, lhs, rhs) => {
            lower_cmp(ctx, dest, lhs, rhs, IntCC::Equal, FloatCC::Equal)
        }
        InstructionKind::Ne(dest, lhs, rhs) => {
            lower_cmp(ctx, dest, lhs, rhs, IntCC::NotEqual, FloatCC::NotEqual)
        }
        InstructionKind::SLt(dest, lhs, rhs) => {
            lower_cmp(ctx, dest, lhs, rhs, IntCC::SignedLessThan, FloatCC::LessThan)
        }
        InstructionKind::SLe(dest, lhs, rhs) => {
            lower_cmp(
                ctx,
                dest,
                lhs,
                rhs,
                IntCC::SignedLessThanOrEqual,
                FloatCC::LessThanOrEqual,
            )
        }
        InstructionKind::SGt(dest, lhs, rhs) => {
            lower_cmp(
                ctx,
                dest,
                lhs,
                rhs,
                IntCC::SignedGreaterThan,
                FloatCC::GreaterThan,
            )
        }
        InstructionKind::SGe(dest, lhs, rhs) => {
            lower_cmp(
                ctx,
                dest,
                lhs,
                rhs,
                IntCC::SignedGreaterThanOrEqual,
                FloatCC::GreaterThanOrEqual,
            )
        }
        InstructionKind::FLt(dest, lhs, rhs) => {
            lower_cmp(ctx, dest, lhs, rhs, IntCC::Equal, FloatCC::LessThan)
        }
        InstructionKind::FLe(dest, lhs, rhs) => {
            lower_cmp(ctx, dest, lhs, rhs, IntCC::Equal, FloatCC::LessThanOrEqual)
        }
        InstructionKind::FGt(dest, lhs, rhs) => {
            lower_cmp(ctx, dest, lhs, rhs, IntCC::Equal, FloatCC::GreaterThan)
        }
        InstructionKind::FGe(dest, lhs, rhs) => {
            lower_cmp(ctx, dest, lhs, rhs, IntCC::Equal, FloatCC::GreaterThanOrEqual)
        }
        InstructionKind::ULt(dest, lhs, rhs) => {
            lower_cmp(ctx, dest, lhs, rhs, IntCC::UnsignedLessThan, FloatCC::Equal)
        }
        InstructionKind::ULe(dest, lhs, rhs) => {
            lower_cmp(
                ctx,
                dest,
                lhs,
                rhs,
                IntCC::UnsignedLessThanOrEqual,
                FloatCC::Equal,
            )
        }
        InstructionKind::UGt(dest, lhs, rhs) => {
            lower_cmp(
                ctx,
                dest,
                lhs,
                rhs,
                IntCC::UnsignedGreaterThan,
                FloatCC::Equal,
            )
        }
        InstructionKind::UGe(dest, lhs, rhs) => {
            lower_cmp(
                ctx,
                dest,
                lhs,
                rhs,
                IntCC::UnsignedGreaterThanOrEqual,
                FloatCC::Equal,
            )
        }
        InstructionKind::IToF(dest, src, ty) => {
            let s = get_val(&ctx.values, src);
            let target_ty = super::translate_type(ty);
            let res = ctx.builder.ins().fcvt_from_sint(target_ty, s);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::FToI(dest, src, ty) => {
            let s = get_val(&ctx.values, src);
            let target_ty = super::translate_type(ty);
            let res = ctx.builder.ins().fcvt_to_sint(target_ty, s);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::FConv(dest, src, target_ty) => {
            let s = get_val(&ctx.values, src);
            let src_ty = ctx.ssa_func.get_type(*src);
            let target_cl_ty = super::translate_type(target_ty);
            let res = match (src_ty, target_ty) {
                (Type::F32, Type::F64) => ctx.builder.ins().fpromote(target_cl_ty, s),
                (Type::F64, Type::F32) => ctx.builder.ins().fdemote(target_cl_ty, s),
                _ => s, // Same precision or incompatible types (let Cranelift verify)
            };
            ctx.values.insert(*dest, res);
        }
        _ => return Err(LoweringError::InstructionNotSupported(format!("{:?}", kind), None)),
    }
    Ok(())
}
