use super::{get_val, CodegenContext};
use cranelift::prelude::*;
use cranelift_module::Module;
use lila_ir::ir::{InstructionKind, Type};

pub fn lower<M: Module>(ctx: &mut CodegenContext<M>, kind: &InstructionKind) -> Result<(), String> {
    match kind {
        InstructionKind::Add(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().iadd(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Sub(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().isub(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Mul(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().imul(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::SDiv(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().sdiv(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::UDiv(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().udiv(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::SRem(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().srem(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::URem(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().urem(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::FAdd(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().fadd(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::FSub(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().fsub(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::FMul(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().fmul(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::FDiv(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().fdiv(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::FSqrt(dest, src) => {
            let s = get_val(&ctx.values, src);
            let res = ctx.builder.ins().sqrt(s);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::SIMDSplat(dest, scalar) => {
            let s = get_val(&ctx.values, scalar);
            let ty = ctx.ssa_func.get_type(*dest);
            let cl_ty = super::translate_type(&ty);
            let res = ctx.builder.ins().splat(cl_ty, s);
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

        InstructionKind::And(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().band(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Or(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().bor(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Xor(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().bxor(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Shl(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().ishl(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::LShr(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().ushr(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::AShr(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().sshr(l, r);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Not(dest, src) => {
            let s = get_val(&ctx.values, src);
            let res = ctx.builder.ins().bnot(s);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::Eq(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let l_ty = ctx.builder.func.dfg.value_type(l);
            let res = if l_ty.is_float() {
                ctx.builder.ins().fcmp(FloatCC::Equal, l, r)
            } else {
                ctx.builder.ins().icmp(IntCC::Equal, l, r)
            };
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::Ne(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let l_ty = ctx.builder.func.dfg.value_type(l);
            let res = if l_ty.is_float() {
                ctx.builder.ins().fcmp(FloatCC::NotEqual, l, r)
            } else {
                ctx.builder.ins().icmp(IntCC::NotEqual, l, r)
            };
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::SLt(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().icmp(IntCC::SignedLessThan, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::SLe(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::SGt(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().icmp(IntCC::SignedGreaterThan, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::SGe(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx
                .builder
                .ins()
                .icmp(IntCC::SignedGreaterThanOrEqual, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::FLt(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().fcmp(FloatCC::LessThan, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::FLe(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().fcmp(FloatCC::LessThanOrEqual, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::FGt(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().fcmp(FloatCC::GreaterThan, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::FGe(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::ULt(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().icmp(IntCC::UnsignedLessThan, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::ULe(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::UGt(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx.builder.ins().icmp(IntCC::UnsignedGreaterThan, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
        }
        InstructionKind::UGe(dest, lhs, rhs) => {
            let l = get_val(&ctx.values, lhs);
            let r = get_val(&ctx.values, rhs);
            let res = ctx
                .builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, l, r);
            let res_ty = super::translate_type(&ctx.ssa_func.get_type(*dest));
            let res_final = ctx.builder.ins().bmask(res_ty, res);
            ctx.values.insert(*dest, res_final);
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
        _ => return Err(format!("Not an arithmetic instruction: {:?}", kind)),
    }
    Ok(())
}
