use super::{get_len, get_val, translate_type, CodegenContext};
use crate::ssa::ir::{InstructionKind, Type as SsaType};
use cranelift::prelude::*;
use cranelift_module::Module;

pub fn lower<M: Module>(ctx: &mut CodegenContext<M>, kind: &InstructionKind) -> Result<(), String> {
    match kind {
        InstructionKind::BufferLoad(dest, buf, idx) => {
            let buf_ptr = get_val(&ctx.values, buf);
            let idx_val = get_val(&ctx.values, idx);
            let dest_ty = ctx.ssa_func.get_type(*dest);
            let cl_ty = translate_type(&dest_ty);
            let elem_size = match &ctx.ssa_func.get_type(*buf) {
                SsaType::Buffer(inner) => inner.size(&ctx.ssa_func.struct_layouts),
                _ => 8,
            };
            let offset = ctx.builder.ins().imul_imm(idx_val, elem_size as i64);
            let addr = ctx.builder.ins().iadd(buf_ptr, offset);
            let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), addr, 0);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::BufferStore(dest, buf, idx, val, ty) => {
            let buf_ptr = get_val(&ctx.values, buf);
            let idx_val = get_val(&ctx.values, idx);
            let val_val = get_val(&ctx.values, val);
            let elem_size = match &ctx.ssa_func.get_type(*buf) {
                SsaType::Buffer(inner) => inner.size(&ctx.ssa_func.struct_layouts),
                _ => 8,
            };
            let offset = ctx.builder.ins().imul_imm(idx_val, elem_size as i64);
            let addr = ctx.builder.ins().iadd(buf_ptr, offset);

            let cl_ty = translate_type(ty);
            let val_to_store = if ctx.builder.func.dfg.value_type(val_val) != cl_ty {
                if cl_ty.is_int() && ctx.builder.func.dfg.value_type(val_val).is_int() {
                    ctx.builder.ins().ireduce(cl_ty, val_val)
                } else {
                    val_val
                }
            } else {
                val_val
            };

            ctx.builder
                .ins()
                .store(MemFlags::new(), val_to_store, addr, 0);
            ctx.values.insert(*dest, buf_ptr);

            if let Some(len) = ctx.buffer_lengths.get(buf) {
                let len_val = *len;
                ctx.buffer_lengths.insert(*dest, len_val);
            }
        }
        InstructionKind::BufferLen(dest, buf) => {
            let len = get_len(&ctx.buffer_lengths, buf);
            ctx.values.insert(*dest, len);
        }
        InstructionKind::ArrayLoad(dest, arr, idx) => {
            let arr_ptr = get_val(&ctx.values, arr);
            let idx_val = get_val(&ctx.values, idx);
            let dest_ty = ctx.ssa_func.get_type(*dest);
            let cl_ty = translate_type(&dest_ty);
            let elem_size = match &ctx.ssa_func.get_type(*arr) {
                SsaType::Array(inner, _) => inner.size(&ctx.ssa_func.struct_layouts),
                _ => 8,
            };
            let offset = ctx.builder.ins().imul_imm(idx_val, elem_size as i64);
            let addr = ctx.builder.ins().iadd(arr_ptr, offset);
            let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), addr, 0);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::ArrayStore(dest, arr, idx, val, ty) => {
            let arr_ptr = get_val(&ctx.values, arr);
            let idx_val = get_val(&ctx.values, idx);
            let val_val = get_val(&ctx.values, val);
            let elem_size = match &ctx.ssa_func.get_type(*arr) {
                SsaType::Array(inner, _) => inner.size(&ctx.ssa_func.struct_layouts),
                _ => 8,
            };
            let offset = ctx.builder.ins().imul_imm(idx_val, elem_size as i64);
            let addr = ctx.builder.ins().iadd(arr_ptr, offset);

            let cl_ty = translate_type(ty);
            let val_to_store = if ctx.builder.func.dfg.value_type(val_val) != cl_ty {
                if cl_ty.is_int() && ctx.builder.func.dfg.value_type(val_val).is_int() {
                    ctx.builder.ins().ireduce(cl_ty, val_val)
                } else {
                    val_val
                }
            } else {
                val_val
            };

            ctx.builder
                .ins()
                .store(MemFlags::new(), val_to_store, addr, 0);
            ctx.values.insert(*dest, arr_ptr);
        }
        InstructionKind::StructLoad(dest, obj, offset) => {
            let obj_ptr = get_val(&ctx.values, obj);
            let dest_ty = ctx.ssa_func.get_type(*dest);
            let cl_ty = translate_type(&dest_ty);
            let res = ctx
                .builder
                .ins()
                .load(cl_ty, MemFlags::new(), obj_ptr, *offset as i32);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::StructOffset(dest, obj, offset) => {
            let obj_ptr = get_val(&ctx.values, obj);
            let res = ctx.builder.ins().iadd_imm(obj_ptr, *offset as i64);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::StructSet(dest, obj, offset, val, ty) => {
            let obj_ptr = get_val(&ctx.values, obj);
            let val_val = get_val(&ctx.values, val);
            let cl_ty = translate_type(ty);
            let val_to_store = if ctx.builder.func.dfg.value_type(val_val) != cl_ty {
                if cl_ty.is_int() && ctx.builder.func.dfg.value_type(val_val).is_int() {
                    ctx.builder.ins().ireduce(cl_ty, val_val)
                } else {
                    val_val
                }
            } else {
                val_val
            };
            ctx.builder
                .ins()
                .store(MemFlags::new(), val_to_store, obj_ptr, *offset as i32);
            ctx.values.insert(*dest, obj_ptr);
        }
        InstructionKind::Borrow(dest, src) | InstructionKind::MutBorrow(dest, src) => {
            let s = get_val(&ctx.values, src);
            ctx.values.insert(*dest, s);
        }
        _ => return Err(format!("Not a memory instruction: {:?}", kind)),
    }
    Ok(())
}
