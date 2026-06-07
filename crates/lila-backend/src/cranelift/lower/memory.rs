use super::{get_len, get_val, translate_type, CodegenContext};
use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};
use lila_ir::ir::{InstructionKind, Type as SsaType};

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
        InstructionKind::StructCreate(dest, struct_name, args) => {
            let dest_ty = ctx.ssa_func.get_type(*dest);
            let size = dest_ty.size(&ctx.ssa_func.struct_layouts);

            let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                size as u32,
            ));

            let fields = ctx
                .ssa_func
                .struct_layouts
                .get(struct_name)
                .unwrap()
                .clone();
            let mut field_offset = 0;
            for (i, p_val) in args.iter().enumerate() {
                let cl_p_val = get_val(&ctx.values, p_val);
                let f_ty = &fields[i].1;
                let f_align = f_ty.align(&ctx.ssa_func.struct_layouts);
                field_offset = (field_offset + f_align - 1) & !(f_align - 1);

                if f_ty.is_composite() {
                    let f_size = f_ty.size(&ctx.ssa_func.struct_layouts);
                    super::copy_to_stack(
                        &mut ctx.builder,
                        cl_p_val,
                        slot,
                        field_offset as i32,
                        f_size,
                    );
                } else {
                    let cl_ty = super::translate_type(f_ty);
                    let val_to_store = if ctx.builder.func.dfg.value_type(cl_p_val) != cl_ty {
                        if cl_ty.is_int() && ctx.builder.func.dfg.value_type(cl_p_val).is_int() {
                            ctx.builder.ins().ireduce(cl_ty, cl_p_val)
                        } else {
                            cl_p_val
                        }
                    } else {
                        cl_p_val
                    };

                    ctx.builder
                        .ins()
                        .stack_store(val_to_store, slot, field_offset as i32);
                }
                field_offset += f_ty.size(&ctx.ssa_func.struct_layouts);
            }

            let dest_addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
            ctx.values.insert(*dest, dest_addr);
        }
        InstructionKind::StructLoad(dest, obj, offset) => {
            let obj_ptr = get_val(&ctx.values, obj);
            let dest_ty = ctx.ssa_func.get_type(*dest);
            if dest_ty.is_composite() {
                let res = ctx.builder.ins().iadd_imm(obj_ptr, *offset as i64);
                ctx.values.insert(*dest, res);
            } else {
                let cl_ty = translate_type(&dest_ty);
                let res = ctx
                    .builder
                    .ins()
                    .load(cl_ty, MemFlags::new(), obj_ptr, *offset as i32);
                ctx.values.insert(*dest, res);
            }
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
        InstructionKind::EnumCreate(dest, enum_name, tag_idx, payload) => {
            let mut size = 0;
            if let Some(variants) = ctx.ssa_func.enum_layouts.get(enum_name) {
                let mut max_payload_size = 0;
                let mut max_align = 1;
                for (_, f_ty) in variants {
                    let sz = f_ty.size(&ctx.ssa_func.struct_layouts);
                    if sz > max_payload_size {
                        max_payload_size = sz;
                    }
                    let a = f_ty.align(&ctx.ssa_func.struct_layouts);
                    if a > max_align {
                        max_align = a;
                    }
                }
                let mut offset = 1;
                offset = (offset + max_align - 1) & !(max_align - 1);
                offset += max_payload_size;
                size = (offset + max_align - 1) & !(max_align - 1);
            }

            let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                size as u32,
            ));

            let tag_val = ctx.builder.ins().iconst(types::I8, *tag_idx as i64);
            ctx.builder.ins().stack_store(tag_val, slot, 0);

            if let Some(p) = payload {
                let p_val = get_val(&ctx.values, p);
                let variants = ctx.ssa_func.enum_layouts.get(enum_name).unwrap();
                let payload_ty = &variants[*tag_idx].1;

                let p_align = payload_ty.align(&ctx.ssa_func.struct_layouts);
                let mut offset = 1;
                offset = (offset + p_align - 1) & !(p_align - 1);

                if payload_ty.is_composite() {
                    let p_size = payload_ty.size(&ctx.ssa_func.struct_layouts);
                    super::copy_to_stack(&mut ctx.builder, p_val, slot, offset as i32, p_size);
                } else {
                    ctx.builder.ins().stack_store(p_val, slot, offset as i32);
                }
            }

            let dest_addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
            ctx.values.insert(*dest, dest_addr);
        }
        InstructionKind::EnumIsVariant(dest, obj, tag_idx) => {
            let obj_ptr = get_val(&ctx.values, obj);
            let tag_val = ctx
                .builder
                .ins()
                .load(types::I8, MemFlags::new(), obj_ptr, 0);
            let expected_tag = ctx.builder.ins().iconst(types::I8, *tag_idx as i64);
            let is_match =
                ctx.builder
                    .ins()
                    .icmp(cranelift::prelude::IntCC::Equal, tag_val, expected_tag);

            let dest_ty = ctx.ssa_func.get_type(*dest);
            let cl_ty = translate_type(&dest_ty);
            let res = ctx.builder.ins().bmask(cl_ty, is_match);
            ctx.values.insert(*dest, res);
        }
        InstructionKind::EnumGetTag(dest, obj) => {
            let obj_ptr = get_val(&ctx.values, obj);
            let tag_val = ctx
                .builder
                .ins()
                .load(types::I8, MemFlags::new(), obj_ptr, 0);
            ctx.values.insert(*dest, tag_val);
        }
        InstructionKind::EnumExtract(dest, obj, tag_idx) => {
            let obj_ptr = get_val(&ctx.values, obj);
            let enum_name = match ctx.ssa_func.get_type(*obj) {
                SsaType::Enum(ref name) => name.clone(),
                _ => unreachable!(),
            };
            let variants = ctx.ssa_func.enum_layouts.get(&enum_name).unwrap();
            let payload_ty = &variants[*tag_idx].1;
            let p_align = payload_ty.align(&ctx.ssa_func.struct_layouts);

            let mut offset = 1;
            offset = (offset + p_align - 1) & !(p_align - 1);

            let addr = ctx.builder.ins().iadd_imm(obj_ptr, offset as i64);
            if payload_ty.is_composite() {
                ctx.values.insert(*dest, addr);
            } else {
                let cl_ty = translate_type(payload_ty);
                let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), addr, 0);
                ctx.values.insert(*dest, res);
            }
        }
        InstructionKind::Alloc(dest, ty) => {
            let size = ty.size(&ctx.ssa_func.struct_layouts);
            let mut sig = ctx.module.make_signature();
            sig.params.push(AbiParam::new(types::I64)); // size
            sig.returns.push(AbiParam::new(types::I64)); // ptr
            let callee = ctx
                .module
                .declare_function("malloc", Linkage::Import, &sig)
                .unwrap();
            let local_callee = ctx
                .module
                .declare_func_in_func(callee, ctx.builder.func);
            let size_val = ctx.builder.ins().iconst(types::I64, size as i64);
            let call = ctx.builder.ins().call(local_callee, &[size_val]);
            let res = ctx.builder.inst_results(call)[0];
            ctx.values.insert(*dest, res);
        }
        InstructionKind::PointerLoad(dest, ptr) => {
            let ptr_val = get_val(&ctx.values, ptr);
            let dest_ty = ctx.ssa_func.get_type(*dest);
            if dest_ty.is_composite() {
                // For composite types, we just return the pointer itself?
                // Wait, if it's a Box<Struct>, pload should probably return the address of the struct.
                // Which IS the ptr_val.
                ctx.values.insert(*dest, ptr_val);
            } else {
                let cl_ty = translate_type(&dest_ty);
                let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), ptr_val, 0);
                ctx.values.insert(*dest, res);
            }
        }
        InstructionKind::PointerStore(ptr, val) => {
            let ptr_val = get_val(&ctx.values, ptr);
            let val_val = get_val(&ctx.values, val);
            let val_ty = ctx.ssa_func.get_type(*val);

            if val_ty.is_composite() {
                let size = val_ty.size(&ctx.ssa_func.struct_layouts);
                let mut sig = ctx.module.make_signature();
                sig.params.push(AbiParam::new(types::I64)); // dest
                sig.params.push(AbiParam::new(types::I64)); // src
                sig.params.push(AbiParam::new(types::I64)); // n
                sig.returns.push(AbiParam::new(types::I64)); // dest

                let callee = ctx
                    .module
                    .declare_function("memcpy", Linkage::Import, &sig)
                    .unwrap();
                let local_callee = ctx
                    .module
                    .declare_func_in_func(callee, ctx.builder.func);

                let size_val = ctx.builder.ins().iconst(types::I64, size as i64);
                ctx.builder
                    .ins()
                    .call(local_callee, &[ptr_val, val_val, size_val]);
            } else {
                ctx.builder.ins().store(MemFlags::new(), val_val, ptr_val, 0);
            }
        }
        _ => return Err(format!("Not a memory instruction: {:?}", kind)),
    }
    Ok(())
}
