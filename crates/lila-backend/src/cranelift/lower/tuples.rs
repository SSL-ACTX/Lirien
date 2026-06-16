use super::{get_val, CodegenContext, LoweringError};
use cranelift::prelude::*;
use cranelift_module::Module;
use lila_ir::ir::InstructionKind;

pub fn lower<M: Module>(
    ctx: &mut CodegenContext<M>,
    inst_kind: &InstructionKind,
) -> Result<(), LoweringError> {
    match inst_kind {
        InstructionKind::TupleCreate(dest, elts) => {
            let tuple_ty = ctx.ssa_func.get_type(*dest);
            let size = tuple_ty.size(&ctx.ssa_func.struct_layouts);

            let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                size as u32,
            ));

            let mut offset = 0;
            let mut all_flat_vals = Vec::new();
            for elt in elts {
                let elt_ty = ctx.ssa_func.get_type(*elt);
                let elt_align = elt_ty.align(&ctx.ssa_func.struct_layouts);
                offset = (offset + elt_align - 1) & !(elt_align - 1);

                super::store_to_stack(ctx, *elt, slot, offset as i32);
                all_flat_vals.extend(super::get_all_cl_values(ctx, elt));
                offset += elt_ty.size(&ctx.ssa_func.struct_layouts);
            }

            let addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
            ctx.values.insert(*dest, addr);
            ctx.unpacked_values.insert(*dest, all_flat_vals);
        }
        InstructionKind::TupleExtract(dest, tuple_val, idx) => {
            let tuple_ty = ctx.ssa_func.get_type(*tuple_val);

            // Handle extraction from register-flattened tuples
            if let Some(cl_vals) = ctx.unpacked_values.get(tuple_val) {
                if let lila_ir::ir::Type::Tuple(ref elt_types) = tuple_ty {
                    let mut start_idx = 0;
                    for i in 0..*idx {
                        start_idx += super::get_flattened_types(ctx.ssa_func, &elt_types[i]).len();
                    }
                    
                    let dest_ty = &elt_types[*idx];
                    let count = super::get_flattened_types(ctx.ssa_func, dest_ty).len();
                    let slice = cl_vals[start_idx..start_idx + count].to_vec();
                    
                    if dest_ty.is_composite() {
                        ctx.unpacked_values.insert(*dest, slice);
                    } else {
                        ctx.values.insert(*dest, slice[0]);
                    }
                    return Ok(());
                }
            }

            // Fallback: handle extraction from memory-allocated tuples
            let tuple_addr = get_val(&ctx.values, tuple_val);
            if let lila_ir::ir::Type::Tuple(elt_types) = tuple_ty {
                let mut offset = 0;
                for elt_ty in elt_types.iter().take(*idx) {
                    let elt_align = elt_ty.align(&ctx.ssa_func.struct_layouts);
                    offset = (offset + elt_align - 1) & !(elt_align - 1);
                    offset += elt_ty.size(&ctx.ssa_func.struct_layouts);
                }
                let dest_ty = &elt_types[*idx];
                let dest_align = dest_ty.align(&ctx.ssa_func.struct_layouts);
                offset = (offset + dest_align - 1) & !(dest_align - 1);

                if dest_ty.is_composite() {
                    let res = ctx.builder.ins().iadd_imm(tuple_addr, offset as i64);
                    ctx.values.insert(*dest, res);
                } else {
                    let cl_ty = super::translate_type(dest_ty);
                    let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), tuple_addr, offset as i32);
                    ctx.values.insert(*dest, res);
                }
            }
        }
        _ => {}
    }
    Ok(())
}
