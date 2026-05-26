use super::{get_val, CodegenContext};
use crate::ssa::ir::{InstructionKind, Value as SsaValue};
use cranelift::prelude::*;
use cranelift_module::Module;

pub fn lower<M: Module>(
    ctx: &mut CodegenContext<M>,
    inst_kind: &InstructionKind,
) -> Result<(), String> {
    match inst_kind {
        InstructionKind::TupleCreate(dest, elts) => {
            // For now, we'll represent the tuple as a stack slot if it's large,
            // or just a group of values.
            // But since we need to return it, let's use a stack slot.
            let tuple_ty = ctx.ssa_func.get_type(*dest);
            let size = tuple_ty.size(&ctx.ssa_func.struct_layouts);
            let _align = tuple_ty.align(&ctx.ssa_func.struct_layouts) as u32;

            let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                size as u32,
            ));

            let mut offset = 0;
            for elt in elts {
                let elt_val = get_val(&ctx.values, elt);
                let elt_ty = ctx.ssa_func.get_type(*elt);
                let elt_align = elt_ty.align(&ctx.ssa_func.struct_layouts);
                offset = (offset + elt_align - 1) & !(elt_align - 1);

                ctx.builder.ins().stack_store(elt_val, slot, offset as i32);
                offset += elt_ty.size(&ctx.ssa_func.struct_layouts);
            }

            let addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
            ctx.values.insert(*dest, addr);
        }
        InstructionKind::TupleExtract(dest, tuple_val, idx) => {
            let tuple_addr = get_val(&ctx.values, tuple_val);
            let tuple_ty = ctx.ssa_func.get_type(*tuple_val);

            if let crate::ssa::ir::Type::Tuple(elt_types) = tuple_ty {
                let mut offset = 0;
                for i in 0..*idx {
                    let elt_align = elt_types[i].align(&ctx.ssa_func.struct_layouts);
                    offset = (offset + elt_align - 1) & !(elt_align - 1);
                    offset += elt_types[i].size(&ctx.ssa_func.struct_layouts);
                }
                let dest_ty = &elt_types[*idx];
                let dest_align = dest_ty.align(&ctx.ssa_func.struct_layouts);
                offset = (offset + dest_align - 1) & !(dest_align - 1);

                let cl_ty = super::translate_type(dest_ty);
                let res = ctx
                    .builder
                    .ins()
                    .load(cl_ty, MemFlags::new(), tuple_addr, offset as i32);
                ctx.values.insert(*dest, res);
            }
        }
        _ => {}
    }
    Ok(())
}
