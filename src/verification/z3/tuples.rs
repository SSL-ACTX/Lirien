use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind};
use z3::ast::{Array, Ast, Bool, Int, BV};

pub fn translate(
    ctx: &mut TranslationContext,
    inst: &Instruction,
    path_cond: &Bool,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::TupleCreate(dest, elts) => {
            if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                let int_sort = Int::from_i64(0).get_sort();
                let val_sort = BV::from_i64(0, 64).get_sort();
                let z3_zero_arr = Array::new_const(
                    format!("{}_v{}_zero_{}", ctx.func.name, dest.0, ctx.uid),
                    &int_sort,
                    &val_sort,
                );
                let mut current_state = z3_zero_arr;

                let tuple_ty = ctx.func.get_type(*dest);
                if let crate::ssa::ir::Type::Tuple(elt_types) = tuple_ty {
                    let mut offset = 0;
                    for (i, elt) in elts.iter().enumerate() {
                        let elt_ty = &elt_types[i];
                        let elt_align = elt_ty.align(&ctx.func.struct_layouts);
                        offset = (offset + elt_align - 1) & !(elt_align - 1);

                        if elt_ty.is_composite() {
                            current_state = super::copy_composite_z3(
                                ctx,
                                current_state,
                                *elt,
                                elt_ty,
                                offset as i64,
                            );
                        } else {
                            let z3_offset = Int::from_i64(offset as i64);
                            if let Some(z3_v) = ctx.z3_bvs.get(elt) {
                                current_state = current_state.store(&z3_offset, z3_v);
                            } else if let Some(z3_v) = ctx.z3_floats.get(elt) {
                                current_state = current_state.store(&z3_offset, z3_v);
                            }
                        }
                        offset += elt_ty.size(&ctx.func.struct_layouts);
                    }
                }
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(&current_state)));
            }
        }
        InstructionKind::TupleExtract(dest, tuple_val, idx) => {
            let tuple_ty = ctx.func.get_type(*tuple_val);
            if let crate::ssa::ir::Type::Tuple(elt_types) = tuple_ty {
                let mut offset = 0;
                for elt_ty in elt_types.iter().take(*idx) {
                    let elt_align = elt_ty.align(&ctx.func.struct_layouts);
                    offset = (offset + elt_align - 1) & !(elt_align - 1);
                    offset += elt_ty.size(&ctx.func.struct_layouts);
                }
                let dest_ty = &elt_types[*idx];
                let dest_align = dest_ty.align(&ctx.func.struct_layouts);
                offset = (offset + dest_align - 1) & !(dest_align - 1);

                if dest_ty.is_composite() {
                    if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                        let z3_obj = ctx.z3_arrays.get(tuple_val).unwrap();
                        let mut current_state = z3_dest.clone();
                        let mut leaves = Vec::new();
                        super::get_leaf_offsets(dest_ty, &ctx.func.struct_layouts, 0, &mut leaves);

                        for (l_offset, _) in leaves {
                            let src_offset_z3 = Int::from_i64((offset + l_offset) as i64);
                            let dest_offset_z3 = Int::from_i64(l_offset as i64);
                            let val = z3_obj.select(&src_offset_z3);
                            current_state = current_state.store(&dest_offset_z3, &val);
                        }
                        ctx.solver
                            .assert(path_cond.implies(z3_dest.eq(&current_state)));
                    }
                } else {
                    let z3_obj = ctx.z3_arrays.get(tuple_val).unwrap();
                    let z3_offset = Int::from_i64(offset as i64);
                    if let Some(z3_dest) = ctx.z3_bvs.get(dest) {
                        let select_res = z3_obj.select(&z3_offset);
                        let res = select_res.as_bv().unwrap();
                        ctx.solver.assert(path_cond.implies(z3_dest.eq(&res)));
                    } else if let Some(z3_dest) = ctx.z3_floats.get(dest) {
                        let select_res = z3_obj.select(&z3_offset);
                        let res = select_res.as_float().unwrap();
                        ctx.solver.assert(path_cond.implies(z3_dest.eq(&res)));
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}
