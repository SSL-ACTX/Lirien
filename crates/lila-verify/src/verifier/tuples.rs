use super::TranslationContext;
use lila_ir::ir::{Instruction, InstructionKind};
use z3::ast::Bool;

pub fn translate<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &mut TranslationContext<'_, B>,
    inst: &Instruction,
    path_cond: &Bool,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::TupleCreate(dest, elts) => {
            if let Some(z3_dest) = ctx.z3_arrays.get(dest).cloned() {
                let z3_zero_arr = ctx.backend.array_const(
                    &format!("{}_v{}_zero_{}", ctx.func.name, dest.0, ctx.uid),
                    false,
                    64,
                );
                let mut current_state = z3_zero_arr.clone();
                let ty = ctx.func.get_type(*dest);

                if let lila_ir::ir::Type::Tuple(elt_types) = ty {
                    let mut offset = 0;
                    for (i, elt) in elts.iter().enumerate() {
                        let elt_ty = &elt_types[i];
                        let elt_align = elt_ty.align(&ctx.func.struct_layouts);
                        offset = (offset + elt_align - 1) & !(elt_align - 1);

                        if elt_ty.is_composite() {
                            current_state = super::copy_composite(
                                ctx,
                                current_state,
                                *elt,
                                elt_ty,
                                offset as i64,
                            );
                        } else {
                            let z3_offset = ctx.backend.int_from_i64(offset as i64);
                            if let Some(v) = ctx.z3_bvs.get(elt).cloned() {
                                current_state =
                                    ctx.backend.array_store_bv(&current_state, &z3_offset, &v);
                            } else if let Some(v) = ctx.z3_floats.get(elt).cloned() {
                                current_state =
                                    ctx.backend
                                        .array_store_float(&current_state, &z3_offset, &v);
                            } else if let Some(v) = ctx.z3_ints.get(elt).cloned() {
                                current_state =
                                    ctx.backend.array_store_int(&current_state, &z3_offset, &v);
                            } else {
                                continue;
                            };
                        }
                        offset += elt_ty.size(&ctx.func.struct_layouts);
                    }

                    let __tmp = ctx.backend.array_eq(&z3_dest, &current_state);
                    let __tmp2 = ctx.backend.bool_implies(path_cond, &__tmp);
                    ctx.backend.assert(&__tmp2);
                }
            }
        }
        InstructionKind::TupleExtract(dest, src, idx) => {
            let ty = ctx.func.get_type(*src);
            if let lila_ir::ir::Type::Tuple(elt_types) = ty {
                let mut offset = 0;
                for elt_ty in elt_types.iter().take(*idx) {
                    let elt_align = elt_ty.align(&ctx.func.struct_layouts);
                    offset = (offset + elt_align - 1) & !(elt_align - 1);
                    offset += elt_ty.size(&ctx.func.struct_layouts);
                }
                let target_ty = &elt_types[*idx];
                let target_align = target_ty.align(&ctx.func.struct_layouts);
                offset = (offset + target_align - 1) & !(target_align - 1);

                if let Some(z3_src) = ctx.z3_arrays.get(src).cloned() {
                    if target_ty.is_composite() {
                        if let Some(z3_dest) = ctx.z3_arrays.get(dest).cloned() {
                            let current_state = super::copy_composite(
                                ctx,
                                z3_dest.clone(),
                                *src,
                                target_ty,
                                -(offset as i64),
                            );
                            let __tmp = ctx.backend.array_eq(&z3_dest, &current_state);
                            let __tmp2 = ctx.backend.bool_implies(path_cond, &__tmp);
                            ctx.backend.assert(&__tmp2);
                        }
                    } else {
                        let z3_offset = ctx.backend.int_from_i64(offset as i64);

                        if let Some(z3_dest) = ctx.z3_bvs.get(dest).cloned() {
                            let res = ctx.backend.array_select_bv(&z3_src, &z3_offset);
                            let __tmp = ctx.backend.bv_eq(&z3_dest, &res);
                            let __tmp2 = ctx.backend.bool_implies(path_cond, &__tmp);
                            ctx.backend.assert(&__tmp2);
                        } else if let Some(z3_dest) = ctx.z3_floats.get(dest).cloned() {
                            let res = ctx.backend.array_select_float(&z3_src, &z3_offset);
                            let __tmp = ctx.backend.float_eq(&z3_dest, &res);
                            let __tmp2 = ctx.backend.bool_implies(path_cond, &__tmp);
                            ctx.backend.assert(&__tmp2);
                        } else if let Some(z3_dest) = ctx.z3_ints.get(dest).cloned() {
                            let res = ctx.backend.array_select_int(&z3_src, &z3_offset);
                            let __tmp = ctx.backend.int_eq(&z3_dest, &res);
                            let __tmp2 = ctx.backend.bool_implies(path_cond, &__tmp);
                            ctx.backend.assert(&__tmp2);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}
