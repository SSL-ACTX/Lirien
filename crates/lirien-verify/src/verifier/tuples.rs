use super::TranslationContext;
use lirien_ir::ir::{Instruction, InstructionKind, Type};

pub fn translate<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &mut TranslationContext<B>,
    inst: &Instruction,
    path_cond: &z3::ast::Bool,
) {
    match &inst.kind {
        InstructionKind::TupleCreate(dest, elements) => {
            let z3_dest = ctx.z3_arrays.get(dest).cloned().unwrap();
            let mut current_state = z3_dest;
            let mut offset = 0;

            for elt in elements {
                let elt_ty = ctx.func.get_type(*elt);
                let z3_offset = ctx.backend.int_from_i64(offset as i64);

                if let Some(z3_val) = ctx.z3_bvs.get(elt).cloned() {
                    current_state = ctx.backend.array_store_bv(&current_state, &z3_offset, &z3_val);
                } else if let Some(z3_val) = ctx.z3_floats.get(elt).cloned() {
                    current_state = ctx.backend.array_store_float(&current_state, &z3_offset, &z3_val, matches!(elt_ty, Type::F32));
                } else if let Some(z3_val) = ctx.z3_arrays.get(elt).cloned() {
                    let __inner = ctx.backend.array_eq(&current_state, &z3_val);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                }

                offset += elt_ty.size(&ctx.func.struct_layouts);
            }
        }
        InstructionKind::TupleExtract(dest, src, index) => {
            let z3_src = ctx.z3_arrays.get(src).cloned().unwrap();
            let src_ty = ctx.func.get_type(*src);

            if let Type::Tuple(tys) = src_ty {
                let mut offset = 0;
                for (i, f_ty) in tys.iter().enumerate() {
                    if i == *index {
                        let z3_offset = ctx.backend.int_from_i64(offset as i64);

                        if f_ty.is_composite() {
                            if let Some(z3_dest) = ctx.z3_arrays.get(dest).cloned() {
                                let __inner = ctx.backend.array_eq(&z3_dest, &z3_src);
                                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                                ctx.backend.assert(&__tmp);
                            }
                        } else {
                            if let Some(z3_dest) = ctx.z3_bvs.get(dest).cloned() {
                                let res = ctx.backend.array_select_bv(&z3_src, &z3_offset);
                                let __inner = ctx.backend.bv_eq(&z3_dest, &res);
                                let __tmp2 = ctx.backend.bool_implies(path_cond, &__inner);
                                ctx.backend.assert(&__tmp2);
                            } else if let Some(z3_dest) = ctx.z3_floats.get(dest).cloned() {
                                let res = ctx.backend.array_select_float(&z3_src, &z3_offset, matches!(f_ty, Type::F32));
                                let __inner = ctx.backend.float_eq(&z3_dest, &res);
                                let __tmp2 = ctx.backend.bool_implies(path_cond, &__inner);
                                ctx.backend.assert(&__tmp2);
                            }
                        }
                        break;
                    }
                    offset += f_ty.size(&ctx.func.struct_layouts);
                }
            }
        }
        _ => {}
    }
}
