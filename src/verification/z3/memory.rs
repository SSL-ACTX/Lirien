use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind, Type, Value};
use crate::verification::refinement_parser::{parse_array_refinement, parse_refinement};
use z3::ast::{Array, Ast, Bool, Float, BV};
use z3::SatResult;

pub fn init_values(ctx: &mut TranslationContext) -> Result<(), String> {
    for i in 0..ctx.func.value_count {
        let val = Value(i);
        let ty = ctx.func.get_type(val);

        let mut is_mem_obj = false;
        let mut curr_ty = ty.clone();
        let mut inner_ty = ty.clone();
        loop {
            match curr_ty {
                Type::Array(inner, _) => {
                    is_mem_obj = true;
                    inner_ty = *inner;
                    break;
                }
                Type::Struct(_) => {
                    is_mem_obj = true;
                    // Structs are modeled as Int -> BV for now (field addressed via byte offsets)
                    inner_ty = Type::I64;
                    break;
                }
                Type::Mut(inner) | Type::Ref(inner) | Type::Owned(inner) => {
                    curr_ty = *inner;
                }
                _ => break,
            }
        }

        if is_mem_obj {
            let value_sort = if inner_ty.is_float() {
                if matches!(inner_ty, Type::F32) {
                    Float::from_f32(0.0).get_sort()
                } else {
                    Float::from_f64(0.0).get_sort()
                }
            } else {
                let bit_width = inner_ty.int_bit_width().unwrap_or(64);
                BV::from_i64(0, bit_width).get_sort()
            };
            let int_sort = z3::ast::Int::from_i64(0).get_sort();
            let z3_val = Array::new_const(
                format!("{}_v{}_{}", ctx.func.name, i, ctx.uid),
                &int_sort,
                &value_sort,
            );
            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let ref_expr = parse_array_refinement(refinement, &z3_val, inner_ty.is_float())?;
                ctx.solver.assert(ref_expr);
            }

            ctx.z3_arrays.insert(val, z3_val);
        } else if let Type::Enum(_) = ty {
            // Model Enums as a tag (BV) and a payload (Array)
            let tag_val = BV::new_const(format!("{}_v{}_tag_{}", ctx.func.name, i, ctx.uid), 8);
            ctx.z3_bvs.insert(val, tag_val);

            let int_sort = z3::ast::Int::from_i64(0).get_sort();
            let val_sort = BV::from_i64(0, 64).get_sort();
            let payload_val = Array::new_const(
                format!("{}_v{}_payload_{}", ctx.func.name, i, ctx.uid),
                &int_sort,
                &val_sort,
            );
            ctx.z3_arrays.insert(val, payload_val);
        } else if let Type::Buffer(_) = ty {
            let z3_len = BV::new_const(format!("{}_v{}_len_{}", ctx.func.name, i, ctx.uid), 64);
            let zero = BV::from_i64(0, 64);
            ctx.solver.assert(z3_len.bvsge(&zero));
            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let z3_int = z3_len.to_int(true);
                let ref_expr = parse_refinement(refinement, &z3_int)?;
                ctx.solver.assert(ref_expr);
                ctx.z3_ints.insert(val, z3_int);
            }
            ctx.z3_bvs.insert(val, z3_len);
        } else if ty.is_float() {
            let z3_val = if matches!(ty, Type::F32) {
                Float::new_const_float32(format!("{}_v{}_{}", ctx.func.name, i, ctx.uid))
            } else {
                Float::new_const_double(format!("{}_v{}_{}", ctx.func.name, i, ctx.uid))
            };
            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let ref_expr = crate::verification::refinement_parser::parse_float_refinement(
                    refinement, &z3_val,
                )?;
                ctx.solver.assert(ref_expr);
            }
            ctx.z3_floats.insert(val, z3_val);
        } else {
            let bit_width = ty.int_bit_width().unwrap_or(64);
            let z3_val = BV::new_const(format!("{}_v{}_{}", ctx.func.name, i, ctx.uid), bit_width);

            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let is_signed = !matches!(
                    ty,
                    Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::Bool
                );
                let z3_int = z3_val.to_int(is_signed);
                let ref_expr = parse_refinement(refinement, &z3_int)?;
                ctx.solver.assert(ref_expr);
                ctx.z3_ints.insert(val, z3_int);
            }
            ctx.z3_bvs.insert(val, z3_val);
        }
    }
    Ok(())
}

pub fn translate(
    ctx: &mut TranslationContext,
    inst: &Instruction,
    path_cond: &Bool,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::ArrayLoad(dest, arr, idx) => {
            let z3_arr = ctx
                .z3_arrays
                .get(arr)
                .ok_or_else(|| format!("Array {} not modeled", arr))?;
            let z3_idx = ctx
                .z3_bvs
                .get(idx)
                .ok_or_else(|| format!("Index {} not modeled", idx))?;

            if let Type::Array(_, Some(size)) = ctx.func.get_type(*arr) {
                check_bounds(ctx, path_cond, z3_idx, size as i64, dest.0)?;
            }

            // z3_arr is indexed by Int internally in our modeling (see Sort::int)
            let z3_idx_int = z3_idx.to_int(true);

            if let Some(z3_dest) = ctx.z3_bvs.get(dest) {
                let select_res = z3_arr.select(&z3_idx_int);
                let res = select_res.as_bv().unwrap();
                ctx.solver.assert(path_cond.implies(z3_dest.eq(&res)));
            } else if let Some(z3_dest) = ctx.z3_floats.get(dest) {
                let select_res = z3_arr.select(&z3_idx_int);
                let res = select_res.as_float().unwrap();
                ctx.solver.assert(path_cond.implies(z3_dest.eq(&res)));
            }
        }
        InstructionKind::ArrayStore(dest, arr, idx, val, _ty) => {
            let z3_dest = ctx.z3_arrays.get(dest).unwrap();
            let z3_arr = ctx.z3_arrays.get(arr).unwrap();
            let z3_idx = ctx.z3_bvs.get(idx).unwrap();

            if let Type::Array(_, Some(size)) = ctx.func.get_type(*arr) {
                check_bounds(ctx, path_cond, z3_idx, size as i64, dest.0)?;
            }

            let z3_idx_int = z3_idx.to_int(true);

            if let Some(z3_val) = ctx.z3_bvs.get(val) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_arr.store(&z3_idx_int, z3_val))));
            } else if let Some(z3_val) = ctx.z3_floats.get(val) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_arr.store(&z3_idx_int, z3_val))));
            } else {
                ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_arr)));
            }
        }
        InstructionKind::BufferLoad(dest, buf, idx) => {
            let z3_idx = ctx.z3_bvs.get(idx).unwrap();
            let z3_len = ctx.z3_bvs.get(buf).unwrap();
            check_buffer_bounds(ctx, path_cond, z3_idx, z3_len, dest.0)?;
        }
        InstructionKind::BufferStore(dest, buf, idx, _val, _ty) => {
            let z3_idx = ctx.z3_bvs.get(idx).unwrap();
            let z3_len = ctx.z3_bvs.get(buf).unwrap();
            check_buffer_bounds(ctx, path_cond, z3_idx, z3_len, dest.0)?;
            if let (Some(z3_dest_len), Some(z3_buf_len)) =
                (ctx.z3_bvs.get(dest), ctx.z3_bvs.get(buf))
            {
                ctx.solver
                    .assert(path_cond.implies(z3_dest_len.eq(z3_buf_len)));
            }
        }
        InstructionKind::BufferLen(dest, buf) => {
            let z3_len = ctx.z3_bvs.get(buf).unwrap();
            let z3_dest = ctx.z3_bvs.get(dest).unwrap();
            ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_len)));
        }
        InstructionKind::StructCreate(dest, struct_name, args) => {
            if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                let int_sort = z3::ast::Int::from_i64(0).get_sort();
                let val_sort = BV::from_i64(0, 64).get_sort();
                let z3_zero_arr = Array::new_const(
                    format!("{}_v{}_zero_{}", ctx.func.name, dest.0, ctx.uid),
                    &int_sort,
                    &val_sort,
                );
                let mut current_state = z3_zero_arr;

                let fields = ctx.func.struct_layouts.get(struct_name).unwrap();
                let mut offset = 0;
                for (i, p_val) in args.iter().enumerate() {
                    let f_ty = &fields[i].1;
                    let f_align = f_ty.align(&ctx.func.struct_layouts);
                    offset = (offset + f_align - 1) & !(f_align - 1);

                    let z3_offset = z3::ast::Int::from_i64(offset as i64);
                    if let Some(z3_v) = ctx.z3_bvs.get(p_val) {
                        current_state = current_state.store(&z3_offset, z3_v);
                    } else if let Some(z3_v) = ctx.z3_floats.get(p_val) {
                        current_state = current_state.store(&z3_offset, z3_v);
                    }
                    offset += f_ty.size(&ctx.func.struct_layouts);
                }
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(&current_state)));
            }
        }
        InstructionKind::StructLoad(dest, obj, offset) => {
            let z3_obj = ctx.z3_arrays.get(obj).unwrap();
            let z3_offset = z3::ast::Int::from_i64(*offset as i64);
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
        InstructionKind::StructOffset(dest, obj, _offset) => {
            if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                let z3_obj = ctx.z3_arrays.get(obj).unwrap();
                ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_obj)));
            }
        }
        InstructionKind::StructSet(dest, obj, offset, val, _ty) => {
            let z3_dest = ctx.z3_arrays.get(dest).unwrap();
            let z3_obj = ctx.z3_arrays.get(obj).unwrap();
            let z3_offset = z3::ast::Int::from_i64(*offset as i64);
            if let Some(z3_val) = ctx.z3_bvs.get(val) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_obj.store(&z3_offset, z3_val))));
            } else if let Some(z3_val) = ctx.z3_floats.get(val) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_obj.store(&z3_offset, z3_val))));
            }
        }
        InstructionKind::EnumCreate(dest, _enum_name, tag_idx, payload) => {
            if let Some(z3_dest_tag) = ctx.z3_bvs.get(dest) {
                let z3_tag_val = BV::from_i64(*tag_idx as i64, 8);
                ctx.solver
                    .assert(path_cond.implies(z3_dest_tag.eq(&z3_tag_val)));
            }
            if let Some(z3_dest_payload) = ctx.z3_arrays.get(dest) {
                let int_sort = z3::ast::Int::from_i64(0).get_sort();
                let val_sort = BV::from_i64(0, 64).get_sort();
                let z3_zero_arr = Array::new_const(
                    format!("{}_v{}_zero_{}", ctx.func.name, dest.0, ctx.uid),
                    &int_sort,
                    &val_sort,
                );
                let mut current_state = z3_zero_arr.clone();

                if let Some(payload_val) = payload {
                    if let Some(z3_src_payload) = ctx.z3_arrays.get(payload_val) {
                        current_state = z3_src_payload.clone();
                    }
                }
                ctx.solver
                    .assert(path_cond.implies(z3_dest_payload.eq(&current_state)));
            }
        }
        InstructionKind::EnumIsVariant(dest, obj, tag_idx) => {
            if let Some(z3_dest) = ctx.z3_bvs.get(dest) {
                let z3_obj_tag = ctx.z3_bvs.get(obj).unwrap();
                let expected_tag = BV::from_i64(*tag_idx as i64, 8);

                let is_match = z3_obj_tag.eq(&expected_tag);
                let dest_size = z3_dest.get_size();
                let one = BV::from_i64(1, dest_size);
                let zero = BV::from_i64(0, dest_size);
                let result_val = is_match.ite(&one, &zero);

                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(&result_val)));
            }
        }
        InstructionKind::EnumExtract(dest, obj, _tag_idx) => {
            let z3_obj_payload = ctx.z3_arrays.get(obj).unwrap();
            if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                ctx.solver
                    .assert(path_cond.implies(z3_dest.eq(z3_obj_payload)));
            }
        }
        InstructionKind::Reference(dest, src) | InstructionKind::MutReference(dest, src) => {
            if let (Some(z3_dest), Some(z3_src)) = (ctx.z3_bvs.get(dest), ctx.z3_bvs.get(src)) {
                ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
            }
            if let (Some(z3_dest), Some(z3_src)) = (ctx.z3_floats.get(dest), ctx.z3_floats.get(src))
            {
                ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
            }
            if let (Some(z3_dest), Some(z3_src)) = (ctx.z3_arrays.get(dest), ctx.z3_arrays.get(src))
            {
                ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
            }
        }
        _ => {}
    }
    Ok(())
}

fn check_bounds(
    ctx: &TranslationContext,
    path_cond: &Bool,
    idx: &BV,
    size: i64,
    dest_id: usize,
) -> Result<(), String> {
    let bit_width = idx.get_size();
    let zero = BV::from_i64(0, bit_width);
    let sz = BV::from_i64(size, bit_width);

    ctx.solver.push();
    ctx.solver.assert(path_cond);
    ctx.solver.assert(idx.bvslt(&zero));
    if ctx.solver.check() != SatResult::Unsat {
        return Err(format!(
            "Potential out-of-bounds access (index < 0) at v{}",
            dest_id
        ));
    }
    ctx.solver.pop(1);

    ctx.solver.push();
    ctx.solver.assert(path_cond);
    ctx.solver.assert(idx.bvsge(&sz));
    if ctx.solver.check() != SatResult::Unsat {
        return Err(format!(
            "Potential out-of-bounds access (index >= {}) at v{}",
            size, dest_id
        ));
    }
    ctx.solver.pop(1);
    Ok(())
}

fn check_buffer_bounds(
    ctx: &TranslationContext,
    path_cond: &Bool,
    idx: &BV,
    len: &BV,
    dest_id: usize,
) -> Result<(), String> {
    let bit_width = idx.get_size();
    let zero = BV::from_i64(0, bit_width);

    ctx.solver.push();
    ctx.solver.assert(path_cond);
    ctx.solver.assert(idx.bvslt(&zero));
    if ctx.solver.check() != SatResult::Unsat {
        return Err(format!(
            "Potential out-of-bounds buffer access (index < 0) at v{}",
            dest_id
        ));
    }
    ctx.solver.pop(1);

    ctx.solver.push();
    ctx.solver.assert(path_cond);

    // Handle potential bit width mismatch between index and length
    let len_sz = len.get_size();
    let mut check_len = len.clone();
    if len_sz < bit_width {
        check_len = len.sign_ext(bit_width - len_sz);
    } else if len_sz > bit_width {
        check_len = len.extract(bit_width - 1, 0);
    }

    ctx.solver.assert(idx.bvsge(&check_len));
    if ctx.solver.check() != SatResult::Unsat {
        return Err(format!(
            "Potential out-of-bounds buffer access (index >= len) at v{}",
            dest_id
        ));
    }
    ctx.solver.pop(1);
    Ok(())
}
