use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind, Type, Value};
use crate::verification::refinement_parser::{parse_array_refinement, parse_refinement};
use z3::ast::{Array, Bool, Float, BV};
use z3::{SatResult, Sort};

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
                    Sort::float(8, 24)
                } else {
                    Sort::float(11, 53)
                }
            } else {
                let bit_width = inner_ty.int_bit_width().unwrap_or(64);
                Sort::bitvector(bit_width)
            };
            let z3_val = Array::new_const(format!("v{}", i), &Sort::int(), &value_sort);
            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let ref_expr = parse_array_refinement(refinement, &z3_val, inner_ty.is_float())?;
                ctx.solver.assert(ref_expr);
            }

            ctx.z3_arrays.insert(val, z3_val);
        } else if let Type::Buffer(_) = ty {
            let z3_len = BV::new_const(format!("v{}_len", i), 64);
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
                Float::new_const_float32(format!("v{}", i))
            } else {
                Float::new_const_double(format!("v{}", i))
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
            let z3_val = BV::new_const(format!("v{}", i), bit_width);

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
        InstructionKind::Borrow(dest, src) | InstructionKind::MutBorrow(dest, src) => {
            if let Some(z3_dest) = ctx.z3_bvs.get(dest) {
                if let Some(z3_src) = ctx.z3_bvs.get(src) {
                    ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
                }
            } else if let Some(z3_dest) = ctx.z3_floats.get(dest) {
                if let Some(z3_src) = ctx.z3_floats.get(src) {
                    ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
                }
            } else if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                if let Some(z3_src) = ctx.z3_arrays.get(src) {
                    ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
                }
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

    // We pad or truncate if len is somehow a different bit width, but normally both are 64
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
