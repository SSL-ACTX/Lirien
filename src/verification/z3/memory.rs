use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind, Type, Value};
use crate::verification::refinement_parser::{parse_array_refinement, parse_refinement};
use z3::ast::{Array, Ast, Bool, Int, Real};
use z3::{SatResult, Sort};

pub fn init_values<'ctx>(ctx: &mut TranslationContext<'ctx>) -> Result<(), String> {
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
                    // Structs are modeled as Int -> Int for now (byte addressed)
                    inner_ty = Type::I8;
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
                Sort::real(ctx.ctx)
            } else {
                Sort::int(ctx.ctx)
            };
            let z3_val =
                Array::new_const(ctx.ctx, format!("v{}", i), &Sort::int(ctx.ctx), &value_sort);
            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let ref_expr =
                    parse_array_refinement(ctx.ctx, refinement, &z3_val, inner_ty.is_float())?;
                ctx.solver.assert(&ref_expr);
            }

            ctx.z3_arrays.insert(val, z3_val);
        } else if let Type::Buffer(_) = ty {
            let z3_len = Int::new_const(ctx.ctx, format!("v{}_len", i));
            let zero = Int::from_i64(ctx.ctx, 0);
            ctx.solver.assert(&z3_len.ge(&zero));
            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let ref_expr = parse_refinement(ctx.ctx, refinement, &z3_len)?;
                ctx.solver.assert(&ref_expr);
            }
            ctx.z3_ints.insert(val, z3_len);
        } else if ty.is_float() {
            let z3_val = Real::new_const(ctx.ctx, format!("v{}", i));
            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let ref_expr = crate::verification::refinement_parser::parse_real_refinement(
                    ctx.ctx, refinement, &z3_val,
                )?;
                ctx.solver.assert(&ref_expr);
            }
            ctx.z3_reals.insert(val, z3_val);
        } else {
            let z3_val = Int::new_const(ctx.ctx, format!("v{}", i));
            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let ref_expr = parse_refinement(ctx.ctx, refinement, &z3_val)?;
                ctx.solver.assert(&ref_expr);
            }
            ctx.z3_ints.insert(val, z3_val);
        }
    }
    Ok(())
}

pub fn translate<'ctx>(
    ctx: &mut TranslationContext<'ctx>,
    inst: &Instruction,
    path_cond: &Bool<'ctx>,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::ArrayLoad(dest, arr, idx) => {
            let z3_arr = ctx
                .z3_arrays
                .get(arr)
                .ok_or_else(|| format!("Array {} not modeled", arr))?;
            let z3_idx = ctx
                .z3_ints
                .get(idx)
                .ok_or_else(|| format!("Index {} not modeled", idx))?;

            if let Type::Array(_, Some(size)) = ctx.func.get_type(*arr) {
                check_bounds(ctx, path_cond, z3_idx, size as i64, dest.0)?;
            }

            if let Some(z3_dest) = ctx.z3_ints.get(dest) {
                let select_res = z3_arr.select(z3_idx);
                if let Some(res) = select_res.as_int() {
                    ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&res)));
                }
            } else if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                let select_res = z3_arr.select(z3_idx);
                if let Some(res) = select_res.as_real() {
                    ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&res)));
                }
            }
        }
        InstructionKind::ArrayStore(dest, arr, idx, val, _ty) => {
            let z3_dest = ctx.z3_arrays.get(dest).unwrap();
            let z3_arr = ctx.z3_arrays.get(arr).unwrap();
            let z3_idx = ctx.z3_ints.get(idx).unwrap();

            if let Type::Array(_, Some(size)) = ctx.func.get_type(*arr) {
                check_bounds(ctx, path_cond, z3_idx, size as i64, dest.0)?;
            }

            if let Some(z3_val) = ctx.z3_ints.get(val) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&z3_arr.store(z3_idx, z3_val))));
            } else if let Some(z3_val) = ctx.z3_reals.get(val) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&z3_arr.store(z3_idx, z3_val))));
            } else {
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(z3_arr)));
            }
        }
        InstructionKind::BufferLoad(dest, buf, idx) => {
            let z3_idx = ctx.z3_ints.get(idx).unwrap();
            let z3_len = ctx.z3_ints.get(buf).unwrap();
            check_buffer_bounds(ctx, path_cond, z3_idx, z3_len, dest.0)?;
        }
        InstructionKind::BufferStore(dest, buf, idx, _val, _ty) => {
            let z3_idx = ctx.z3_ints.get(idx).unwrap();
            let z3_len = ctx.z3_ints.get(buf).unwrap();
            check_buffer_bounds(ctx, path_cond, z3_idx, z3_len, dest.0)?;
            if let (Some(z3_dest_len), Some(z3_buf_len)) =
                (ctx.z3_ints.get(dest), ctx.z3_ints.get(buf))
            {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest_len._eq(z3_buf_len)));
            }
        }
        InstructionKind::BufferLen(dest, buf) => {
            let z3_len = ctx.z3_ints.get(buf).unwrap();
            let z3_dest = ctx.z3_ints.get(dest).unwrap();
            ctx.solver.assert(&path_cond.implies(&z3_dest._eq(z3_len)));
        }
        InstructionKind::StructLoad(dest, obj, offset) => {
            let z3_obj = ctx.z3_arrays.get(obj).unwrap();
            let z3_offset = Int::from_i64(ctx.ctx, *offset as i64);
            if let Some(z3_dest) = ctx.z3_ints.get(dest) {
                let select_res = z3_obj.select(&z3_offset);
                if let Some(res) = select_res.as_int() {
                    ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&res)));
                }
            } else if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                let select_res = z3_obj.select(&z3_offset);
                if let Some(res) = select_res.as_real() {
                    ctx.solver.assert(&path_cond.implies(&z3_dest._eq(&res)));
                }
            }
        }
        InstructionKind::StructOffset(dest, obj, _offset) => {
            if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                let z3_obj = ctx.z3_arrays.get(obj).unwrap();
                ctx.solver.assert(&path_cond.implies(&z3_dest._eq(z3_obj)));
            }
        }
        InstructionKind::StructSet(dest, obj, offset, val, _ty) => {
            let z3_dest = ctx.z3_arrays.get(dest).unwrap();
            let z3_obj = ctx.z3_arrays.get(obj).unwrap();
            let z3_offset = Int::from_i64(ctx.ctx, *offset as i64);
            if let Some(z3_val) = ctx.z3_ints.get(val) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&z3_obj.store(&z3_offset, z3_val))));
            } else if let Some(z3_val) = ctx.z3_reals.get(val) {
                ctx.solver
                    .assert(&path_cond.implies(&z3_dest._eq(&z3_obj.store(&z3_offset, z3_val))));
            }
        }
        InstructionKind::Borrow(dest, src) | InstructionKind::MutBorrow(dest, src) => {
            if let Some(z3_dest) = ctx.z3_ints.get(dest) {
                if let Some(z3_src) = ctx.z3_ints.get(src) {
                    ctx.solver.assert(&path_cond.implies(&z3_dest._eq(z3_src)));
                }
            } else if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                if let Some(z3_src) = ctx.z3_reals.get(src) {
                    ctx.solver.assert(&path_cond.implies(&z3_dest._eq(z3_src)));
                }
            } else if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                if let Some(z3_src) = ctx.z3_arrays.get(src) {
                    ctx.solver.assert(&path_cond.implies(&z3_dest._eq(z3_src)));
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
    idx: &Int,
    size: i64,
    dest_id: usize,
) -> Result<(), String> {
    let zero = Int::from_i64(ctx.ctx, 0);
    let sz = Int::from_i64(ctx.ctx, size);

    ctx.solver.push();
    ctx.solver.assert(path_cond);
    ctx.solver.assert(&idx.lt(&zero));
    if ctx.solver.check() == SatResult::Sat {
        return Err(format!(
            "Potential out-of-bounds access (index < 0) at v{}",
            dest_id
        ));
    }
    ctx.solver.pop(1);

    ctx.solver.push();
    ctx.solver.assert(path_cond);
    ctx.solver.assert(&idx.ge(&sz));
    if ctx.solver.check() == SatResult::Sat {
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
    idx: &Int,
    len: &Int,
    dest_id: usize,
) -> Result<(), String> {
    let zero = Int::from_i64(ctx.ctx, 0);

    ctx.solver.push();
    ctx.solver.assert(path_cond);
    ctx.solver.assert(&idx.lt(&zero));
    if ctx.solver.check() == SatResult::Sat {
        return Err(format!(
            "Potential out-of-bounds buffer access (index < 0) at v{}",
            dest_id
        ));
    }
    ctx.solver.pop(1);

    ctx.solver.push();
    ctx.solver.assert(path_cond);
    ctx.solver.assert(&idx.ge(len));
    if ctx.solver.check() == SatResult::Sat {
        return Err(format!(
            "Potential out-of-bounds buffer access (index >= len) at v{}",
            dest_id
        ));
    }
    ctx.solver.pop(1);
    Ok(())
}
