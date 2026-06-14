use super::TranslationContext;
use crate::refinement_parser::{parse_array_refinement, parse_refinement};
use lila_ir::ir::{Instruction, InstructionKind, Type, Value};

pub fn init_values<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &mut TranslationContext<'_, B>,
) -> Result<(), String> {
    for i in 0..ctx.func.value_count {
        let val = Value(i);
        let ty = ctx.func.get_type(val);

        let mut is_mem_obj = false;
        let mut curr_ty = ty.clone();
        let mut inner_ty = ty.clone();
        let mut propagated_constraint = None;
        loop {
            match curr_ty {
                Type::Refined(inner, constraint) => {
                    propagated_constraint = Some(constraint.clone());
                    curr_ty = *inner.clone();
                    inner_ty = *inner;
                }
                Type::Array(inner, _) => {
                    is_mem_obj = true;
                    inner_ty = *inner;
                    break;
                }
                Type::Struct(_) | Type::Tuple(_) => {
                    is_mem_obj = true;
                    // Composite types and pointers are modeled as Int -> BV.
                    inner_ty = Type::I64;
                    break;
                }
                _ => break,
            }
        }

        if is_mem_obj {
            let bit_width = inner_ty.int_bit_width().unwrap_or(64);
            let z3_val = ctx.backend.array_const(
                &format!("{}_v{}_{}", ctx.func.name, i, ctx.uid),
                inner_ty.is_float(),
                bit_width,
            );
            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let ref_expr = parse_array_refinement(refinement, &z3_val, inner_ty.is_float())?;
                ctx.backend.assert(&ref_expr);
                ctx.has_refinements = true;
            }
            if let Some(constraint) = &propagated_constraint {
                let ref_expr = parse_array_refinement(constraint, &z3_val, inner_ty.is_float())?;
                ctx.backend.assert(&ref_expr);
            }

            ctx.z3_arrays.insert(val, z3_val);
        } else if let Type::Enum(ref name) = inner_ty {
            // Model Enums as a tag (BV) and a payload (Array)
            let tag_val = ctx
                .backend
                .bv_const(&format!("{}_v{}_tag_{}", ctx.func.name, i, ctx.uid), 8);

            // Assert valid tag range
            if let Some(variants) = ctx.func.enum_layouts.get(name) {
                let num_variants = variants.len();
                let max_tag = ctx.backend.bv_from_i64(num_variants as i64, 8);
                let __inner = ctx.backend.bv_ult(&tag_val, &max_tag);
                ctx.backend.assert(&__inner);
            }

            ctx.z3_bvs.insert(val, tag_val);

            let payload_val = ctx.backend.array_const(
                &format!("{}_v{}_payload_{}", ctx.func.name, i, ctx.uid),
                false,
                64,
            );
            ctx.z3_arrays.insert(val, payload_val);
        } else if let Type::Tensor(base_ty, dims) = &inner_ty {
            let mut z3_dims = Vec::new();
            let zero = ctx.backend.int_from_i64(0);
            for dim_name in dims.iter() {
                // Ensure unique name across different tensors sharing same dim name string
                // Wait, if two tensors have "M", they MUST share the same size. 
                // We should scope it globally per function.
                let z3_dim = ctx.backend.int_const(dim_name);
                let __tmp = ctx.backend.int_lt(&zero, &z3_dim);
                ctx.backend.assert(&__tmp);
                z3_dims.push(z3_dim);
            }
            ctx.z3_tensor_dims.insert(val, z3_dims);

            let bit_width = base_ty.int_bit_width().unwrap_or(64);
            let payload_val = ctx.backend.array_const(
                &format!("{}_v{}_tensor_data_{}", ctx.func.name, i, ctx.uid),
                base_ty.is_float(),
                bit_width,
            );
            ctx.z3_arrays.insert(val, payload_val);
        } else if let Type::Buffer(_) = inner_ty {
            let z3_len = ctx
                .backend
                .bv_const(&format!("{}_v{}_len_{}", ctx.func.name, i, ctx.uid), 64);
            let zero = ctx.backend.bv_from_i64(0, 64);
            let __tmp = ctx.backend.bv_sge(&z3_len, &zero);
            ctx.backend.assert(&__tmp);

            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let z3_int = ctx.backend.bv_to_int(&z3_len, true);
                let ref_expr = parse_refinement(refinement, &z3_int, Some(&z3_len))?;
                ctx.backend.assert(&ref_expr);
                ctx.has_refinements = true;
                ctx.z3_ints.insert(val, z3_int.clone());
            }
            if let Some(constraint) = &propagated_constraint {
                let z3_int = ctx.backend.bv_to_int(&z3_len, true);
                let ref_expr = parse_refinement(constraint, &z3_int, Some(&z3_len))?;
                ctx.backend.assert(&ref_expr);
                ctx.z3_ints.insert(val, z3_int);
            }
            ctx.z3_bvs.insert(val, z3_len);
        } else if let Type::Pointer(_) = inner_ty {
            let addr_val = ctx
                .backend
                .bv_const(&format!("{}_v{}_ptr_{}", ctx.func.name, i, ctx.uid), 64);
            ctx.z3_bvs.insert(val, addr_val);

            let payload_val = ctx.backend.array_const(
                &format!("{}_v{}_heap_{}", ctx.func.name, i, ctx.uid),
                false,
                64,
            );
            ctx.z3_arrays.insert(val, payload_val);
        } else if inner_ty.is_float() {
            let is_f32 = inner_ty.is_float32();
            let z3_val = ctx
                .backend
                .float_const(&format!("{}_v{}_{}", ctx.func.name, i, ctx.uid), is_f32);

            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let ref_expr =
                    crate::refinement_parser::parse_float_refinement(refinement, &z3_val)?;
                ctx.backend.assert(&ref_expr);
                ctx.has_refinements = true;
            }
            if let Some(constraint) = &propagated_constraint {
                let ref_expr =
                    crate::refinement_parser::parse_float_refinement(constraint, &z3_val)?;
                ctx.backend.assert(&ref_expr);
            }
            ctx.z3_floats.insert(val, z3_val);
        } else {
            let bit_width = inner_ty.int_bit_width().unwrap_or(64);
            let z3_val = ctx
                .backend
                .bv_const(&format!("{}_v{}_{}", ctx.func.name, i, ctx.uid), bit_width);

            if let Some(refinement) = ctx.func.refinements.get(&val) {
                let is_signed = !matches!(
                    inner_ty,
                    Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::Bool
                );
                let z3_int = ctx.backend.bv_to_int(&z3_val, is_signed);
                let ref_expr = parse_refinement(refinement, &z3_int, Some(&z3_val))?;
                ctx.backend.assert(&ref_expr);
                ctx.has_refinements = true;
                ctx.z3_ints.insert(val, z3_int);
            }
            if let Some(constraint) = &propagated_constraint {
                let is_signed = !matches!(
                    inner_ty,
                    Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::Bool
                );
                let z3_int = ctx.backend.bv_to_int(&z3_val, is_signed);
                let ref_expr = parse_refinement(constraint, &z3_int, Some(&z3_val))?;
                ctx.backend.assert(&ref_expr);
                ctx.z3_ints.insert(val, z3_int);
            }
            ctx.z3_bvs.insert(val, z3_val);
        }
    }
    Ok(())
}

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
    path_cond: &B::Bool,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::ArrayLoad(dest, arr, idx) => {
            let z3_arr = ctx
                .z3_arrays
                .get(arr)
                .cloned()
                .ok_or_else(|| format!("Array {} not modeled", arr))?;
            let z3_idx = ctx
                .z3_bvs
                .get(idx)
                .cloned()
                .ok_or_else(|| format!("Index {} not modeled", idx))?;

            if let Type::Array(_, Some(size)) = ctx.func.get_type(*arr) {
                check_bounds(ctx, path_cond, &z3_idx, size as i64, dest.0, inst.location)?;
            }

            let z3_idx_int = ctx.backend.bv_to_int(&z3_idx, true);

            if let Some(z3_dest) = ctx.z3_bvs.get(dest).cloned() {
                let res = ctx.backend.array_select_bv(&z3_arr, &z3_idx_int);
                let __inner = ctx.backend.bv_eq(&z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else if let Some(z3_dest) = ctx.z3_floats.get(dest).cloned() {
                let res = ctx.backend.array_select_float(&z3_arr, &z3_idx_int);
                let __inner = ctx.backend.float_eq(&z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::ArrayStore(dest, arr, idx, val, _ty) => {
            let z3_dest = ctx.z3_arrays.get(dest).cloned().unwrap();
            let z3_arr = ctx.z3_arrays.get(arr).cloned().unwrap();
            let z3_idx = ctx.z3_bvs.get(idx).cloned().unwrap();

            if let Type::Array(_, Some(size)) = ctx.func.get_type(*arr) {
                check_bounds(ctx, path_cond, &z3_idx, size as i64, dest.0, inst.location)?;
            }

            let z3_idx_int = ctx.backend.bv_to_int(&z3_idx, true);

            if let Some(z3_val) = ctx.z3_bvs.get(val).cloned() {
                let stored = ctx.backend.array_store_bv(&z3_arr, &z3_idx_int, &z3_val);
                let __inner = ctx.backend.array_eq(&z3_dest, &stored);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else if let Some(z3_val) = ctx.z3_floats.get(val).cloned() {
                let stored = ctx.backend.array_store_float(&z3_arr, &z3_idx_int, &z3_val);
                let __inner = ctx.backend.array_eq(&z3_dest, &stored);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else {
                let __inner = ctx.backend.array_eq(&z3_dest, &z3_arr);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::TensorLoad(dest, tensor, indices) => {
            let z3_tensor_data = ctx.z3_arrays.get(tensor).cloned().unwrap();
            let dims = ctx
                .z3_tensor_dims
                .get(tensor)
                .expect("Tensor dimensions not found")
                .clone();

            // Calculate flat index in Z3
            let mut z3_flat_idx = ctx.backend.bv_from_i64(0, 64);
            let mut z3_stride = ctx.backend.bv_from_i64(1, 64);

            for i in (0..indices.len()).rev() {
                let idx_val = indices[i];
                let z3_idx = ctx
                    .z3_bvs
                    .get(&idx_val)
                    .cloned()
                    .expect("Index not modeled");
                let z3_dim_int = &dims[i];

                check_symbolic_bounds(
                    ctx,
                    path_cond,
                    &z3_idx,
                    z3_dim_int,
                    dest.0,
                    inst.location,
                )?;

                let z3_dim_bv = ctx.backend.int_to_bv(z3_dim_int, 64);
                let term = ctx.backend.bv_mul(&z3_idx, &z3_stride);
                z3_flat_idx = ctx.backend.bv_add(&z3_flat_idx, &term);

                if i > 0 {
                    z3_stride = ctx.backend.bv_mul(&z3_stride, &z3_dim_bv);
                }
            }

            let z3_idx_int = ctx.backend.bv_to_int(&z3_flat_idx, true);

            if let Some(z3_dest) = ctx.z3_bvs.get(dest).cloned() {
                let res = ctx.backend.array_select_bv(&z3_tensor_data, &z3_idx_int);
                let __inner = ctx.backend.bv_eq(&z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else if let Some(z3_dest) = ctx.z3_floats.get(dest).cloned() {
                let res = ctx
                    .backend
                    .array_select_float(&z3_tensor_data, &z3_idx_int);
                let __inner = ctx.backend.float_eq(&z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::TensorDim(dest, tensor, index) => {
            let dims = ctx
                .z3_tensor_dims
                .get(tensor)
                .expect("Tensor dimensions not found");
            let dim = &dims[*index];
            let z3_dest = ctx.z3_bvs.get(dest).expect("Dest not modeled");

            let dim_bv = ctx.backend.int_to_bv(dim, 64);
            let __inner = ctx.backend.bv_eq(z3_dest, &dim_bv);
            let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
            ctx.backend.assert(&__tmp);
        }
        InstructionKind::TensorBroadcast(dest, _src, target_dims) => {
            let mut z3_target_dims = Vec::new();
            for dim_val in target_dims {
                let z3_dim_bv = ctx.z3_bvs.get(dim_val).expect("Dim value not modeled");
                let z3_dim_int = ctx.backend.bv_to_int(z3_dim_bv, false);
                z3_target_dims.push(z3_dim_int);
            }
            ctx.z3_tensor_dims.insert(*dest, z3_target_dims);
        }
        InstructionKind::TensorStore(dest, tensor, indices, val) => {
            let z3_dest_data = ctx.z3_arrays.get(dest).cloned().unwrap();
            let z3_tensor_data = ctx.z3_arrays.get(tensor).cloned().unwrap();
            let dims = ctx
                .z3_tensor_dims
                .get(tensor)
                .expect("Tensor dimensions not found")
                .clone();

            // Calculate flat index in Z3
            let mut z3_flat_idx = ctx.backend.bv_from_i64(0, 64);
            let mut z3_stride = ctx.backend.bv_from_i64(1, 64);

            for i in (0..indices.len()).rev() {
                let idx_val = indices[i];
                let z3_idx = ctx
                    .z3_bvs
                    .get(&idx_val)
                    .cloned()
                    .expect("Index not modeled");
                let z3_dim_int = &dims[i];

                check_symbolic_bounds(
                    ctx,
                    path_cond,
                    &z3_idx,
                    z3_dim_int,
                    dest.0,
                    inst.location,
                )?;

                let z3_dim_bv = ctx.backend.int_to_bv(z3_dim_int, 64);
                let term = ctx.backend.bv_mul(&z3_idx, &z3_stride);
                z3_flat_idx = ctx.backend.bv_add(&z3_flat_idx, &term);

                if i > 0 {
                    z3_stride = ctx.backend.bv_mul(&z3_stride, &z3_dim_bv);
                }
            }

            let z3_idx_int = ctx.backend.bv_to_int(&z3_flat_idx, true);

            if let Some(z3_val) = ctx.z3_bvs.get(val).cloned() {
                let stored = ctx
                    .backend
                    .array_store_bv(&z3_tensor_data, &z3_idx_int, &z3_val);
                let __inner = ctx.backend.array_eq(&z3_dest_data, &stored);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else if let Some(z3_val) = ctx.z3_floats.get(val).cloned() {
                let stored = ctx
                    .backend
                    .array_store_float(&z3_tensor_data, &z3_idx_int, &z3_val);
                let __inner = ctx.backend.array_eq(&z3_dest_data, &stored);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }

            // Propagate dimensions
            ctx.z3_tensor_dims.insert(*dest, dims.clone());
        }
        InstructionKind::BufferLoad(dest, buf, idx) => {
            let z3_idx = ctx.z3_bvs.get(idx).cloned().ok_or_else(|| {
                format!("BufferLoad: v{} (idx) not found in z3_bvs", idx.0)
            })?;
            let z3_len = ctx.z3_bvs.get(buf).cloned().ok_or_else(|| {
                format!("BufferLoad: v{} (buf) not found in z3_bvs", buf.0)
            })?;
            check_buffer_bounds(ctx, path_cond, &z3_idx, &z3_len, dest.0, inst.location)?;

            // Model the loaded value as an unconstrained constant of the appropriate sort
            let dest_ty = ctx.func.get_type(*dest);
            if dest_ty.is_float() {
                let is_f32 = dest_ty.is_float32();
                let z3_val = ctx.backend.float_const(
                    &format!("{}_v{}_load_{}", ctx.func.name, dest.0, ctx.uid),
                    is_f32,
                );
                ctx.z3_floats.insert(*dest, z3_val);
            } else {
                let bit_width = dest_ty.int_bit_width().unwrap_or(64);
                let z3_val = ctx.backend.bv_const(
                    &format!("{}_v{}_load_{}", ctx.func.name, dest.0, ctx.uid),
                    bit_width,
                );
                ctx.z3_bvs.insert(*dest, z3_val);
            }
        }
        InstructionKind::TensorSum(dest, tensor)
        | InstructionKind::TensorMax(dest, tensor)
        | InstructionKind::TensorMin(dest, tensor) => {
            if !ctx.z3_tensor_dims.contains_key(tensor) {
                let loc_info = inst
                    .location
                    .map(|l| format!(" at {}", l))
                    .unwrap_or_default();
                return Err(format!(
                    "Invalid reduction: v{} is not a tensor{}",
                    tensor.0, loc_info
                ));
            }
            // Reduction results in a scalar (rank 0)
            ctx.z3_tensor_dims.insert(*dest, Vec::new());
        }
        InstructionKind::BufferStore(dest, buf, idx, _val, _) => {
            let z3_idx = ctx.z3_bvs.get(idx).cloned().ok_or_else(|| {
                format!("BufferStore: v{} (idx) not found in z3_bvs", idx.0)
            })?;
            let z3_len = ctx.z3_bvs.get(buf).cloned().ok_or_else(|| {
                format!("BufferStore: v{} (buf) not found in z3_bvs", buf.0)
            })?;
            check_buffer_bounds(ctx, path_cond, &z3_idx, &z3_len, dest.0, inst.location)?;
            if let (Some(z3_dest_len), Some(z3_buf_len)) =
                (ctx.z3_bvs.get(dest).cloned(), ctx.z3_bvs.get(buf).cloned())
            {
                let __inner = ctx.backend.bv_eq(&z3_dest_len, &z3_buf_len);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::BufferLen(dest, buf) => {
            let z3_len = ctx.z3_bvs.get(buf).cloned().ok_or_else(|| {
                format!("BufferLen: v{} (buf) not found in z3_bvs", buf.0)
            })?;
            let z3_dest = ctx.z3_bvs.get(dest).cloned().ok_or_else(|| {
                format!("BufferLen: v{} (dest) not found in z3_bvs", dest.0)
            })?;
            let __inner = ctx.backend.bv_eq(&z3_dest, &z3_len);
            let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
            ctx.backend.assert(&__tmp);
        }
        InstructionKind::StructCreate(dest, struct_name, args) => {
            if let Some(z3_dest) = ctx.z3_arrays.get(dest).cloned() {
                let z3_zero_arr = ctx.backend.array_const(
                    &format!("{}_v{}_zero_{}", ctx.func.name, dest.0, ctx.uid),
                    false,
                    64,
                );
                let mut current_state = z3_zero_arr;

                let fields = ctx.func.struct_layouts.get(struct_name).unwrap();
                let mut offset = 0;
                for (i, p_val) in args.iter().enumerate() {
                    let f_ty = &fields[i].1;
                    let f_align = f_ty.align(&ctx.func.struct_layouts);
                    offset = (offset + f_align - 1) & !(f_align - 1);

                    if f_ty.is_composite() {
                        current_state =
                            super::copy_composite(ctx, current_state, *p_val, f_ty, offset as i64);
                    } else {
                        let z3_offset = ctx.backend.int_from_i64(offset as i64);
                        if let Some(z3_v) = ctx.z3_bvs.get(p_val).cloned() {
                            current_state =
                                ctx.backend
                                    .array_store_bv(&current_state, &z3_offset, &z3_v);
                        } else if let Some(z3_v) = ctx.z3_floats.get(p_val).cloned() {
                            current_state =
                                ctx.backend
                                    .array_store_float(&current_state, &z3_offset, &z3_v);
                        }
                    }
                    offset += f_ty.size(&ctx.func.struct_layouts);
                }
                let __inner = ctx.backend.array_eq(&z3_dest, &current_state);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::StructLoad(dest, obj, offset) => {
            let z3_obj = ctx.z3_arrays.get(obj).cloned().unwrap();
            let z3_offset = ctx.backend.int_from_i64(*offset as i64);
            if let Some(z3_dest) = ctx.z3_bvs.get(dest).cloned() {
                let res = ctx.backend.array_select_bv(&z3_obj, &z3_offset);
                let __inner = ctx.backend.bv_eq(&z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else if let Some(z3_dest) = ctx.z3_floats.get(dest).cloned() {
                let res = ctx.backend.array_select_float(&z3_obj, &z3_offset);
                let __inner = ctx.backend.float_eq(&z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::StructOffset(dest, obj, _offset) => {
            if let Some(z3_dest) = ctx.z3_arrays.get(dest).cloned() {
                let z3_obj = ctx.z3_arrays.get(obj).cloned().unwrap();
                let __inner = ctx.backend.array_eq(&z3_dest, &z3_obj);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::StructSet(dest, obj, offset, val, _ty) => {
            let z3_dest = ctx.z3_arrays.get(dest).cloned().unwrap();
            let z3_obj = ctx.z3_arrays.get(obj).cloned().unwrap();
            let z3_offset = ctx.backend.int_from_i64(*offset as i64);
            if let Some(z3_val) = ctx.z3_bvs.get(val).cloned() {
                let stored = ctx.backend.array_store_bv(&z3_obj, &z3_offset, &z3_val);
                let __inner = ctx.backend.array_eq(&z3_dest, &stored);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else if let Some(z3_val) = ctx.z3_floats.get(val).cloned() {
                let stored = ctx.backend.array_store_float(&z3_obj, &z3_offset, &z3_val);
                let __inner = ctx.backend.array_eq(&z3_dest, &stored);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::EnumCreate(dest, _enum_name, tag_idx, payload) => {
            if let Some(z3_dest_tag) = ctx.z3_bvs.get(dest).cloned() {
                let z3_tag_val = ctx.backend.bv_from_i64(*tag_idx as i64, 8);
                let __inner = ctx.backend.bv_eq(&z3_dest_tag, &z3_tag_val);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
            if let Some(z3_dest_payload) = ctx.z3_arrays.get(dest).cloned() {
                let z3_zero_arr = ctx.backend.array_const(
                    &format!("{}_v{}_zero_{}", ctx.func.name, dest.0, ctx.uid),
                    false,
                    64,
                );
                let mut current_state = z3_zero_arr.clone();

                if let Some(payload_val) = payload {
                    let enum_layouts = ctx.func.enum_layouts.get(_enum_name).unwrap();
                    let payload_ty = &enum_layouts[*tag_idx].1;
                    let p_align = payload_ty.align(&ctx.func.struct_layouts);
                    let mut offset = 1;
                    offset = (offset + p_align - 1) & !(p_align - 1);

                    current_state = super::copy_composite(
                        ctx,
                        current_state,
                        *payload_val,
                        payload_ty,
                        offset as i64,
                    );
                }
                let __inner = ctx.backend.array_eq(&z3_dest_payload, &current_state);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::EnumIsVariant(dest, obj, tag_idx) => {
            if let Some(z3_dest) = ctx.z3_bvs.get(dest).cloned() {
                let z3_obj_tag = ctx.z3_bvs.get(obj).cloned().unwrap();
                let expected_tag = ctx.backend.bv_from_i64(*tag_idx as i64, 8);

                let is_match = ctx.backend.bv_eq(&z3_obj_tag, &expected_tag);

                let dest_size = ctx.func.get_type(*dest).int_bit_width().unwrap_or(1);
                let one = ctx.backend.bv_from_i64(1, dest_size);
                let zero = ctx.backend.bv_from_i64(0, dest_size);

                let __is_true_eq_one = ctx.backend.bv_eq(&z3_dest, &one);
                let __implies1 = ctx.backend.bool_implies(&is_match, &__is_true_eq_one);

                let __not_is_true = ctx.backend.bool_not(&is_match);
                let __is_false_eq_zero = ctx.backend.bv_eq(&z3_dest, &zero);
                let __implies2 = ctx
                    .backend
                    .bool_implies(&__not_is_true, &__is_false_eq_zero);

                let __both = ctx.backend.bool_and(&[&__implies1, &__implies2]);
                let __tmp = ctx.backend.bool_implies(path_cond, &__both);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::EnumGetTag(dest, obj) => {
            if let Some(z3_dest) = ctx.z3_bvs.get(dest).cloned() {
                let z3_obj_tag = ctx.z3_bvs.get(obj).cloned().unwrap();
                let __inner = ctx.backend.bv_eq(&z3_dest, &z3_obj_tag);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::EnumExtract(dest, obj, _tag_idx) => {
            let z3_obj_payload = ctx.z3_arrays.get(obj).cloned().unwrap();
            if let Some(z3_dest) = ctx.z3_arrays.get(dest).cloned() {
                let __inner = ctx.backend.array_eq(&z3_dest, &z3_obj_payload);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::Alloc(dest, _ty) => {
            // In our model, Alloc produces a fresh value which is already initialized in init_values.
            // We just need to assert it's not null.
            let ptr_val = ctx.z3_bvs.get(dest).cloned().unwrap();
            let zero = ctx.backend.bv_from_i64(0, 64);
            let is_null = ctx.backend.bv_eq(&ptr_val, &zero);
            let not_null = ctx.backend.bool_not(&is_null);
            let __tmp = ctx.backend.bool_implies(path_cond, &not_null);
            ctx.backend.assert(&__tmp);
        }
        InstructionKind::PointerLoad(dest, ptr) => {
            let ptr_payload = ctx.z3_arrays.get(ptr).cloned().unwrap();
            let dest_ty = ctx.func.get_type(*dest);
            if dest_ty.is_composite() {
                let dest_payload = ctx.z3_arrays.get(dest).cloned().unwrap();
                let __inner = ctx.backend.array_eq(&dest_payload, &ptr_payload);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else {
                // Primitive load
                if dest_ty.is_float() {
                    // Floats not yet supported in heap model
                } else if let Some(dest_bv) = ctx.z3_bvs.get(dest).cloned() {
                    let zero = ctx.backend.int_from_i64(0);
                    let res = ctx.backend.array_select_bv(&ptr_payload, &zero);
                    let __inner = ctx.backend.bv_eq(&dest_bv, &res);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                }
            }
        }
        InstructionKind::PointerStore(ptr, val) => {
            let ptr_payload = ctx.z3_arrays.get(ptr).cloned().unwrap();
            let val_ty = ctx.func.get_type(*val);
            if val_ty.is_composite() {
                let val_payload = ctx.z3_arrays.get(val).cloned().unwrap();
                let __inner = ctx.backend.array_eq(&ptr_payload, &val_payload);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else {
                // Primitive store
                if val_ty.is_float() {
                    // Floats not yet supported in heap model
                } else if let Some(val_bv) = ctx.z3_bvs.get(val).cloned() {
                    let zero = ctx.backend.int_from_i64(0);
                    let new_payload = ctx.backend.array_store_bv(&ptr_payload, &zero, &val_bv);
                    // We need to update the ptr_payload mapping.
                    ctx.z3_arrays.insert(*ptr, new_payload);
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn check_bounds<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &mut TranslationContext<'_, B>,
    path_cond: &B::Bool,
    idx: &B::BV,
    size: i64,
    dest_id: usize,
    location: Option<lila_ir::ir::SourceLocation>,
) -> Result<(), String> {
    let bit_width = ctx
        .func
        .get_type(lila_ir::ir::Value(dest_id))
        .int_bit_width()
        .unwrap_or(64);
    let zero = ctx.backend.bv_from_i64(0, bit_width);

    ctx.backend.push();
    ctx.backend.assert(path_cond);
    let __tmp = ctx.backend.bv_slt(idx, &zero);
    ctx.backend.assert(&__tmp);
    if ctx.backend.check()? {
        let loc_info = location.map(|l| format!(" at {}", l)).unwrap_or_default();
        return Err(format!(
            "Potential out-of-bounds access (negative index) at v{}{}",
            dest_id, loc_info
        ));
    }
    ctx.backend.pop(1);

    ctx.backend.push();
    ctx.backend.assert(path_cond);
    let check_len = ctx.backend.bv_from_i64(size, bit_width);
    let __tmp2 = ctx.backend.bv_sge(idx, &check_len);
    ctx.backend.assert(&__tmp2);
    if ctx.backend.check()? {
        let loc_info = location.map(|l| format!(" at {}", l)).unwrap_or_default();
        return Err(format!(
            "Potential out-of-bounds access (index >= length) at v{}{}",
            dest_id, loc_info
        ));
    }
    ctx.backend.pop(1);
    Ok(())
}

fn check_symbolic_bounds<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &mut TranslationContext<'_, B>,
    path_cond: &B::Bool,
    idx: &B::BV,
    size: &B::Int,
    dest_id: usize,
    location: Option<lila_ir::ir::SourceLocation>,
) -> Result<(), String> {
    let idx_int = ctx.backend.bv_to_int(idx, true);
    let zero = ctx.backend.int_from_i64(0);

    ctx.backend.push();
    ctx.backend.assert(path_cond);
    let __tmp = ctx.backend.int_lt(&idx_int, &zero);
    ctx.backend.assert(&__tmp);
    if ctx.backend.check()? {
        let loc_info = location.map(|l| format!(" at {}", l)).unwrap_or_default();
        return Err(format!(
            "Potential out-of-bounds access (negative index) at v{}{}",
            dest_id, loc_info
        ));
    }
    ctx.backend.pop(1);

    ctx.backend.push();
    ctx.backend.assert(path_cond);
    let __tmp2 = ctx.backend.int_ge(&idx_int, size);
    ctx.backend.assert(&__tmp2);
    if ctx.backend.check()? {
        let loc_info = location.map(|l| format!(" at {}", l)).unwrap_or_default();
        return Err(format!(
            "Potential out-of-bounds access (index >= length) at v{}{}",
            dest_id, loc_info
        ));
    }
    ctx.backend.pop(1);
    Ok(())
}

fn check_buffer_bounds<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &mut TranslationContext<'_, B>,
    path_cond: &B::Bool,
    idx: &B::BV,
    len: &B::BV,
    dest_id: usize,
    location: Option<lila_ir::ir::SourceLocation>,
) -> Result<(), String> {
    // Buffer indices and lengths are always 64-bit in Lila
    let zero = ctx.backend.bv_from_i64(0, 64);

    ctx.backend.push();
    ctx.backend.assert(path_cond);
    let __tmp = ctx.backend.bv_slt(idx, &zero);
    ctx.backend.assert(&__tmp);
    if ctx.backend.check()? {
        let loc_info = location.map(|l| format!(" at {}", l)).unwrap_or_default();
        return Err(format!(
            "Potential out-of-bounds buffer access (index < 0) at v{}{}",
            dest_id, loc_info
        ));
    }
    ctx.backend.pop(1);

    ctx.backend.push();
    ctx.backend.assert(path_cond);

    let __tmp2 = ctx.backend.bv_sge(idx, len);
    ctx.backend.assert(&__tmp2);
    if ctx.backend.check()? {
        let loc_info = location.map(|l| format!(" at {}", l)).unwrap_or_default();
        return Err(format!(
            "Potential out-of-bounds buffer access (index >= len) at v{}{}",
            dest_id, loc_info
        ));
    }
    ctx.backend.pop(1);
    Ok(())
}
