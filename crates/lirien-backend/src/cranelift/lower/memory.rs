use super::{get_len, get_val, translate_type, CodegenContext, LoweringError};
use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};
use lirien_ir::ir::{InstructionKind, Type as SsaType};

fn compute_tensor_flat_index<M: Module>(
    ctx: &mut CodegenContext<M>,
    indices: &[lirien_ir::ir::Value],
    dims: &[Value],
) -> Value {
    let mut flat_idx = get_val(&ctx.values, &indices[indices.len() - 1]);
    let mut stride = ctx.builder.ins().iconst(types::I64, 1);

    for i in (0..indices.len() - 1).rev() {
        let dim_val = dims[i + 1];
        stride = ctx.builder.ins().imul(stride, dim_val);
        let idx_val = get_val(&ctx.values, &indices[i]);
        let term = ctx.builder.ins().imul(idx_val, stride);
        flat_idx = ctx.builder.ins().iadd(flat_idx, term);
    }
    flat_idx
}

fn lower_tensor_arith<M: Module>(
    ctx: &mut CodegenContext<M>,
    dest: &lirien_ir::ir::Value,
    lhs: &lirien_ir::ir::Value,
    rhs: &lirien_ir::ir::Value,
    op_code: u8,
) -> Result<(), LoweringError> {
    let l_ptr = get_val(&ctx.values, lhs);
    let r_ptr = get_val(&ctx.values, rhs);
    let dims = ctx
        .tensor_dims
        .get(lhs)
        .expect("Tensor dimensions not found")
        .clone();

    let mut total_size = dims[0];
    for &dim in dims.iter().skip(1) {
        total_size = ctx.builder.ins().imul(total_size, dim);
    }

    let op_val = ctx.builder.ins().iconst(types::I8, op_code as i64);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(types::I64)); // a
    sig.params.push(AbiParam::new(types::I64)); // b
    sig.params.push(AbiParam::new(types::I64)); // size
    sig.params.push(AbiParam::new(types::I8)); // op
    sig.returns.push(AbiParam::new(types::I64)); // result ptr

    let func_id = ctx
        .module
        .declare_function("lirien_tensor_arith_f32", Linkage::Import, &sig)
        .map_err(|e| e.to_string())?;
    let local_func = ctx.module.declare_func_in_func(func_id, ctx.builder.func);

    let call = ctx
        .builder
        .ins()
        .call(local_func, &[l_ptr, r_ptr, total_size, op_val]);
    let res_ptr = ctx.builder.inst_results(call)[0];

    ctx.values.insert(*dest, res_ptr);
    ctx.tensor_dims.insert(*dest, dims);
    Ok(())
}

fn lower_tensor_reduce<M: Module>(
    ctx: &mut CodegenContext<M>,
    dest: &lirien_ir::ir::Value,
    tensor: &lirien_ir::ir::Value,
    op_code: u8,
) -> Result<(), LoweringError> {
    let t_ptr = get_val(&ctx.values, tensor);
    let dims = ctx
        .tensor_dims
        .get(tensor)
        .expect("Tensor dimensions not found");

    let mut total_size = dims[0];
    for &dim in dims.iter().skip(1) {
        total_size = ctx.builder.ins().imul(total_size, dim);
    }

    let op_val = ctx.builder.ins().iconst(types::I8, op_code as i64);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(types::I64)); // a
    sig.params.push(AbiParam::new(types::I64)); // size
    sig.params.push(AbiParam::new(types::I8)); // op
    sig.returns.push(AbiParam::new(types::F32)); // result

    let func_id = ctx
        .module
        .declare_function("lirien_tensor_reduce_f32", Linkage::Import, &sig)
        .map_err(|e| e.to_string())?;
    let local_func = ctx.module.declare_func_in_func(func_id, ctx.builder.func);

    let call = ctx
        .builder
        .ins()
        .call(local_func, &[t_ptr, total_size, op_val]);
    let res = ctx.builder.inst_results(call)[0];
    ctx.values.insert(*dest, res);
    Ok(())
}

fn lower_tensor_scalar_arith<M: Module>(
    ctx: &mut CodegenContext<M>,
    dest: &lirien_ir::ir::Value,
    tensor: &lirien_ir::ir::Value,
    scalar: &lirien_ir::ir::Value,
    op_code: u8,
) -> Result<(), LoweringError> {
    let t_ptr = get_val(&ctx.values, tensor);
    let s_val = get_val(&ctx.values, scalar);
    let dims = ctx
        .tensor_dims
        .get(tensor)
        .expect("Tensor dimensions not found")
        .clone();

    let mut total_size = dims[0];
    for &dim in dims.iter().skip(1) {
        total_size = ctx.builder.ins().imul(total_size, dim);
    }

    let op_val = ctx.builder.ins().iconst(types::I8, op_code as i64);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(types::I64)); // a
    sig.params.push(AbiParam::new(types::F32)); // b (scalar)
    sig.params.push(AbiParam::new(types::I64)); // size
    sig.params.push(AbiParam::new(types::I8)); // op
    sig.returns.push(AbiParam::new(types::I64)); // result ptr

    let func_id = ctx
        .module
        .declare_function("lirien_tensor_scalar_arith_f32", Linkage::Import, &sig)
        .map_err(|e| e.to_string())?;
    let local_func = ctx.module.declare_func_in_func(func_id, ctx.builder.func);

    let call = ctx
        .builder
        .ins()
        .call(local_func, &[t_ptr, s_val, total_size, op_val]);
    let res_ptr = ctx.builder.inst_results(call)[0];

    ctx.values.insert(*dest, res_ptr);
    ctx.tensor_dims.insert(*dest, dims);
    Ok(())
}

pub fn lower<M: Module>(
    ctx: &mut CodegenContext<M>,
    kind: &InstructionKind,
) -> Result<(), LoweringError> {
    match kind {
        InstructionKind::ListCreate(dest, _elem_ty) => {
            let mut sig = ctx.module.make_signature();
            sig.returns.push(AbiParam::new(types::I64)); // list ptr

            let func = ctx
                .module
                .declare_function("lirien_list_new", Linkage::Import, &sig)?;
            let local_func = ctx.module.declare_func_in_func(func, ctx.builder.func);
            let call = ctx.builder.ins().call(local_func, &[]);
            let list_ptr = ctx.builder.inst_results(call)[0];
            ctx.values.insert(*dest, list_ptr);
        }
        InstructionKind::ListAppend(dest, list, val) => {
            let list_ptr = get_val(&ctx.values, list);
            let val_ty = ctx.ssa_func.get_type(*val);
            let val_ptr = if val_ty.is_composite() {
                get_val(&ctx.values, val)
            } else {
                let size = val_ty.size(&ctx.ssa_func.struct_layouts);
                let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    size as u32,
                ));
                let addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
                super::storage::store_to_memory(ctx, *val, addr, 0);
                addr
            };
            let elem_size = val_ty.size(&ctx.ssa_func.struct_layouts);
            let elem_size_val = ctx.builder.ins().iconst(types::I64, elem_size as i64);

            let mut sig = ctx.module.make_signature();
            sig.params.push(AbiParam::new(types::I64)); // list
            sig.params.push(AbiParam::new(types::I64)); // val_ptr
            sig.params.push(AbiParam::new(types::I64)); // elem_size

            let func = ctx
                .module
                .declare_function("lirien_list_append", Linkage::Import, &sig)?;
            let local_func = ctx.module.declare_func_in_func(func, ctx.builder.func);
            ctx.builder
                .ins()
                .call(local_func, &[list_ptr, val_ptr, elem_size_val]);

            ctx.values.insert(*dest, list_ptr);
        }
        InstructionKind::ListLen(dest, list) => {
            let list_ptr = get_val(&ctx.values, list);
            let mut sig = ctx.module.make_signature();
            sig.params.push(AbiParam::new(types::I64)); // list
            sig.returns.push(AbiParam::new(types::I64)); // len

            let func = ctx
                .module
                .declare_function("lirien_list_len", Linkage::Import, &sig)?;
            let local_func = ctx.module.declare_func_in_func(func, ctx.builder.func);
            let call = ctx.builder.ins().call(local_func, &[list_ptr]);
            let len_val = ctx.builder.inst_results(call)[0];
            ctx.values.insert(*dest, len_val);
        }
        InstructionKind::ListLoad(dest, list, index) => {
            let list_ptr = get_val(&ctx.values, list);
            let idx_val = get_val(&ctx.values, index);
            let dest_ty = ctx.ssa_func.get_type(*dest);
            let elem_size = dest_ty.size(&ctx.ssa_func.struct_layouts);
            let elem_size_val = ctx.builder.ins().iconst(types::I64, elem_size as i64);

            let mut sig = ctx.module.make_signature();
            sig.params.push(AbiParam::new(types::I64)); // list
            sig.params.push(AbiParam::new(types::I64)); // index
            sig.params.push(AbiParam::new(types::I64)); // elem_size
            sig.returns.push(AbiParam::new(types::I64)); // elem_ptr

            let func = ctx
                .module
                .declare_function("lirien_list_get", Linkage::Import, &sig)?;
            let local_func = ctx.module.declare_func_in_func(func, ctx.builder.func);
            let call = ctx
                .builder
                .ins()
                .call(local_func, &[list_ptr, idx_val, elem_size_val]);
            let addr = ctx.builder.inst_results(call)[0];

            if dest_ty.is_composite() {
                ctx.values.insert(*dest, addr);
            } else {
                let cl_ty = translate_type(&dest_ty);
                let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), addr, 0);
                ctx.values.insert(*dest, res);
            }
        }
        InstructionKind::ListStore(dest, list, index, val) => {
            let list_ptr = get_val(&ctx.values, list);
            let idx_val = get_val(&ctx.values, index);
            let val_ty = ctx.ssa_func.get_type(*val);
            let val_ptr = if val_ty.is_composite() {
                get_val(&ctx.values, val)
            } else {
                let size = val_ty.size(&ctx.ssa_func.struct_layouts);
                let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    size as u32,
                ));
                let addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
                super::storage::store_to_memory(ctx, *val, addr, 0);
                addr
            };
            let elem_size = val_ty.size(&ctx.ssa_func.struct_layouts);
            let elem_size_val = ctx.builder.ins().iconst(types::I64, elem_size as i64);

            let mut sig = ctx.module.make_signature();
            sig.params.push(AbiParam::new(types::I64)); // list
            sig.params.push(AbiParam::new(types::I64)); // index
            sig.params.push(AbiParam::new(types::I64)); // val_ptr
            sig.params.push(AbiParam::new(types::I64)); // elem_size

            let func = ctx
                .module
                .declare_function("lirien_list_set", Linkage::Import, &sig)?;
            let local_func = ctx.module.declare_func_in_func(func, ctx.builder.func);
            ctx.builder
                .ins()
                .call(local_func, &[list_ptr, idx_val, val_ptr, elem_size_val]);

            ctx.values.insert(*dest, list_ptr);
        }
        InstructionKind::BufferLoad(dest, buf, idx) => {
            let buf_ptr = get_val(&ctx.values, buf);
            let idx_val = get_val(&ctx.values, idx);
            let dest_ty = ctx.ssa_func.get_type(*dest);
            let elem_size = match &ctx.ssa_func.get_type(*buf) {
                SsaType::Buffer(inner) => inner.size(&ctx.ssa_func.struct_layouts),
                _ => 8,
            };
            let offset = ctx.builder.ins().imul_imm(idx_val, elem_size as i64);
            let addr = ctx.builder.ins().iadd(buf_ptr, offset);

            if dest_ty.is_composite() {
                ctx.values.insert(*dest, addr);
            } else {
                let cl_ty = translate_type(&dest_ty);
                let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), addr, 0);
                ctx.values.insert(*dest, res);
            }
        }
        InstructionKind::BufferStore(dest, buf, idx, val, _ty) => {
            let buf_ptr = get_val(&ctx.values, buf);
            let idx_val = get_val(&ctx.values, idx);
            let elem_size = match &ctx.ssa_func.get_type(*buf) {
                SsaType::Buffer(inner) => inner.size(&ctx.ssa_func.struct_layouts),
                _ => 8,
            };
            let offset = ctx.builder.ins().imul_imm(idx_val, elem_size as i64);
            let addr = ctx.builder.ins().iadd(buf_ptr, offset);

            super::store_to_memory(ctx, *val, addr, 0);

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
        InstructionKind::TensorLoad(dest, tensor, indices) => {
            let tensor_ptr = get_val(&ctx.values, tensor);
            let dims = ctx
                .tensor_dims
                .get(tensor)
                .expect("Tensor dimensions not found")
                .clone();

            let flat_idx = compute_tensor_flat_index(ctx, indices, &dims);

            let dest_ty = ctx.ssa_func.get_type(*dest);
            let elem_size = match &ctx.ssa_func.get_type(*tensor) {
                SsaType::Tensor(inner, _) => inner.size(&ctx.ssa_func.struct_layouts),
                _ => 4, // Default to float (4 bytes)
            };

            let offset = ctx.builder.ins().imul_imm(flat_idx, elem_size as i64);
            let addr = ctx.builder.ins().iadd(tensor_ptr, offset);

            if dest_ty.is_composite() {
                ctx.values.insert(*dest, addr);
            } else {
                let cl_ty = translate_type(&dest_ty);
                let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), addr, 0);
                ctx.values.insert(*dest, res);
            }
        }
        InstructionKind::TensorStore(dest, tensor, indices, val) => {
            let tensor_ptr = get_val(&ctx.values, tensor);
            let dims = ctx
                .tensor_dims
                .get(tensor)
                .expect("Tensor dimensions not found")
                .clone();

            let flat_idx = compute_tensor_flat_index(ctx, indices, &dims);

            let val_val = get_val(&ctx.values, val);
            let (inner_ty, _dims_strings) = match &ctx.ssa_func.get_type(*tensor) {
                SsaType::Tensor(inner, d) => (inner.clone(), d.clone()),
                _ => (Box::new(SsaType::F32), Vec::new()),
            };
            let elem_size = inner_ty.size(&ctx.ssa_func.struct_layouts);

            let offset = ctx.builder.ins().imul_imm(flat_idx, elem_size as i64);
            let addr = ctx.builder.ins().iadd(tensor_ptr, offset);

            if inner_ty.is_composite() {
                let size_val = ctx.builder.ins().iconst(types::I64, elem_size as i64);
                ctx.builder
                    .call_memcpy(ctx.module.target_config(), addr, val_val, size_val);
            } else {
                let cl_ty = translate_type(&inner_ty);
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
            }

            ctx.values.insert(*dest, tensor_ptr);
            // Re-register dimensions for the new tensor value
            ctx.tensor_dims.insert(*dest, dims);
        }
        InstructionKind::TensorDim(dest, tensor, index) => {
            let dims = ctx
                .tensor_dims
                .get(tensor)
                .expect("Tensor dimensions not found");
            let dim = dims[*index];
            ctx.values.insert(*dest, dim);
        }
        InstructionKind::TensorBroadcast(dest, src, target_dims) => {
            let src_ptr = get_val(&ctx.values, src);
            let src_dims = ctx
                .tensor_dims
                .get(src)
                .expect("Source tensor dimensions not found")
                .clone();

            let mut target_dim_vals = Vec::new();
            for dim_val in target_dims {
                target_dim_vals.push(get_val(&ctx.values, dim_val));
            }

            let src_rank = src_dims.len();
            let target_rank = target_dim_vals.len();

            let src_dims_slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                (src_rank * 8) as u32,
            ));
            for (i, &dim) in src_dims.iter().enumerate() {
                ctx.builder
                    .ins()
                    .stack_store(dim, src_dims_slot, (i * 8) as i32);
            }
            let src_dims_ptr = ctx.builder.ins().stack_addr(types::I64, src_dims_slot, 0);

            let target_dims_slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                (target_rank * 8) as u32,
            ));
            for (i, &dim) in target_dim_vals.iter().enumerate() {
                ctx.builder
                    .ins()
                    .stack_store(dim, target_dims_slot, (i * 8) as i32);
            }
            let target_dims_ptr = ctx
                .builder
                .ins()
                .stack_addr(types::I64, target_dims_slot, 0);

            let mut sig = ctx.module.make_signature();
            sig.params.push(AbiParam::new(types::I64)); // src_ptr
            sig.params.push(AbiParam::new(types::I64)); // src_dims_ptr
            sig.params.push(AbiParam::new(types::I64)); // src_rank
            sig.params.push(AbiParam::new(types::I64)); // target_dims_ptr
            sig.params.push(AbiParam::new(types::I64)); // target_rank
            sig.returns.push(AbiParam::new(types::I64)); // dest_ptr

            let func_id = ctx
                .module
                .declare_function("lirien_tensor_broadcast_f32", Linkage::Import, &sig)
                .expect("Failed to declare lirien_tensor_broadcast_f32");
            let local_func = ctx.module.declare_func_in_func(func_id, ctx.builder.func);

            let src_rank_val = ctx.builder.ins().iconst(types::I64, src_rank as i64);
            let target_rank_val = ctx.builder.ins().iconst(types::I64, target_rank as i64);

            let call = ctx.builder.ins().call(
                local_func,
                &[
                    src_ptr,
                    src_dims_ptr,
                    src_rank_val,
                    target_dims_ptr,
                    target_rank_val,
                ],
            );
            let res_ptr = ctx.builder.inst_results(call)[0];

            ctx.values.insert(*dest, res_ptr);
            ctx.tensor_dims.insert(*dest, target_dim_vals);
        }
        InstructionKind::TensorFused(dest, inputs, expr) => {
            let first_tensor_input = inputs
                .iter()
                .find(|&&in_val| ctx.ssa_func.get_type(in_val).is_tensor())
                .expect("TensorFused must have at least one tensor input");
            let dims = ctx
                .tensor_dims
                .get(first_tensor_input)
                .expect("Tensor dimensions not found")
                .clone();

            let mut total_size = dims[0];
            for &dim in dims.iter().skip(1) {
                total_size = ctx.builder.ins().imul(total_size, dim);
            }

            let mut sig = ctx.module.make_signature();
            sig.params.push(AbiParam::new(types::I64)); // size
            sig.returns.push(AbiParam::new(types::I64)); // ptr
            let callee = ctx
                .module
                .declare_function("malloc", Linkage::Import, &sig)
                .unwrap();
            let local_callee = ctx.module.declare_func_in_func(callee, ctx.builder.func);
            let four = ctx.builder.ins().iconst(types::I64, 4);
            let size_in_bytes = ctx.builder.ins().imul(total_size, four);
            let call = ctx.builder.ins().call(local_callee, &[size_in_bytes]);
            let dest_ptr = ctx.builder.inst_results(call)[0];

            let loop_header = ctx.builder.create_block();
            let loop_body_block = ctx.builder.create_block();
            let loop_exit = ctx.builder.create_block();

            ctx.builder.append_block_param(loop_header, types::I64);

            let zero = ctx.builder.ins().iconst(types::I64, 0);
            ctx.builder.ins().jump(loop_header, &[zero]);

            ctx.builder.switch_to_block(loop_header);
            let idx = ctx.builder.block_params(loop_header)[0];
            let cmp = ctx
                .builder
                .ins()
                .icmp(IntCC::SignedLessThan, idx, total_size);
            ctx.builder
                .ins()
                .brif(cmp, loop_body_block, &[], loop_exit, &[]);

            ctx.builder.switch_to_block(loop_body_block);
            let res_val = eval_fused_expr(&mut ctx.builder, &ctx.values, idx, expr);
            let offset = ctx.builder.ins().imul_imm(idx, 4);
            let addr = ctx.builder.ins().iadd(dest_ptr, offset);
            ctx.builder.ins().store(MemFlags::new(), res_val, addr, 0);

            let one = ctx.builder.ins().iconst(types::I64, 1);
            let next_idx = ctx.builder.ins().iadd(idx, one);
            ctx.builder.ins().jump(loop_header, &[next_idx]);

            ctx.builder.switch_to_block(loop_exit);

            ctx.values.insert(*dest, dest_ptr);
            ctx.tensor_dims.insert(*dest, dims);
        }
        InstructionKind::TensorAdd(dest, lhs, rhs) => lower_tensor_arith(ctx, dest, lhs, rhs, 0)?,
        InstructionKind::TensorSub(dest, lhs, rhs) => lower_tensor_arith(ctx, dest, lhs, rhs, 1)?,
        InstructionKind::TensorMul(dest, lhs, rhs) => lower_tensor_arith(ctx, dest, lhs, rhs, 2)?,
        InstructionKind::TensorDiv(dest, lhs, rhs) => lower_tensor_arith(ctx, dest, lhs, rhs, 3)?,

        InstructionKind::TensorSum(dest, tensor) => lower_tensor_reduce(ctx, dest, tensor, 0)?,
        InstructionKind::TensorMax(dest, tensor) => lower_tensor_reduce(ctx, dest, tensor, 1)?,
        InstructionKind::TensorMin(dest, tensor) => lower_tensor_reduce(ctx, dest, tensor, 2)?,

        InstructionKind::TensorScalarAdd(dest, tensor, scalar) => {
            lower_tensor_scalar_arith(ctx, dest, tensor, scalar, 0)?
        }
        InstructionKind::TensorScalarSub(dest, tensor, scalar) => {
            lower_tensor_scalar_arith(ctx, dest, tensor, scalar, 1)?
        }
        InstructionKind::TensorScalarMul(dest, tensor, scalar) => {
            lower_tensor_scalar_arith(ctx, dest, tensor, scalar, 2)?
        }
        InstructionKind::TensorScalarDiv(dest, tensor, scalar) => {
            lower_tensor_scalar_arith(ctx, dest, tensor, scalar, 3)?
        }
        InstructionKind::ArrayLoad(dest, arr, idx) => {
            let arr_ptr = get_val(&ctx.values, arr);
            let idx_val = get_val(&ctx.values, idx);
            let dest_ty = ctx.ssa_func.get_type(*dest);
            let elem_size = match &ctx.ssa_func.get_type(*arr) {
                SsaType::Array(inner, _) => inner.size(&ctx.ssa_func.struct_layouts),
                _ => 8,
            };

            // Apply stride if the source is a strided slice
            let stride = ctx.array_strides.get(arr).copied();
            let scaled_idx = if let Some(step_val) = stride {
                ctx.builder.ins().imul(idx_val, step_val)
            } else {
                idx_val
            };

            let offset = ctx.builder.ins().imul_imm(scaled_idx, elem_size as i64);
            let addr = ctx.builder.ins().iadd(arr_ptr, offset);

            if dest_ty.is_composite() {
                ctx.values.insert(*dest, addr);
            } else {
                let cl_ty = translate_type(&dest_ty);
                let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), addr, 0);
                ctx.values.insert(*dest, res);
            }
        }
        InstructionKind::ArrayStore(dest, arr, idx, val, _ty) => {
            let arr_ptr = get_val(&ctx.values, arr);
            let idx_val = get_val(&ctx.values, idx);
            let elem_size = match &ctx.ssa_func.get_type(*arr) {
                SsaType::Array(inner, _) => inner.size(&ctx.ssa_func.struct_layouts),
                _ => 8,
            };

            let stride = ctx.array_strides.get(arr).copied();
            let scaled_idx = if let Some(step_val) = stride {
                ctx.builder.ins().imul(idx_val, step_val)
            } else {
                idx_val
            };

            let offset = ctx.builder.ins().imul_imm(scaled_idx, elem_size as i64);
            let addr = ctx.builder.ins().iadd(arr_ptr, offset);

            super::store_to_memory(ctx, *val, addr, 0);
            ctx.values.insert(*dest, arr_ptr);
        }
        InstructionKind::ArraySlice(dest, arr, start_idx, step) => {
            let arr_ptr = get_val(&ctx.values, arr);
            let idx_val = get_val(&ctx.values, start_idx);
            let step_val = get_val(&ctx.values, step);
            let elem_size = match &ctx.ssa_func.get_type(*arr) {
                SsaType::Array(inner, _) | SsaType::Buffer(inner) => {
                    inner.size(&ctx.ssa_func.struct_layouts)
                }
                _ => 8,
            };
            let offset = ctx.builder.ins().imul_imm(idx_val, elem_size as i64);
            let addr = ctx.builder.ins().iadd(arr_ptr, offset);
            ctx.values.insert(*dest, addr);
            // Propagate stride from parent if it existed, then multiply by this slice's step
            let parent_stride = ctx.array_strides.get(arr).copied();
            let effective_stride = if let Some(ps) = parent_stride {
                ctx.builder.ins().imul(ps, step_val)
            } else {
                step_val
            };
            ctx.array_strides.insert(*dest, effective_stride);
        }
        InstructionKind::StructCreate(dest, struct_name, args) => {
            let dest_ty = ctx.ssa_func.get_type(*dest);
            if let SsaType::NamedTuple(_) = dest_ty {
                let mut field_vals = Vec::new();
                for arg in args {
                    field_vals.extend(super::get_all_cl_values(ctx, arg));
                }
                ctx.unpacked_values.insert(*dest, field_vals);
                return Ok(());
            }

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
                let f_ty = &fields[i].1;
                let f_align = f_ty.align(&ctx.ssa_func.struct_layouts);
                field_offset = (field_offset + f_align - 1) & !(f_align - 1);

                super::store_to_stack(ctx, *p_val, slot, field_offset as i32);
                field_offset += f_ty.size(&ctx.ssa_func.struct_layouts);
            }

            let dest_addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
            ctx.values.insert(*dest, dest_addr);
        }
        InstructionKind::StructLoad(dest, obj, offset) => {
            let obj_ty = ctx.ssa_func.get_type(*obj);
            if let SsaType::NamedTuple(_) = obj_ty {
                let mut current_offset = 0;
                let mut val_idx = 0;
                let dest_ty = ctx.ssa_func.get_type(*dest);
                let expected_count = super::get_flattened_types(ctx.ssa_func, &dest_ty).len();

                if let Some(start_idx) = super::get_field_info(
                    ctx.ssa_func,
                    &obj_ty,
                    *offset as i32,
                    expected_count,
                    &mut current_offset,
                    &mut val_idx,
                ) {
                    let flat_vals = ctx.unpacked_values.get(obj).unwrap();
                    let extracted = flat_vals[start_idx..start_idx + expected_count].to_vec();
                    if dest_ty.is_composite() {
                        ctx.unpacked_values.insert(*dest, extracted);
                    } else {
                        ctx.values.insert(*dest, extracted[0]);
                    }
                    return Ok(());
                }
                return Err(LoweringError::General(
                    format!(
                        "Field offset {} (count {}) not found in NamedTuple {:?}",
                        offset, expected_count, obj_ty
                    ),
                    None,
                ));
            }

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
        InstructionKind::StructSet(dest, obj, offset, val, _ty) => {
            let obj_ty = ctx.ssa_func.get_type(*obj);
            if let SsaType::NamedTuple(_) = obj_ty {
                let mut current_offset = 0;
                let mut val_idx = 0;
                let val_ty = ctx.ssa_func.get_type(*val);
                let expected_count = super::get_flattened_types(ctx.ssa_func, &val_ty).len();

                if let Some(start_idx) = super::get_field_info(
                    ctx.ssa_func,
                    &obj_ty,
                    *offset as i32,
                    expected_count,
                    &mut current_offset,
                    &mut val_idx,
                ) {
                    let mut new_flat_vals = ctx.unpacked_values.get(obj).unwrap().clone();
                    let val_flat = super::get_all_cl_values(ctx, val);
                    assert_eq!(expected_count, val_flat.len());
                    new_flat_vals[start_idx..(expected_count + start_idx)]
                        .copy_from_slice(&val_flat[..expected_count]);
                    ctx.unpacked_values.insert(*dest, new_flat_vals);
                    return Ok(());
                }
                return Err(LoweringError::General(
                    format!(
                        "Field offset {} (count {}) not found in NamedTuple {:?}",
                        offset, expected_count, obj_ty
                    ),
                    None,
                ));
            }

            let obj_ptr = get_val(&ctx.values, obj);
            super::store_to_memory(ctx, *val, obj_ptr, *offset as i32);
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
            let variants = ctx
                .ssa_func
                .enum_layouts
                .get(&enum_name)
                .expect("Unknown enum layout");

            let mut max_align = 1;
            for (_, v_ty) in variants {
                let align = v_ty.align(&ctx.ssa_func.struct_layouts);
                if align > max_align {
                    max_align = align;
                }
            }

            let mut offset = 1;
            offset = (offset + max_align - 1) & !(max_align - 1);

            let addr = ctx.builder.ins().iadd_imm(obj_ptr, offset as i64);
            let payload_ty = &variants[*tag_idx].1;
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
            let local_callee = ctx.module.declare_func_in_func(callee, ctx.builder.func);
            let size_val = ctx.builder.ins().iconst(types::I64, size as i64);
            let call = ctx.builder.ins().call(local_callee, &[size_val]);
            let res = ctx.builder.inst_results(call)[0];
            ctx.values.insert(*dest, res);
        }
        InstructionKind::PointerLoad(dest, ptr) => {
            let ptr_val = get_val(&ctx.values, ptr);
            let dest_ty = ctx.ssa_func.get_type(*dest);
            if dest_ty.is_composite() {
                // In Lirien SSA, composite types (Structs, Tuples) are represented as pointers
                // to their memory location. Therefore, loading a composite from a pointer
                // simply returns the pointer itself.
                ctx.values.insert(*dest, ptr_val);
            } else {
                let cl_ty = translate_type(&dest_ty);
                let res = ctx.builder.ins().load(cl_ty, MemFlags::new(), ptr_val, 0);
                ctx.values.insert(*dest, res);
            }
        }
        InstructionKind::PointerStore(ptr, val) => {
            let ptr_val = get_val(&ctx.values, ptr);
            super::store_to_memory(ctx, *val, ptr_val, 0);
        }
        _ => {
            return Err(LoweringError::InstructionNotSupported(
                format!("{:?}", kind),
                None,
            ))
        }
    }
    Ok(())
}

fn eval_fused_expr(
    builder: &mut cranelift::frontend::FunctionBuilder,
    values: &std::collections::HashMap<lirien_ir::ir::Value, Value>,
    idx: Value,
    expr: &lirien_ir::ir::FusedExpr,
) -> Value {
    match expr {
        lirien_ir::ir::FusedExpr::Input(val) => {
            let t_ptr = values.get(val).copied().unwrap();
            let offset = builder.ins().imul_imm(idx, 4);
            let addr = builder.ins().iadd(t_ptr, offset);
            builder.ins().load(types::F32, MemFlags::new(), addr, 0)
        }
        lirien_ir::ir::FusedExpr::Scalar(val) => values.get(val).copied().unwrap(),
        lirien_ir::ir::FusedExpr::Add(l, r) => {
            let lv = eval_fused_expr(builder, values, idx, l);
            let rv = eval_fused_expr(builder, values, idx, r);
            builder.ins().fadd(lv, rv)
        }
        lirien_ir::ir::FusedExpr::Sub(l, r) => {
            let lv = eval_fused_expr(builder, values, idx, l);
            let rv = eval_fused_expr(builder, values, idx, r);
            builder.ins().fsub(lv, rv)
        }
        lirien_ir::ir::FusedExpr::Mul(l, r) => {
            let lv = eval_fused_expr(builder, values, idx, l);
            let rv = eval_fused_expr(builder, values, idx, r);
            builder.ins().fmul(lv, rv)
        }
        lirien_ir::ir::FusedExpr::Div(l, r) => {
            let lv = eval_fused_expr(builder, values, idx, l);
            let rv = eval_fused_expr(builder, values, idx, r);
            builder.ins().fdiv(lv, rv)
        }
    }
}
