use super::{translate_type, CodegenContext};
use cranelift::codegen::ir::StackSlot;
use cranelift::prelude::*;
use cranelift_module::Module;
use lila_ir::ir::{BlockId as SsaBlockId, Instruction, InstructionKind, Value as SsaValue, Function as SsaFunction};

pub mod arithmetic;
pub mod control_flow;
pub mod higher_order;
pub mod intrinsics;
pub mod memory;
pub mod tuples;

pub fn copy_memory(builder: &mut FunctionBuilder, src_ptr: Value, dest_ptr: Value, size: usize) {
    let mut curr_offset = 0;
    while curr_offset < size {
        let bytes_left = size - curr_offset;
        let (cl_ty, chunk_size) = if bytes_left >= 8 {
            (types::I64, 8)
        } else if bytes_left >= 4 {
            (types::I32, 4)
        } else if bytes_left >= 2 {
            (types::I16, 2)
        } else {
            (types::I8, 1)
        };

        let val = builder
            .ins()
            .load(cl_ty, MemFlags::new(), src_ptr, curr_offset as i32);
        builder
            .ins()
            .store(MemFlags::new(), val, dest_ptr, curr_offset as i32);
        curr_offset += chunk_size;
    }
}

pub fn copy_to_stack(
    builder: &mut FunctionBuilder,
    src_ptr: Value,
    slot: StackSlot,
    slot_offset: i32,
    size: usize,
) {
    let mut curr_offset = 0;
    while curr_offset < size {
        let bytes_left = size - curr_offset;
        let (cl_ty, chunk_size) = if bytes_left >= 8 {
            (types::I64, 8)
        } else if bytes_left >= 4 {
            (types::I32, 4)
        } else if bytes_left >= 2 {
            (types::I16, 2)
        } else {
            (types::I8, 1)
        };

        let val = builder
            .ins()
            .load(cl_ty, MemFlags::new(), src_ptr, curr_offset as i32);
        builder
            .ins()
            .stack_store(val, slot, slot_offset + curr_offset as i32);
        curr_offset += chunk_size;
    }
}

pub fn get_flattened_types(
    ssa_func: &SsaFunction,
    ty: &lila_ir::ir::Type,
) -> Vec<types::Type> {
    match ty {
        lila_ir::ir::Type::NamedTuple(ref name) => {
            let fields = ssa_func.struct_layouts.get(name).unwrap();
            let mut res = Vec::new();
            for (_, f_ty) in fields {
                res.extend(get_flattened_types(ssa_func, f_ty));
            }
            res
        }
        lila_ir::ir::Type::Tuple(ref types) => {
            let mut res = Vec::new();
            for t in types {
                res.extend(get_flattened_types(ssa_func, t));
            }
            res
        }
        lila_ir::ir::Type::Buffer(_) => vec![types::I64, types::I64],
        lila_ir::ir::Type::Tensor(_, ref dims) => {
            let mut res = vec![types::I64];
            for _ in 0..dims.len() {
                res.push(types::I64);
            }
            res
        }
        _ => vec![super::translate_type(ty)],
    }
}

pub fn get_field_info(
    ssa_func: &SsaFunction,
    ty: &lila_ir::ir::Type,
    target_offset: i32,
    expected_count: usize,
    current_offset: &mut i32,
    val_idx: &mut usize,
) -> Option<usize> {
    let align = ty.align(&ssa_func.struct_layouts) as i32;
    *current_offset = (*current_offset + align - 1) & !(align - 1);

    if *current_offset == target_offset {
        let count = get_flattened_types(ssa_func, ty).len();
        if count == expected_count {
            return Some(*val_idx);
        }
    }

    match ty {
        lila_ir::ir::Type::NamedTuple(ref name) => {
            let fields = ssa_func.struct_layouts.get(name).unwrap();
            for (_, f_ty) in fields {
                if let Some(res) = get_field_info(ssa_func, f_ty, target_offset, expected_count, current_offset, val_idx) {
                    return Some(res);
                }
            }
        }
        lila_ir::ir::Type::Tuple(ref types) => {
            for t in types {
                if let Some(res) = get_field_info(ssa_func, t, target_offset, expected_count, current_offset, val_idx) {
                    return Some(res);
                }
            }
        }
        lila_ir::ir::Type::Buffer(_) => {
            *current_offset += 16;
            *val_idx += 2;
        }
        lila_ir::ir::Type::Tensor(_, dims) => {
            *current_offset += 8 + 8 * dims.len() as i32;
            *val_idx += 1 + dims.len();
        }
        _ => {
            *current_offset += ty.size(&ssa_func.struct_layouts) as i32;
            *val_idx += 1;
        }
    }
    None
}

pub fn store_to_stack_recursive<M: Module>(
    ctx: &mut CodegenContext<M>,
    ty: &lila_ir::ir::Type,
    flat_vals: &[Value],
    slot: StackSlot,
    current_offset: &mut i32,
    val_idx: &mut usize,
) {
    let align = ty.align(&ctx.ssa_func.struct_layouts) as i32;
    *current_offset = (*current_offset + align - 1) & !(align - 1);
    let start_offset = *current_offset;

    match ty {
        lila_ir::ir::Type::NamedTuple(ref name) => {
            let fields = ctx.ssa_func.struct_layouts.get(name).unwrap();
            for (_, f_ty) in fields {
                store_to_stack_recursive(ctx, f_ty, flat_vals, slot, current_offset, val_idx);
            }
        }
        lila_ir::ir::Type::Tuple(ref types) => {
            for t in types {
                store_to_stack_recursive(ctx, t, flat_vals, slot, current_offset, val_idx);
            }
        }
        lila_ir::ir::Type::Buffer(_) => {
            let ptr_val = flat_vals[*val_idx];
            let len_val = flat_vals[*val_idx + 1];
            ctx.builder.ins().stack_store(ptr_val, slot, start_offset);
            ctx.builder.ins().stack_store(len_val, slot, start_offset + 8);
            *current_offset += 16;
            *val_idx += 2;
        }
        lila_ir::ir::Type::Tensor(_, dims) => {
            let ptr_val = flat_vals[*val_idx];
            ctx.builder.ins().stack_store(ptr_val, slot, start_offset);
            for i in 0..dims.len() {
                let dim_val = flat_vals[*val_idx + 1 + i];
                ctx.builder.ins().stack_store(dim_val, slot, start_offset + 8 + 8 * i as i32);
            }
            *current_offset += 8 + 8 * dims.len() as i32;
            *val_idx += 1 + dims.len();
        }
        _ => {
            let val = flat_vals[*val_idx];
            let size = ty.size(&ctx.ssa_func.struct_layouts) as i32;
            if ty.is_composite() {
                copy_to_stack(&mut ctx.builder, val, slot, start_offset, size as usize);
            } else {
                ctx.builder.ins().stack_store(val, slot, start_offset);
            }
            *current_offset += size;
            *val_idx += 1;
        }
    }
}

pub fn store_to_stack<M: Module>(
    ctx: &mut CodegenContext<M>,
    src: SsaValue,
    slot: StackSlot,
    slot_offset: i32,
) {
    let ty = ctx.ssa_func.get_type(src);
    if let Some(flat_vals) = ctx.unpacked_values.get(&src).cloned() {
        let mut current_offset = slot_offset;
        let mut val_idx = 0;
        store_to_stack_recursive(ctx, &ty, &flat_vals, slot, &mut current_offset, &mut val_idx);
    } else {
        let val = get_val(&ctx.values, &src);
        let size = ty.size(&ctx.ssa_func.struct_layouts) as i32;
        if ty.is_composite() {
            copy_to_stack(&mut ctx.builder, val, slot, slot_offset, size as usize);
        } else {
            ctx.builder.ins().stack_store(val, slot, slot_offset);
        }
    }
}

pub fn store_to_memory_recursive<M: Module>(
    ctx: &mut CodegenContext<M>,
    ty: &lila_ir::ir::Type,
    flat_vals: &[Value],
    dest_ptr: Value,
    current_offset: &mut i32,
    val_idx: &mut usize,
) {
    let align = ty.align(&ctx.ssa_func.struct_layouts) as i32;
    *current_offset = (*current_offset + align - 1) & !(align - 1);
    let start_offset = *current_offset;

    match ty {
        lila_ir::ir::Type::NamedTuple(ref name) => {
            let fields = ctx.ssa_func.struct_layouts.get(name).unwrap();
            for (_, f_ty) in fields {
                store_to_memory_recursive(ctx, f_ty, flat_vals, dest_ptr, current_offset, val_idx);
            }
        }
        lila_ir::ir::Type::Tuple(ref types) => {
            for t in types {
                store_to_memory_recursive(ctx, t, flat_vals, dest_ptr, current_offset, val_idx);
            }
        }
        lila_ir::ir::Type::Buffer(_) => {
            let ptr_val = flat_vals[*val_idx];
            let len_val = flat_vals[*val_idx + 1];
            ctx.builder.ins().store(MemFlags::new(), ptr_val, dest_ptr, start_offset);
            ctx.builder.ins().store(MemFlags::new(), len_val, dest_ptr, start_offset + 8);
            *current_offset += 16;
            *val_idx += 2;
        }
        lila_ir::ir::Type::Tensor(_, dims) => {
            let ptr_val = flat_vals[*val_idx];
            ctx.builder.ins().store(MemFlags::new(), ptr_val, dest_ptr, start_offset);
            for i in 0..dims.len() {
                let dim_val = flat_vals[*val_idx + 1 + i];
                ctx.builder.ins().store(MemFlags::new(), dim_val, dest_ptr, start_offset + 8 + 8 * i as i32);
            }
            *current_offset += 8 + 8 * dims.len() as i32;
            *val_idx += 1 + dims.len();
        }
        _ => {
            let val = flat_vals[*val_idx];
            let size = ty.size(&ctx.ssa_func.struct_layouts) as i32;
            if ty.is_composite() {
                let size_val = ctx.builder.ins().iconst(types::I64, size as i64);
                let dest_addr = ctx.builder.ins().iadd_imm(dest_ptr, start_offset as i64);
                ctx.builder.call_memcpy(ctx.module.target_config(), dest_addr, val, size_val);
            } else {
                ctx.builder.ins().store(MemFlags::new(), val, dest_ptr, start_offset);
            }
            *current_offset += size;
            *val_idx += 1;
        }
    }
}

pub fn store_to_memory<M: Module>(
    ctx: &mut CodegenContext<M>,
    src: SsaValue,
    dest_ptr: Value,
    dest_offset: i32,
) {
    let ty = ctx.ssa_func.get_type(src);
    if let Some(flat_vals) = ctx.unpacked_values.get(&src).cloned() {
        let mut current_offset = dest_offset;
        let mut val_idx = 0;
        store_to_memory_recursive(ctx, &ty, &flat_vals, dest_ptr, &mut current_offset, &mut val_idx);
    } else {
        let val = get_val(&ctx.values, &src);
        let size = ty.size(&ctx.ssa_func.struct_layouts) as i32;
        if ty.is_composite() {
            let size_val = ctx.builder.ins().iconst(types::I64, size as i64);
            let dest_addr = ctx.builder.ins().iadd_imm(dest_ptr, dest_offset as i64);
            ctx.builder.call_memcpy(ctx.module.target_config(), dest_addr, val, size_val);
        } else {
            ctx.builder.ins().store(MemFlags::new(), val, dest_ptr, dest_offset);
        }
    }
}

pub fn build_cranelift_signature(
    ssa_func: &SsaFunction,
    arg_types: &[lila_ir::ir::Type],
    ret_ty: &lila_ir::ir::Type,
    is_closure: bool,
    module: &impl Module,
) -> (Signature, bool, bool) {
    let mut sig = module.make_signature();
    let mut is_sret = false;
    let mut is_register_composite_ret = false;

    // 1. Handle Return Type
    if matches!(ret_ty, lila_ir::ir::Type::NamedTuple(_) | lila_ir::ir::Type::Tuple(_)) {
        let cl_types = get_flattened_types(ssa_func, ret_ty);
        if cl_types.len() <= 2 {
            for cl_ty in cl_types {
                sig.returns.push(AbiParam::new(cl_ty));
            }
            is_register_composite_ret = true;
        } else {
            sig.params.push(AbiParam::new(types::I64)); // SRet pointer
            is_sret = true;
        }
    } else if ret_ty.is_simd() {
        sig.params.push(AbiParam::new(types::I64)); // SRet pointer
        is_sret = true;
    } else if *ret_ty != lila_ir::ir::Type::Unknown {
        sig.returns.push(AbiParam::new(translate_type(ret_ty)));
    }

    // 2. Handle Arguments
    if is_closure {
        sig.params.push(AbiParam::new(types::I64)); // context pointer
    }

    for arg_ty in arg_types {
        match arg_ty {
            lila_ir::ir::Type::NamedTuple(_) | lila_ir::ir::Type::Tuple(_) => {
                let cl_types = get_flattened_types(ssa_func, arg_ty);
                for cl_ty in cl_types {
                    sig.params.push(AbiParam::new(cl_ty));
                }
            }
            lila_ir::ir::Type::Buffer(_) => {
                sig.params.push(AbiParam::new(types::I64)); // Ptr
                sig.params.push(AbiParam::new(types::I64)); // Len
            }
            lila_ir::ir::Type::Tensor(_, dims) => {
                sig.params.push(AbiParam::new(types::I64)); // Ptr
                for _ in 0..dims.len() {
                    sig.params.push(AbiParam::new(types::I64)); // Dim length
                }
            }
            _ if arg_ty.is_simd() => {
                sig.params.push(AbiParam::new(types::I64)); // Pass by pointer
            }
            _ => {
                sig.params.push(AbiParam::new(translate_type(arg_ty)));
            }
        }
    }

    (sig, is_sret, is_register_composite_ret)
}

pub fn lower_instruction<M: Module>(
    ctx: &mut CodegenContext<M>,
    inst: &Instruction,
    current_ssa_block: SsaBlockId,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::Phi(_, _) => Ok(()), // Handled in Pass 1

        InstructionKind::ConstInt(dest, val) => {
            let ty = ctx.ssa_func.get_type(*dest);
            let cl_ty = super::translate_type(&ty);
            let res = ctx.builder.ins().iconst(cl_ty, *val);
            ctx.values.insert(*dest, res);
            Ok(())
        }
        InstructionKind::ConstFloat(dest, val) => {
            let ty = ctx.ssa_func.get_type(*dest);
            let res = if ty.is_float32() {
                ctx.builder.ins().f32const(*val as f32)
            } else {
                ctx.builder.ins().f64const(*val)
            };
            ctx.values.insert(*dest, res);
            Ok(())
        }
        InstructionKind::Assign(dest, src) => {
            let ty = ctx.ssa_func.get_type(*dest);
            if let lila_ir::ir::Type::NamedTuple(_) = ty {
                let s_vals = ctx.unpacked_values.get(src).unwrap().clone();
                ctx.unpacked_values.insert(*dest, s_vals);
            } else {
                let s = get_val(&ctx.values, src);
                ctx.values.insert(*dest, s);
                // Also handle Buffer/Tensor metadata if needed
                if let Some(len) = ctx.buffer_lengths.get(src) {
                    let l = *len;
                    ctx.buffer_lengths.insert(*dest, l);
                }
                if let Some(dims) = ctx.tensor_dims.get(src) {
                    let d = dims.clone();
                    ctx.tensor_dims.insert(*dest, d);
                }
            }
            Ok(())
        }

        InstructionKind::Add(_, _, _)
        | InstructionKind::Sub(_, _, _)
        | InstructionKind::Mul(_, _, _)
        | InstructionKind::SDiv(_, _, _)
        | InstructionKind::UDiv(_, _, _)
        | InstructionKind::SRem(_, _, _)
        | InstructionKind::URem(_, _, _)
        | InstructionKind::And(_, _, _)
        | InstructionKind::Or(_, _, _)
        | InstructionKind::Xor(_, _, _)
        | InstructionKind::Shl(_, _, _)
        | InstructionKind::LShr(_, _, _)
        | InstructionKind::AShr(_, _, _)
        | InstructionKind::Not(_, _)
        | InstructionKind::FAdd(_, _, _)
        | InstructionKind::FSub(_, _, _)
        | InstructionKind::FMul(_, _, _)
        | InstructionKind::FDiv(_, _, _)
        | InstructionKind::FSqrt(_, _)
        | InstructionKind::Abs(_, _)
        | InstructionKind::Neg(_, _)
        | InstructionKind::Min(_, _, _)
        | InstructionKind::Max(_, _, _)
        | InstructionKind::Avg(_, _, _)
        | InstructionKind::SIMDSplat(_, _)
        | InstructionKind::SIMDExtractLane(_, _, _)
        | InstructionKind::SIMDInsertLane(_, _, _, _)
        | InstructionKind::Eq(_, _, _)
        | InstructionKind::Ne(_, _, _)
        | InstructionKind::SLt(_, _, _)
        | InstructionKind::SLe(_, _, _)
        | InstructionKind::SGt(_, _, _)
        | InstructionKind::SGe(_, _, _)
        | InstructionKind::ULt(_, _, _)
        | InstructionKind::ULe(_, _, _)
        | InstructionKind::UGt(_, _, _)
        | InstructionKind::UGe(_, _, _)
        | InstructionKind::FLt(_, _, _)
        | InstructionKind::FLe(_, _, _)
        | InstructionKind::FGt(_, _, _)
        | InstructionKind::FGe(_, _, _)
        | InstructionKind::IToF(_, _, _)
        | InstructionKind::FToI(_, _, _)
        | InstructionKind::FConv(_, _, _) => arithmetic::lower(ctx, &inst.kind),

        InstructionKind::FSin(dest, src) => intrinsics::lower(ctx, *dest, "sin", &[*src]),
        InstructionKind::FCos(dest, src) => intrinsics::lower(ctx, *dest, "cos", &[*src]),
        InstructionKind::FPow(dest, lhs, rhs) => {
            intrinsics::lower(ctx, *dest, "pow", &[*lhs, *rhs])
        }

        InstructionKind::Jump(_)
        | InstructionKind::Branch(_, _, _)
        | InstructionKind::Match(_, _, _, _)
        | InstructionKind::Return(_) => control_flow::lower(ctx, &inst.kind, current_ssa_block),

        InstructionKind::ArrayLoad(_, _, _)
        | InstructionKind::ArrayStore(_, _, _, _, _)
        | InstructionKind::BufferLoad(_, _, _)
        | InstructionKind::BufferStore(_, _, _, _, _)
        | InstructionKind::TensorLoad(_, _, _)
        | InstructionKind::TensorStore(_, _, _, _)
        | InstructionKind::TensorDim(_, _, _)
        | InstructionKind::TensorBroadcast(_, _, _)
        | InstructionKind::TensorAdd(_, _, _)

        | InstructionKind::TensorSub(_, _, _)
        | InstructionKind::TensorMul(_, _, _)
        | InstructionKind::TensorDiv(_, _, _)
        | InstructionKind::TensorScalarAdd(_, _, _)
        | InstructionKind::TensorScalarSub(_, _, _)
        | InstructionKind::TensorScalarMul(_, _, _)
        | InstructionKind::TensorScalarDiv(_, _, _)
        | InstructionKind::TensorSum(_, _)
        | InstructionKind::TensorMax(_, _)
        | InstructionKind::TensorMin(_, _)
        | InstructionKind::BufferLen(_, _)
        | InstructionKind::StructCreate(_, _, _)
        | InstructionKind::StructLoad(_, _, _)
        | InstructionKind::StructOffset(_, _, _)
        | InstructionKind::StructSet(_, _, _, _, _)
        | InstructionKind::EnumCreate(_, _, _, _)
        | InstructionKind::EnumGetTag(_, _)
        | InstructionKind::EnumIsVariant(_, _, _)
        | InstructionKind::EnumAsVariant(_, _, _)
        | InstructionKind::EnumExtract(_, _, _)
        | InstructionKind::Alloc(_, _)
        | InstructionKind::PointerLoad(_, _)
        | InstructionKind::PointerStore(_, _) => memory::lower(ctx, &inst.kind),


        InstructionKind::TupleCreate(_, _) | InstructionKind::TupleExtract(_, _, _) => {
            tuples::lower(ctx, &inst.kind)
        }

        InstructionKind::Call(dest, func, args) => intrinsics::lower(ctx, *dest, func, args),
        InstructionKind::IndirectCall(dest, fn_ptr, args) => {
            higher_order::lower(ctx, *dest, *fn_ptr, args)
        }
        InstructionKind::Lambda(dest, name, captures) => {
            higher_order::lower_lambda(ctx, *dest, name, captures)
        }
        InstructionKind::ParallelFor(index_var, start, ..) => {
            let cl_start = get_val(&ctx.values, start);
            ctx.values.insert(*index_var, cl_start);
            Ok(())
        }
        InstructionKind::MatMult(dest, lhs, rhs) => {
            let a_ptr = get_val(&ctx.values, lhs);
            let b_ptr = get_val(&ctx.values, rhs);
            
            let l_dims = ctx.tensor_dims.get(lhs).expect("LHS tensor dims not found");
            let r_dims = ctx.tensor_dims.get(rhs).expect("RHS tensor dims not found");
            
            let m = l_dims[0];
            let n = l_dims[1];
            let k = r_dims[1];

            // Declare lila_matmul_alloc_f32 in Cranelift
            let mut sig = ctx.module.make_signature();
            sig.params.push(cranelift::prelude::AbiParam::new(types::I64)); // a
            sig.params.push(cranelift::prelude::AbiParam::new(types::I64)); // b
            sig.params.push(cranelift::prelude::AbiParam::new(types::I64)); // m
            sig.params.push(cranelift::prelude::AbiParam::new(types::I64)); // n
            sig.params.push(cranelift::prelude::AbiParam::new(types::I64)); // k
            sig.returns.push(cranelift::prelude::AbiParam::new(types::I64)); // c

            let callee = ctx
                .module
                .declare_function("lila_matmul_alloc_f32", cranelift_module::Linkage::Import, &sig)
                .map_err(|e| e.to_string())?;
            let local_callee = ctx.module.declare_func_in_func(callee, ctx.builder.func);
            
            let call = ctx.builder.ins().call(local_callee, &[a_ptr, b_ptr, m, n, k]);
            let res_ptr = ctx.builder.inst_results(call)[0];
            
            ctx.values.insert(*dest, res_ptr);
            
            // Register dimensions for the returned tensor
            ctx.tensor_dims.insert(*dest, vec![m, k]);
            Ok(())
        }
        InstructionKind::Nop() => Ok(()),
    }
}

pub fn get_val(values: &std::collections::HashMap<SsaValue, Value>, val: &SsaValue) -> Value {
    *values
        .get(val)
        .unwrap_or_else(|| panic!("Value v{} not found", val.0))
}

pub fn get_all_cl_values<M: Module>(ctx: &CodegenContext<M>, val: &SsaValue) -> Vec<Value> {
    let ty = ctx.ssa_func.get_type(*val);
    match ty {
        lila_ir::ir::Type::NamedTuple(_) | lila_ir::ir::Type::Tuple(_) => ctx
            .unpacked_values
            .get(val)
            .cloned()
            .unwrap_or_else(|| vec![get_val(&ctx.values, val)]),
        lila_ir::ir::Type::Buffer(_) => vec![
            get_val(&ctx.values, val),
            *ctx.buffer_lengths
                .get(val)
                .unwrap_or_else(|| panic!("Length for v{} not found", val.0)),
        ],
        lila_ir::ir::Type::Tensor(_, ref _dims) => {
            let mut res = vec![get_val(&ctx.values, val)];
            res.extend(
                ctx.tensor_dims
                    .get(val)
                    .unwrap_or_else(|| panic!("Dims for v{} not found", val.0)),
            );
            res
        }
        _ => vec![get_val(&ctx.values, val)],
    }
}

pub fn get_len(lengths: &std::collections::HashMap<SsaValue, Value>, val: &SsaValue) -> Value {
    *lengths
        .get(val)
        .unwrap_or_else(|| panic!("Length for v{} not found", val.0))
}
