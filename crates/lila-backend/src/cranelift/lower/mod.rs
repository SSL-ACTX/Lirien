use super::{translate_type, CodegenContext};
use cranelift::codegen::ir::StackSlot;
use cranelift::prelude::*;
use cranelift_module::Module;
use lila_ir::ir::{BlockId as SsaBlockId, Instruction, InstructionKind, Value as SsaValue};

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
            let res = ctx.builder.ins().f64const(*val);
            ctx.values.insert(*dest, res);
            Ok(())
        }
        InstructionKind::Assign(dest, src) => {
            let s = get_val(&ctx.values, src);
            ctx.values.insert(*dest, s);
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

pub fn get_len(lengths: &std::collections::HashMap<SsaValue, Value>, val: &SsaValue) -> Value {
    *lengths
        .get(val)
        .unwrap_or_else(|| panic!("Length for v{} not found", val.0))
}
