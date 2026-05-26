use super::{translate_type, CodegenContext};
use crate::ssa::ir::{BlockId as SsaBlockId, Instruction, InstructionKind, Value as SsaValue};
use cranelift::codegen::ir::StackSlot;
use cranelift::prelude::*;
use cranelift_module::Module;

pub mod arithmetic;
pub mod control_flow;
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
            let res = ctx.builder.ins().iconst(types::I64, *val);
            ctx.values.insert(*dest, res);
            Ok(())
        }
        InstructionKind::ConstFloat(dest, val) => {
            let res = ctx.builder.ins().f64const(*val);
            ctx.values.insert(*dest, res);
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
        | InstructionKind::FToI(_, _, _) => arithmetic::lower(ctx, &inst.kind),

        InstructionKind::FSin(dest, src) => intrinsics::lower(ctx, *dest, "sin", &[*src]),
        InstructionKind::FCos(dest, src) => intrinsics::lower(ctx, *dest, "cos", &[*src]),
        InstructionKind::FPow(dest, lhs, rhs) => {
            intrinsics::lower(ctx, *dest, "pow", &[*lhs, *rhs])
        }

        InstructionKind::Jump(_)
        | InstructionKind::Branch(_, _, _)
        | InstructionKind::Return(_) => control_flow::lower(ctx, &inst.kind, current_ssa_block),

        InstructionKind::ArrayLoad(_, _, _)
        | InstructionKind::ArrayStore(_, _, _, _, _)
        | InstructionKind::BufferLoad(_, _, _)
        | InstructionKind::BufferStore(_, _, _, _, _)
        | InstructionKind::BufferLen(_, _)
        | InstructionKind::StructCreate(_, _, _)
        | InstructionKind::StructLoad(_, _, _)
        | InstructionKind::StructOffset(_, _, _)
        | InstructionKind::StructSet(_, _, _, _, _)
        | InstructionKind::Reference(_, _)
        | InstructionKind::MutReference(_, _)
        | InstructionKind::EnumCreate(_, _, _, _)
        | InstructionKind::EnumIsVariant(_, _, _)
        | InstructionKind::EnumExtract(_, _, _) => memory::lower(ctx, &inst.kind),

        InstructionKind::TupleCreate(_, _) | InstructionKind::TupleExtract(_, _, _) => {
            tuples::lower(ctx, &inst.kind)
        }

        InstructionKind::Call(dest, func, args) => intrinsics::lower(ctx, *dest, func, args),
        InstructionKind::Nop => Ok(()),
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
