use super::super::ir::{Function, Instruction, InstructionKind, Value};
use std::collections::HashSet;

pub fn eliminate_dead_code(func: &mut Function) {
    let mut used_values = HashSet::new();

    let mut worklist: Vec<Value> = Vec::new();

    // 1. Identify all values that are "required" (side effects or return values)
    for block in &func.blocks {
        for inst in &block.instructions {
            if has_side_effects(inst) {
                for operand in get_operands(inst) {
                    worklist.push(operand);
                }
            }
        }
    }

    // 2. Propagate "liveness" back through the dependency graph
    let mut visited = HashSet::new();
    while let Some(val) = worklist.pop() {
        if !visited.insert(val) {
            continue;
        }
        used_values.insert(val);

        if let Some(inst) = find_def(func, val) {
            for operand in get_operands(inst) {
                worklist.push(operand);
            }
        }
    }

    // 3. Remove instructions that don't produce used values and have no side effects
    for block in &mut func.blocks {
        block.instructions.retain(|inst| {
            if has_side_effects(inst) {
                return true;
            }
            if let Some(def) = get_def(inst) {
                return used_values.contains(&def);
            }
            // Instructions that don't define a value and have no side effects are useless
            false
        });
    }
}

fn find_def(func: &Function, val: Value) -> Option<&Instruction> {
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(d) = get_def(inst) {
                if d == val {
                    return Some(inst);
                }
            }
        }
    }
    None
}

/// DETECTOR: This function MUST be exhaustive.
/// If you add a new InstructionKind, the compiler will "shout" at you to update this.
fn get_def(inst: &Instruction) -> Option<Value> {
    match &inst.kind {
        InstructionKind::Add(d, _, _)
        | InstructionKind::Sub(d, _, _)
        | InstructionKind::Mul(d, _, _)
        | InstructionKind::SDiv(d, _, _)
        | InstructionKind::UDiv(d, _, _)
        | InstructionKind::SRem(d, _, _)
        | InstructionKind::URem(d, _, _)
        | InstructionKind::And(d, _, _)
        | InstructionKind::Or(d, _, _)
        | InstructionKind::Xor(d, _, _)
        | InstructionKind::Shl(d, _, _)
        | InstructionKind::LShr(d, _, _)
        | InstructionKind::AShr(d, _, _)
        | InstructionKind::Not(d, _)
        | InstructionKind::FAdd(d, _, _)
        | InstructionKind::FSub(d, _, _)
        | InstructionKind::FMul(d, _, _)
        | InstructionKind::FDiv(d, _, _)
        | InstructionKind::FSqrt(d, _)
        | InstructionKind::FSin(d, _)
        | InstructionKind::FCos(d, _)
        | InstructionKind::FPow(d, _, _)
        | InstructionKind::Eq(d, _, _)
        | InstructionKind::Ne(d, _, _)
        | InstructionKind::SLt(d, _, _)
        | InstructionKind::SLe(d, _, _)
        | InstructionKind::SGt(d, _, _)
        | InstructionKind::SGe(d, _, _)
        | InstructionKind::ULt(d, _, _)
        | InstructionKind::ULe(d, _, _)
        | InstructionKind::UGt(d, _, _)
        | InstructionKind::UGe(d, _, _)
        | InstructionKind::FLt(d, _, _)
        | InstructionKind::FLe(d, _, _)
        | InstructionKind::FGt(d, _, _)
        | InstructionKind::FGe(d, _, _)
        | InstructionKind::IToF(d, _, _)
        | InstructionKind::FToI(d, _, _)
        | InstructionKind::ConstInt(d, _)
        | InstructionKind::ConstFloat(d, _)
        | InstructionKind::Phi(d, _)
        | InstructionKind::Call(d, _, _)
        | InstructionKind::ArrayLoad(d, _, _)
        | InstructionKind::BufferLoad(d, _, _)
        | InstructionKind::BufferLen(d, _)
        | InstructionKind::StructCreate(d, _, _)
        | InstructionKind::StructLoad(d, _, _)
        | InstructionKind::StructOffset(d, _, _)
        | InstructionKind::EnumCreate(d, _, _, _)
        | InstructionKind::EnumIsVariant(d, _, _)
        | InstructionKind::EnumGetTag(d, _)
        | InstructionKind::EnumExtract(d, _, _)
        | InstructionKind::TupleCreate(d, _)
        | InstructionKind::TupleExtract(d, _, _)
        | InstructionKind::Alloc(d, _)
        | InstructionKind::PointerLoad(d, _)
        | InstructionKind::Lambda(d, _, _)
        | InstructionKind::IndirectCall(d, _, _) => Some(*d),

        // Instructions that NEVER define a value
        InstructionKind::Jump(_)
        | InstructionKind::Branch(_, _, _)
        | InstructionKind::Match(_, _, _, _)
        | InstructionKind::Return(_)
        | InstructionKind::ArrayStore(_, _, _, _, _)
        | InstructionKind::BufferStore(_, _, _, _, _)
        | InstructionKind::StructSet(_, _, _, _, _)
        | InstructionKind::PointerStore(_, _)
        | InstructionKind::ParallelFor { .. }
        | InstructionKind::Nop => None,
    }
}

/// DETECTOR: This function MUST be exhaustive.
/// Every operand of an instruction must be accounted for to ensure it's marked as "used".
fn get_operands(inst: &Instruction) -> Vec<Value> {
    let mut operands = Vec::new();
    match &inst.kind {
        InstructionKind::Add(_, l, r)
        | InstructionKind::Sub(_, l, r)
        | InstructionKind::Mul(_, l, r)
        | InstructionKind::SDiv(_, l, r)
        | InstructionKind::UDiv(_, l, r)
        | InstructionKind::SRem(_, l, r)
        | InstructionKind::URem(_, l, r)
        | InstructionKind::And(_, l, r)
        | InstructionKind::Or(_, l, r)
        | InstructionKind::Xor(_, l, r)
        | InstructionKind::Shl(_, l, r)
        | InstructionKind::LShr(_, l, r)
        | InstructionKind::AShr(_, l, r)
        | InstructionKind::FAdd(_, l, r)
        | InstructionKind::FSub(_, l, r)
        | InstructionKind::FMul(_, l, r)
        | InstructionKind::FDiv(_, l, r)
        | InstructionKind::FPow(_, l, r)
        | InstructionKind::Eq(_, l, r)
        | InstructionKind::Ne(_, l, r)
        | InstructionKind::SLt(_, l, r)
        | InstructionKind::SLe(_, l, r)
        | InstructionKind::SGt(_, l, r)
        | InstructionKind::SGe(_, l, r)
        | InstructionKind::ULt(_, l, r)
        | InstructionKind::ULe(_, l, r)
        | InstructionKind::UGt(_, l, r)
        | InstructionKind::UGe(_, l, r)
        | InstructionKind::FLt(_, l, r)
        | InstructionKind::FLe(_, l, r)
        | InstructionKind::FGt(_, l, r)
        | InstructionKind::FGe(_, l, r) => {
            operands.push(*l);
            operands.push(*r);
        }
        InstructionKind::Not(_, s)
        | InstructionKind::FSqrt(_, s)
        | InstructionKind::FSin(_, s)
        | InstructionKind::FCos(_, s)
        | InstructionKind::IToF(_, s, _)
        | InstructionKind::FToI(_, s, _)
        | InstructionKind::BufferLen(_, s)
        | InstructionKind::StructLoad(_, s, _)
        | InstructionKind::StructOffset(_, s, _)
        | InstructionKind::EnumIsVariant(_, s, _)
        | InstructionKind::EnumGetTag(_, s)
        | InstructionKind::EnumExtract(_, s, _)
        | InstructionKind::PointerLoad(_, s)
        | InstructionKind::TupleExtract(_, s, _) => {
            operands.push(*s);
        }
        InstructionKind::PointerStore(p, v) => {
            operands.push(*p);
            operands.push(*v);
        }
        InstructionKind::Branch(c, _, _) => {
            operands.push(*c);
        }
        InstructionKind::Match(s, _, _, _) => {
            operands.push(*s);
        }
        InstructionKind::Return(Some(v)) => {
            operands.push(*v);
        }
        InstructionKind::Return(None)
        | InstructionKind::Alloc(_, _)
        | InstructionKind::Jump(_)
        | InstructionKind::Nop => {}

        InstructionKind::Phi(_, mappings) => {
            for v in mappings.values() {
                operands.push(*v);
            }
        }
        InstructionKind::Call(_, _, args)
        | InstructionKind::StructCreate(_, _, args)
        | InstructionKind::TupleCreate(_, args) => {
            for v in args {
                operands.push(*v);
            }
        }

        InstructionKind::ArrayLoad(_, arr, idx) | InstructionKind::BufferLoad(_, arr, idx) => {
            operands.push(*arr);
            operands.push(*idx);
        }
        InstructionKind::ArrayStore(_, arr, idx, val, _)
        | InstructionKind::BufferStore(_, arr, idx, val, _) => {
            operands.push(*arr);
            operands.push(*idx);
            operands.push(*val);
        }
        InstructionKind::StructSet(_, obj, _, val, _) => {
            operands.push(*obj);
            operands.push(*val);
        }
        InstructionKind::EnumCreate(_, _, _, payload) => {
            if let Some(v) = payload {
                operands.push(*v);
            }
        }
        InstructionKind::Lambda(_, _, captures) => {
            for v in captures {
                operands.push(*v);
            }
        }
        InstructionKind::IndirectCall(_, fn_ptr, args) => {
            operands.push(*fn_ptr);
            for v in args {
                operands.push(*v);
            }
        }
        InstructionKind::ParallelFor {
            start,
            stop,
            step,
            captures,
            ..
        } => {
            operands.push(*start);
            operands.push(*stop);
            operands.push(*step);
            for v in captures {
                operands.push(*v);
            }
        }
        InstructionKind::ConstInt(_, _) | InstructionKind::ConstFloat(_, _) => {}
    }
    operands
}

/// DETECTOR: This function MUST be exhaustive.
/// It defines which instructions have side effects and therefore CANNOT be removed even if their result is unused.
fn has_side_effects(inst: &Instruction) -> bool {
    match &inst.kind {
        // Control flow ALWAYS has side effects
        InstructionKind::Return(_)
        | InstructionKind::Branch(_, _, _)
        | InstructionKind::Match(_, _, _, _)
        | InstructionKind::Jump(_) => true,

        // External calls and indirect calls (could be anything)
        InstructionKind::Call(_, _, _) | InstructionKind::IndirectCall(_, _, _) => true,

        // Memory writes
        InstructionKind::ArrayStore(_, _, _, _, _)
        | InstructionKind::BufferStore(_, _, _, _, _)
        | InstructionKind::StructSet(_, _, _, _, _)
        | InstructionKind::PointerStore(_, _)
        | InstructionKind::Alloc(_, _) => true,

        // Parallel loop
        InstructionKind::ParallelFor { .. } => true,

        // Lambda creation has the side effect of capturing and heap-allocating
        InstructionKind::Lambda(_, _, _) => true,

        // Everything else is pure and can be removed if unused
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
        | InstructionKind::FSin(_, _)
        | InstructionKind::FCos(_, _)
        | InstructionKind::FPow(_, _, _)
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
        | InstructionKind::ConstInt(_, _)
        | InstructionKind::ConstFloat(_, _)
        | InstructionKind::Phi(_, _)
        | InstructionKind::ArrayLoad(_, _, _)
        | InstructionKind::BufferLoad(_, _, _)
        | InstructionKind::BufferLen(_, _)
        | InstructionKind::PointerLoad(_, _)
        | InstructionKind::StructCreate(_, _, _)
        | InstructionKind::StructLoad(_, _, _)
        | InstructionKind::StructOffset(_, _, _)
        | InstructionKind::EnumCreate(_, _, _, _)
        | InstructionKind::EnumIsVariant(_, _, _)
        | InstructionKind::EnumGetTag(_, _)
        | InstructionKind::EnumExtract(_, _, _)
        | InstructionKind::TupleCreate(_, _)
        | InstructionKind::TupleExtract(_, _, _)
        | InstructionKind::Nop => false,
    }
}
