use super::super::ir::{Function, Instruction, InstructionKind, Value};
use std::collections::HashSet;

pub fn eliminate_dead_code(func: &mut Function) {
    let mut used_values = HashSet::new();

    let mut worklist: Vec<Value> = Vec::new();

    for block in &func.blocks {
        for inst in &block.instructions {
            if has_side_effects(inst) {
                for operand in get_operands(inst) {
                    worklist.push(operand);
                }
            }
        }
    }

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

    for block in &mut func.blocks {
        block.instructions.retain(|inst| {
            if has_side_effects(inst) {
                return true;
            }
            if let Some(def) = get_def(inst) {
                return used_values.contains(&def);
            }
            true
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
        | InstructionKind::ConstInt(d, _)
        | InstructionKind::ConstFloat(d, _)
        | InstructionKind::Phi(d, _)
        | InstructionKind::Call(d, _, _)
        | InstructionKind::Borrow(d, _)
        | InstructionKind::MutBorrow(d, _)
        | InstructionKind::ArrayLoad(d, _, _)
        | InstructionKind::ArrayStore(d, _, _, _, _)
        | InstructionKind::BufferLoad(d, _, _)
        | InstructionKind::BufferStore(d, _, _, _, _)
        | InstructionKind::BufferLen(d, _)
        | InstructionKind::StructLoad(d, _, _)
        | InstructionKind::StructOffset(d, _, _)
        | InstructionKind::StructSet(d, _, _, _, _) => Some(*d),
        _ => None,
    }
}

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
        InstructionKind::Not(_, s) => {
            operands.push(*s);
        }
        InstructionKind::Branch(c, _, _) => {
            operands.push(*c);
        }
        InstructionKind::Return(Some(v)) => {
            operands.push(*v);
        }
        InstructionKind::Phi(_, mappings) => {
            for v in mappings.values() {
                operands.push(*v);
            }
        }
        InstructionKind::Call(_, _, args) => {
            for v in args {
                operands.push(*v);
            }
        }
        InstructionKind::Borrow(_, s) | InstructionKind::MutBorrow(_, s) => {
            operands.push(*s);
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
        InstructionKind::BufferLen(_, buf) => {
            operands.push(*buf);
        }
        InstructionKind::StructLoad(_, obj, _) => {
            operands.push(*obj);
        }
        InstructionKind::StructOffset(_, obj, _) => {
            operands.push(*obj);
        }
        InstructionKind::StructSet(_, obj, _, val, _) => {
            operands.push(*obj);
            operands.push(*val);
        }
        _ => {}
    }
    operands
}

fn has_side_effects(inst: &Instruction) -> bool {
    matches!(
        &inst.kind,
        InstructionKind::Return(_)
            | InstructionKind::Branch(_, _, _)
            | InstructionKind::Jump(_)
            | InstructionKind::Call(_, _, _)
            | InstructionKind::Borrow(_, _)
            | InstructionKind::MutBorrow(_, _)
            | InstructionKind::ArrayStore(_, _, _, _, _)
            | InstructionKind::BufferStore(_, _, _, _, _)
            | InstructionKind::StructSet(_, _, _, _, _)
    )
}
