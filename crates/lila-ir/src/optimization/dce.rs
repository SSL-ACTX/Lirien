use super::super::ir::{Function, Instruction, InstructionKind, Value};
use std::collections::HashSet;

pub fn eliminate_dead_code(func: &mut Function) {
    let mut used_values = HashSet::new();

    let mut worklist: Vec<Value> = Vec::new();

    // 1. Identify all values that are "required" (side effects or return values)
    for block in &func.blocks {
        for inst in &block.instructions {
            if inst.has_side_effects() {
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
            if inst.has_side_effects() {
                return true;
            }
            if let Some(def) = inst.get_def() {
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
            if let Some(d) = inst.get_def() {
                if d == val {
                    return Some(inst);
                }
            }
        }
    }
    None
}

fn get_operands(inst: &Instruction) -> Vec<Value> {
    let mut operands = inst.get_uses();
    // Special handling for Phi nodes in DCE (we want ALL inputs)
    if let InstructionKind::Phi(_, mappings) = &inst.kind {
        for v in mappings.values() {
            operands.push(*v);
        }
    }
    operands
}
