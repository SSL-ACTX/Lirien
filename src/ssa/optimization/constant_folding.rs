use super::super::ir::{Function, Instruction, InstructionKind, Value};
use std::collections::HashMap;

pub fn fold_constants(func: &mut Function) {
    let mut constants: HashMap<Value, Constant> = HashMap::new();
    // Track values that are redefined (e.g. in loops or different blocks)
    // Actually, in SSA, values should NEVER be redefined.
    // But our "optimization" is a bit naive and processes blocks sequentially.
    // Let's at least clear constants between blocks if we are not doing proper dataflow.

    for block in &mut func.blocks {
        // constants.clear(); // If we clear, we lose cross-block constants.
        // If it's true SSA, we don't need to clear.

        let mut folded_instructions = Vec::new();

        for inst in block.instructions.drain(..) {
            let location = inst.location;
            match &inst.kind {
                InstructionKind::ConstInt(d, v) => {
                    constants.insert(*d, Constant::Int(*v));
                    folded_instructions.push(inst);
                }
                InstructionKind::ConstFloat(d, v) => {
                    constants.insert(*d, Constant::Float(*v));
                    folded_instructions.push(inst);
                }
                InstructionKind::Add(d, l, r) => {
                    if let (Some(Constant::Int(lv)), Some(Constant::Int(rv))) =
                        (constants.get(l), constants.get(r))
                    {
                        let res = lv + rv;
                        constants.insert(*d, Constant::Int(res));
                        folded_instructions.push(Instruction::new(
                            InstructionKind::ConstInt(*d, res),
                            location,
                        ));
                    } else {
                        folded_instructions.push(inst);
                    }
                }
                InstructionKind::Sub(d, l, r) => {
                    if let (Some(Constant::Int(lv)), Some(Constant::Int(rv))) =
                        (constants.get(l), constants.get(r))
                    {
                        let res = lv - rv;
                        constants.insert(*d, Constant::Int(res));
                        folded_instructions.push(Instruction::new(
                            InstructionKind::ConstInt(*d, res),
                            location,
                        ));
                    } else {
                        folded_instructions.push(inst);
                    }
                }
                InstructionKind::Mul(d, l, r) => {
                    if let (Some(Constant::Int(lv)), Some(Constant::Int(rv))) =
                        (constants.get(l), constants.get(r))
                    {
                        let res = lv * rv;
                        constants.insert(*d, Constant::Int(res));
                        folded_instructions.push(Instruction::new(
                            InstructionKind::ConstInt(*d, res),
                            location,
                        ));
                    } else {
                        folded_instructions.push(inst);
                    }
                }
                InstructionKind::FAdd(d, l, r) => {
                    if let (Some(Constant::Float(lv)), Some(Constant::Float(rv))) =
                        (constants.get(l), constants.get(r))
                    {
                        let res = lv + rv;
                        constants.insert(*d, Constant::Float(res));
                        folded_instructions.push(Instruction::new(
                            InstructionKind::ConstFloat(*d, res),
                            location,
                        ));
                    } else {
                        folded_instructions.push(inst);
                    }
                }
                _ => {
                    folded_instructions.push(inst);
                }
            }
        }

        block.instructions = folded_instructions;
    }
}

#[derive(Clone, Debug)]
enum Constant {
    Int(i64),
    Float(f64),
}
