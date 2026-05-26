use crate::ssa::ir::{Function, InstructionKind, Type};
use std::collections::HashMap;

pub fn propagate_types(func: &mut Function) {
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 100 {
        changed = false;
        iterations += 1;
        let mut new_types = HashMap::new();

        for block in &func.blocks {
            for inst in &block.instructions {
                match &inst.kind {
                    InstructionKind::Add(d, l, r)
                    | InstructionKind::Sub(d, l, r)
                    | InstructionKind::Mul(d, l, r)
                    | InstructionKind::SDiv(d, l, r)
                    | InstructionKind::UDiv(d, l, r)
                    | InstructionKind::SRem(d, l, r)
                    | InstructionKind::URem(d, l, r)
                    | InstructionKind::And(d, l, r)
                    | InstructionKind::Or(d, l, r)
                    | InstructionKind::Xor(d, l, r)
                    | InstructionKind::Shl(d, l, r)
                    | InstructionKind::LShr(d, l, r)
                    | InstructionKind::AShr(d, l, r) => {
                        let l_ty = func.get_type(*l);
                        let r_ty = func.get_type(*r);
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            if l_ty != Type::Unknown {
                                new_types.insert(*d, l_ty);
                            } else if r_ty != Type::Unknown {
                                new_types.insert(*d, r_ty);
                            }
                        }
                    }
                    InstructionKind::Not(d, s) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let s_ty = func.get_type(*s);
                            if s_ty != Type::Unknown {
                                new_types.insert(*d, s_ty);
                            }
                        }
                    }
                    InstructionKind::Eq(d, l, r)
                    | InstructionKind::Ne(d, l, r)
                    | InstructionKind::SLt(d, l, r)
                    | InstructionKind::SLe(d, l, r)
                    | InstructionKind::SGt(d, l, r)
                    | InstructionKind::SGe(d, l, r)
                    | InstructionKind::ULt(d, l, r)
                    | InstructionKind::ULe(d, l, r)
                    | InstructionKind::UGt(d, l, r)
                    | InstructionKind::UGe(d, l, r)
                    | InstructionKind::FLt(d, l, r)
                    | InstructionKind::FLe(d, l, r)
                    | InstructionKind::FGt(d, l, r)
                    | InstructionKind::FGe(d, l, r) => {
                        let l_ty = func.get_type(*l);
                        let r_ty = func.get_type(*r);
                        let current_ty = func.get_type(*d);

                        if current_ty == Type::Unknown {
                            if l_ty != Type::Unknown {
                                new_types.insert(*d, l_ty);
                            } else if r_ty != Type::Unknown {
                                new_types.insert(*d, r_ty);
                            }
                        }
                    }
                    InstructionKind::FAdd(d, l, r)
                    | InstructionKind::FSub(d, l, r)
                    | InstructionKind::FMul(d, l, r)
                    | InstructionKind::FDiv(d, l, r) => {
                        let l_ty = func.get_type(*l);
                        let r_ty = func.get_type(*r);
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            if l_ty == Type::F64 || r_ty == Type::F64 {
                                new_types.insert(*d, Type::F64);
                            } else if l_ty == Type::F32 || r_ty == Type::F32 {
                                new_types.insert(*d, Type::F32);
                            }
                        }
                    }
                    InstructionKind::Phi(d, mappings) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            for val in mappings.values() {
                                let ty = func.get_type(*val);
                                if ty != Type::Unknown {
                                    new_types.insert(*d, ty);
                                    break;
                                }
                            }
                        }
                    }
                    InstructionKind::BufferLoad(d, buf, _idx) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            if let Type::Buffer(inner) = func.get_type(*buf) {
                                new_types.insert(*d, *inner);
                            }
                        }
                    }
                    InstructionKind::BufferStore(d, buf, _idx, _val, _ty) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, func.get_type(*buf));
                        }
                    }
                    InstructionKind::BufferLen(d, _buf) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, Type::I64);
                        }
                    }
                    InstructionKind::Return(Some(v)) => {
                        let ty = func.get_type(*v);
                        if func.return_type == Type::Unknown && ty != Type::Unknown {
                            func.return_type = ty;
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        if !new_types.is_empty() {
            for (val, ty) in new_types {
                func.set_type(val, ty);
                changed = true;
            }
        }
    }

    // Fix instruction kinds based on propagated types
    let mut instruction_updates = Vec::new();

    for block in &func.blocks {
        for inst in &block.instructions {
            let new_kind = match &inst.kind {
                InstructionKind::Add(d, l, r) => {
                    let l_ty = func.get_type(*l);
                    let r_ty = func.get_type(*r);
                    if matches!(l_ty, Type::F32 | Type::F64)
                        || matches!(r_ty, Type::F32 | Type::F64)
                    {
                        Some(InstructionKind::FAdd(*d, *l, *r))
                    } else {
                        None
                    }
                }
                InstructionKind::Sub(d, l, r) => {
                    let l_ty = func.get_type(*l);
                    let r_ty = func.get_type(*r);
                    if matches!(l_ty, Type::F32 | Type::F64)
                        || matches!(r_ty, Type::F32 | Type::F64)
                    {
                        Some(InstructionKind::FSub(*d, *l, *r))
                    } else {
                        None
                    }
                }
                InstructionKind::Mul(d, l, r) => {
                    let l_ty = func.get_type(*l);
                    let r_ty = func.get_type(*r);
                    if matches!(l_ty, Type::F32 | Type::F64)
                        || matches!(r_ty, Type::F32 | Type::F64)
                    {
                        Some(InstructionKind::FMul(*d, *l, *r))
                    } else {
                        None
                    }
                }
                InstructionKind::ArrayLoad(d, arr, idx) => {
                    let ty = func.get_type(*arr);
                    if let Type::Buffer(_) = ty {
                        Some(InstructionKind::BufferLoad(*d, *arr, *idx))
                    } else {
                        None
                    }
                }
                InstructionKind::ArrayStore(d, arr, idx, val, _ty_hint) => {
                    let ty = func.get_type(*arr);
                    if let Type::Buffer(inner) = ty {
                        Some(InstructionKind::BufferStore(*d, *arr, *idx, *val, *inner))
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(k) = new_kind {
                instruction_updates.push((block.id, inst as *const _ as usize, k));
            }
        }
    }

    for (block_id, inst_addr, new_kind) in instruction_updates {
        if let Some(block) = func.blocks.iter_mut().find(|b| b.id == block_id) {
            for inst in &mut block.instructions {
                if (inst as *const _) as usize == inst_addr {
                    inst.kind = new_kind;
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::ir::{BasicBlock, BlockId, Instruction, InstructionKind};

    #[test]
    fn test_type_propagation_phi() {
        let mut func = Function::new("test".to_string());

        let v0 = func.next_value();
        let v1 = func.next_value();
        let v2 = func.next_value();

        func.set_type(v0, Type::F64);

        let b0_id = BlockId(0);
        let b1_id = BlockId(1);

        let b0 = BasicBlock {
            id: b0_id,
            instructions: vec![
                Instruction::new(InstructionKind::ConstFloat(v0, 1.0), None),
                Instruction::new(InstructionKind::Jump(b1_id), None),
            ],
            predecessors: vec![],
            successors: vec![b1_id],
        };

        let mut phi_map = HashMap::new();
        phi_map.insert(b0_id, v0);
        phi_map.insert(b1_id, v2);

        let b1 = BasicBlock {
            id: b1_id,
            instructions: vec![
                Instruction::new(InstructionKind::Phi(v1, phi_map), None),
                Instruction::new(InstructionKind::Add(v2, v1, v0), None),
                Instruction::new(InstructionKind::Jump(b1_id), None),
            ],
            predecessors: vec![b0_id, b1_id],
            successors: vec![b1_id],
        };

        func.blocks.push(b0);
        func.blocks.push(b1);

        propagate_types(&mut func);

        assert_eq!(func.get_type(v1), Type::F64);
        assert_eq!(func.get_type(v2), Type::F64);

        let b1_ref = func.blocks.iter().find(|b| b.id == b1_id).unwrap();
        match &b1_ref.instructions[1].kind {
            InstructionKind::FAdd(d, l, r) => {
                assert_eq!(*d, v2);
                assert_eq!(*l, v1);
                assert_eq!(*r, v0);
            }
            k => panic!("Expected FAdd, got {:?}", k),
        }
    }

    #[test]
    fn test_type_propagation_buffer_phi() {
        let mut func = Function::new("test_buffer".to_string());

        let v0 = func.next_value(); // data: Buffer[f64]
        let v1 = func.next_value(); // factor: f64
        let v2 = func.next_value(); // len
        let v3 = func.next_value(); // 0
        let v4 = func.next_value(); // i (phi)
        let v6 = func.next_value(); // data (phi)
        let v7 = func.next_value(); // data[i]
        let v9 = func.next_value(); // data[i] * factor
        let v10 = func.next_value(); // data[i] = ...
        let v12 = func.next_value(); // i + 1

        func.set_type(v0, Type::Buffer(Box::new(Type::F64)));
        func.set_type(v1, Type::F64);

        let b0_id = BlockId(0);
        let b1_id = BlockId(1);
        let b2_id = BlockId(2);

        let b0 = BasicBlock {
            id: b0_id,
            instructions: vec![
                Instruction::new(InstructionKind::BufferLen(v2, v0), None),
                Instruction::new(InstructionKind::ConstInt(v3, 0), None),
                Instruction::new(InstructionKind::Jump(b1_id), None),
            ],
            predecessors: vec![],
            successors: vec![b1_id],
        };

        let mut i_phi = HashMap::new();
        i_phi.insert(b0_id, v3);
        i_phi.insert(b2_id, v12);

        let mut data_phi = HashMap::new();
        data_phi.insert(b0_id, v0);
        data_phi.insert(b2_id, v10);

        let b1 = BasicBlock {
            id: b1_id,
            instructions: vec![
                Instruction::new(InstructionKind::Phi(v4, i_phi), None),
                Instruction::new(InstructionKind::Phi(v6, data_phi), None),
                Instruction::new(InstructionKind::Jump(b2_id), None), // Simplified
            ],
            predecessors: vec![b0_id, b2_id],
            successors: vec![b2_id],
        };

        let b2 = BasicBlock {
            id: b2_id,
            instructions: vec![
                Instruction::new(InstructionKind::BufferLoad(v7, v6, v4), None),
                Instruction::new(InstructionKind::FMul(v9, v7, v1), None),
                Instruction::new(
                    InstructionKind::BufferStore(v10, v6, v4, v9, Type::F64),
                    None,
                ),
                Instruction::new(InstructionKind::Add(v12, v4, v3), None),
                Instruction::new(InstructionKind::Jump(b1_id), None),
            ],
            predecessors: vec![b1_id],
            successors: vec![b1_id],
        };

        func.blocks.push(b0);
        func.blocks.push(b1);
        func.blocks.push(b2);

        propagate_types(&mut func);

        assert_eq!(func.get_type(v6), Type::Buffer(Box::new(Type::F64)));
        assert_eq!(func.get_type(v7), Type::F64);
        assert_eq!(func.get_type(v10), Type::Buffer(Box::new(Type::F64)));
    }
}
