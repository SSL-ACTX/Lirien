use lila_core::ssa::ir::{BasicBlock, Function, Instruction, InstructionKind, Type, Value};
use lila_core::verification::verify;

#[test]
fn test_verify_float_div_zero_fail() {
    let mut func = Function::new("fdiv_zero".to_string());

    let v_zero = func.next_value();
    let v_one = func.next_value();
    let v_res = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::ConstFloat(v_zero, 0.0), None),
            Instruction::new(InstructionKind::ConstFloat(v_one, 1.0), None),
            Instruction::new(InstructionKind::FDiv(v_res, v_one, v_zero), None),
            Instruction::new(InstructionKind::Return(None), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    func.set_type(v_zero, Type::F64);
    func.set_type(v_one, Type::F64);
    func.set_type(v_res, Type::F64);

    let result = verify(&func);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("Potential float division by zero"));
}

#[test]
fn test_verify_fsqrt_negative_fail() {
    let mut func = Function::new("fsqrt_neg".to_string());

    let v_neg = func.next_value();
    let v_res = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::ConstFloat(v_neg, -1.0), None),
            Instruction::new(InstructionKind::FSqrt(v_res, v_neg), None),
            Instruction::new(InstructionKind::Return(None), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    func.set_type(v_neg, Type::F64);
    func.set_type(v_res, Type::F64);

    let result = verify(&func);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("Potential sqrt of negative number"));
}

#[test]
fn test_verify_fpow_domain_fail() {
    let mut func = Function::new("fpow_domain".to_string());

    let v_base = func.next_value();
    let v_exp = func.next_value();
    let v_res = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::ConstFloat(v_base, -2.0), None),
            Instruction::new(InstructionKind::ConstFloat(v_exp, 0.5), None),
            Instruction::new(InstructionKind::FPow(v_res, v_base, v_exp), None),
            Instruction::new(InstructionKind::Return(None), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    func.set_type(v_base, Type::F64);
    func.set_type(v_exp, Type::F64);
    func.set_type(v_res, Type::F64);

    let result = verify(&func);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("Potential domain error in fpow"));
}

#[test]
fn test_verify_float_guarded_success() {
    let mut func = Function::new("fdiv_guarded".to_string());
    func.arg_count = 1; // n: f64

    let v_n = Value(0);
    let v_zero = func.next_value();
    let v_cond = func.next_value();
    let v_res = func.next_value();

    let b_entry = func.next_block();
    let b_true = func.next_block();
    let b_exit = func.next_block();

    func.entry_block = b_entry;

    func.blocks.push(BasicBlock {
        id: b_entry,
        instructions: vec![
            Instruction::new(InstructionKind::ConstFloat(v_zero, 0.0), None),
            Instruction::new(InstructionKind::FGt(v_cond, v_n, v_zero), None),
            Instruction::new(InstructionKind::Branch(v_cond, b_true, b_exit), None),
        ],
        predecessors: vec![],
        successors: vec![b_true, b_exit],
    });

    func.blocks.push(BasicBlock {
        id: b_true,
        instructions: vec![
            Instruction::new(InstructionKind::FDiv(v_res, v_zero, v_n), None), // SAFE because v_n > 0
            Instruction::new(InstructionKind::Jump(b_exit), None),
        ],
        predecessors: vec![b_entry],
        successors: vec![b_exit],
    });

    func.blocks.push(BasicBlock {
        id: b_exit,
        instructions: vec![Instruction::new(InstructionKind::Return(None), None)],
        predecessors: vec![b_entry, b_true],
        successors: vec![],
    });

    func.set_type(v_n, Type::F64);
    func.set_type(v_zero, Type::F64);
    func.set_type(v_cond, Type::I64); // comparisons return i64 (bool)
    func.set_type(v_res, Type::F64);

    let result = verify(&func);
    assert!(result.is_ok(), "Error: {:?}", result.err());
}

#[test]
fn test_verify_float_refined_success() {
    let mut func = Function::new("frefined".to_string());
    func.arg_count = 1;

    let v_x = Value(0);
    func.set_type(v_x, Type::F64);
    func.set_refinement(v_x, "(> {v} 0.0)".to_string());

    let v_one = func.next_value();
    let v_res = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::ConstFloat(v_one, 1.0), None),
            Instruction::new(InstructionKind::FDiv(v_res, v_one, v_x), None), // SAFE due to refinement
            Instruction::new(InstructionKind::Return(None), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    func.set_type(v_one, Type::F64);
    func.set_type(v_res, Type::F64);

    let result = verify(&func);
    assert!(result.is_ok(), "Error: {:?}", result.err());
}
