use lila_core::backend::compile;
use lila_core::ssa::ir::{BasicBlock, Function, Instruction, InstructionKind};

#[test]
fn test_compile_placeholder() {
    let mut func = Function::new("test".to_string());
    func.return_type = lila_core::ssa::ir::Type::I64;
    let v0 = func.next_value();
    func.arg_count = 1;
    let b0 = func.next_block();
    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![Instruction::new(InstructionKind::Return(Some(v0)), None)],
        predecessors: vec![],
        successors: vec![],
    });
    func.entry_block = b0;

    assert!(compile(&func).is_ok());
}

#[test]
fn test_compile_basic_arithmetic() {
    let mut func = Function::new("add_one".to_string());
    func.return_type = lila_core::ssa::ir::Type::I64;
    func.arg_count = 1;

    let v_arg = func.next_value(); // v0: input
    let v_1 = func.next_value(); // v1: 1
    let v_res = func.next_value(); // v2: v0 + v1

    let b0 = func.next_block();
    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::ConstInt(v_1, 1), None),
            Instruction::new(InstructionKind::Add(v_res, v_arg, v_1), None),
            Instruction::new(InstructionKind::Return(Some(v_res)), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });
    func.entry_block = b0;

    let result = compile(&func);
    assert!(result.is_ok());
    assert!(result.unwrap() > 0);
}

#[test]
fn test_compile_float_comparisons() {
    let mut func = Function::new("fcmps".to_string());
    func.return_type = lila_core::ssa::ir::Type::Bool;
    func.arg_count = 2;

    let v_a = func.next_value();
    let v_b = func.next_value();
    let v_res = func.next_value();

    func.set_type(v_a, lila_core::ssa::ir::Type::F64);
    func.set_type(v_b, lila_core::ssa::ir::Type::F64);
    func.set_type(v_res, lila_core::ssa::ir::Type::Bool);

    let b0 = func.next_block();
    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::FLt(v_res, v_a, v_b), None),
            Instruction::new(InstructionKind::Return(Some(v_res)), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });
    func.entry_block = b0;

    let result = compile(&func);
    if let Err(e) = &result {
        println!("Compilation failed: {}", e);
    }
    assert!(result.is_ok());
}

#[test]
fn test_compile_math_intrinsics() {
    let mut func = Function::new("math_test".to_string());
    func.return_type = lila_core::ssa::ir::Type::F64;
    func.arg_count = 1;

    let v_arg = func.next_value();
    let v_sin = func.next_value();
    let v_cos = func.next_value();
    let v_sum = func.next_value();

    func.set_type(v_arg, lila_core::ssa::ir::Type::F64);
    func.set_type(v_sin, lila_core::ssa::ir::Type::F64);
    func.set_type(v_cos, lila_core::ssa::ir::Type::F64);
    func.set_type(v_sum, lila_core::ssa::ir::Type::F64);

    let b0 = func.next_block();
    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::FSin(v_sin, v_arg), None),
            Instruction::new(InstructionKind::FCos(v_cos, v_arg), None),
            Instruction::new(InstructionKind::FAdd(v_sum, v_sin, v_cos), None),
            Instruction::new(InstructionKind::Return(Some(v_sum)), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });
    func.entry_block = b0;

    let result = compile(&func);
    if let Err(e) = &result {
        println!("Compilation failed: {}", e);
    }
    assert!(result.is_ok());
}

#[test]
fn test_compile_pow() {
    let mut func = Function::new("pow_test".to_string());
    func.return_type = lila_core::ssa::ir::Type::F64;
    func.arg_count = 2;

    let v_b = func.next_value();
    let v_e = func.next_value();
    let v_res = func.next_value();

    func.set_type(v_b, lila_core::ssa::ir::Type::F64);
    func.set_type(v_e, lila_core::ssa::ir::Type::F64);
    func.set_type(v_res, lila_core::ssa::ir::Type::F64);

    let b0 = func.next_block();
    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::FPow(v_res, v_b, v_e), None),
            Instruction::new(InstructionKind::Return(Some(v_res)), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });
    func.entry_block = b0;

    let result = compile(&func);
    if let Err(e) = &result {
        println!("Compilation failed: {}", e);
    }
    assert!(result.is_ok());
}
