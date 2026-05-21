use lila_core::ssa::ir::{BasicBlock, Function, Instruction, InstructionKind, Type};
use lila_core::verification::verify;

#[test]
fn test_verify_buffer_len_refinement_ok() {
    let mut func = Function::new("head".to_string());
    func.arg_count = 1;

    // def head(buf: Buffer[i64]):
    //   if len(buf) > 0:
    //     return buf[0]
    //   return -1

    let v_buf = func.next_value();
    func.set_type(v_buf, Type::Buffer(Box::new(Type::I64)));

    let v_len = func.next_value();
    let v_0 = func.next_value();
    let v_cond = func.next_value();
    let v_idx = func.next_value();
    let v_res = func.next_value();
    let v_neg1 = func.next_value();

    let b0 = func.next_block();
    let b1 = func.next_block();
    let b2 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::BufferLen(v_len, v_buf), None),
            Instruction::new(InstructionKind::ConstInt(v_0, 0), None),
            Instruction::new(InstructionKind::SGt(v_cond, v_len, v_0), None),
            Instruction::new(InstructionKind::Branch(v_cond, b1, b2), None),
        ],
        predecessors: vec![],
        successors: vec![b1, b2],
    });

    func.blocks.push(BasicBlock {
        id: b1,
        instructions: vec![
            Instruction::new(InstructionKind::ConstInt(v_idx, 0), None),
            Instruction::new(InstructionKind::BufferLoad(v_res, v_buf, v_idx), None),
            Instruction::new(InstructionKind::Return(Some(v_res)), None),
        ],
        predecessors: vec![b0],
        successors: vec![],
    });

    func.blocks.push(BasicBlock {
        id: b2,
        instructions: vec![
            Instruction::new(InstructionKind::ConstInt(v_neg1, -1), None),
            Instruction::new(InstructionKind::Return(Some(v_neg1)), None),
        ],
        predecessors: vec![b0],
        successors: vec![],
    });

    let result = verify(&func);
    assert!(result.is_ok(), "Expected OK, got {:?}", result.err());
}

#[test]
fn test_verify_buffer_len_refinement_fail() {
    let mut func = Function::new("unsafe_head".to_string());
    func.arg_count = 1;

    // def unsafe_head(buf: Buffer[i64]):
    //   return buf[0] # ERROR: length might be 0

    let v_buf = func.next_value();
    func.set_type(v_buf, Type::Buffer(Box::new(Type::I64)));

    let v_idx = func.next_value();
    let v_res = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::ConstInt(v_idx, 0), None),
            Instruction::new(InstructionKind::BufferLoad(v_res, v_buf, v_idx), None),
            Instruction::new(InstructionKind::Return(Some(v_res)), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    let result = verify(&func);
    assert!(
        result.is_err(),
        "Expected error due to unchecked buffer length"
    );
    assert!(result
        .unwrap_err()
        .contains("Potential out-of-bounds buffer access"));
}

#[test]
fn test_verify_complex_refinement_ok() {
    let mut func = Function::new("complex".to_string());
    func.arg_count = 1;

    // def complex(x: Refined[i64, lambda x: (and (> x 0) (< x 10))]):
    //   y = 100 / x # Safe because x > 0

    let v_x = func.next_value();
    func.set_type(v_x, Type::I64);
    func.set_refinement(v_x, "(and (> {v} 0) (< {v} 10))".to_string());

    let v_100 = func.next_value();
    let v_res = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::ConstInt(v_100, 100), None),
            Instruction::new(InstructionKind::SDiv(v_res, v_100, v_x), None),
            Instruction::new(InstructionKind::Return(Some(v_res)), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    let result = verify(&func);
    assert!(result.is_ok(), "Expected OK, got {:?}", result.err());
}

#[test]
fn test_verify_struct_field_refinement_ok() {
    let mut func = Function::new("struct_ref".to_string());
    func.arg_count = 1;

    // @struct
    // class Point:
    //   x: i64
    //
    // def struct_ref(p: Refined[Point, lambda p: p.x > 0]) -> i64:
    //   return 100 / p.x

    let v_p = func.next_value();
    func.set_type(v_p, Type::Struct("Point".to_string()));
    // Refinement applies to the struct object (modeled as Array in Z3)
    // Offset 0 for field 'x'
    func.set_refinement(v_p, "(> (select {v} 0) 0)".to_string());

    let v_x = func.next_value();
    let v_100 = func.next_value();
    let v_res = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::StructLoad(v_x, v_p, 0), None),
            Instruction::new(InstructionKind::ConstInt(v_100, 100), None),
            Instruction::new(InstructionKind::SDiv(v_res, v_100, v_x), None),
            Instruction::new(InstructionKind::Return(Some(v_res)), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    func.set_type(v_x, Type::I64);

    let result = verify(&func);
    assert!(
        result.is_ok(),
        "Expected OK for struct field refinement, got {:?}",
        result.err()
    );
}

#[test]
fn test_verify_arithmetic_refinement_ok() {
    let mut func = Function::new("arith_ref".to_string());
    func.arg_count = 1;

    // def arith_ref(x: Refined[i64, lambda x: x == 5 + 5]):
    //   y = 10 / x # Safe because x == 10

    let v_x = func.next_value();
    func.set_type(v_x, Type::I64);
    func.set_refinement(v_x, "(= {v} (+ 5 5))".to_string());

    let v_10 = func.next_value();
    let v_res = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::ConstInt(v_10, 10), None),
            Instruction::new(InstructionKind::SDiv(v_res, v_10, v_x), None),
            Instruction::new(InstructionKind::Return(Some(v_res)), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    let result = verify(&func);
    assert!(result.is_ok(), "Expected OK, got {:?}", result.err());
}
