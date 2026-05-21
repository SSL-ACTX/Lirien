use lila_core::ssa::ir::{BasicBlock, Function, Instruction, InstructionKind, Type};
use lila_core::verification::verify;

#[test]
fn test_verify_div_guarded_in_loop() {
    let mut func = Function::new("guarded_loop".to_string());
    func.arg_count = 1;

    // def guarded_loop(n):
    //   i = 0
    //   while i < n:
    //     if i > 0:
    //       x = 10 / i
    //     i = i + 1

    let v_n = func.next_value(); // v0: n
    let v_0 = func.next_value(); // v1: constant 0
    let v_1 = func.next_value(); // v2: constant 1
    let v_10 = func.next_value(); // v3: constant 10

    let v_i_init = func.next_value(); // v4: i = 0

    // Header Phis
    let v_i_phi = func.next_value(); // v5: i_phi

    let v_cond_while = func.next_value(); // v6: i < n
    let v_cond_if = func.next_value(); // v7: i > 0

    let v_div = func.next_value(); // v8: 10 / i
    let v_i_next = func.next_value(); // v9: i + 1

    let b_entry = func.next_block();
    let b_header = func.next_block();
    let b_body = func.next_block();
    let b_if_true = func.next_block();
    let b_if_merge = func.next_block();
    let b_exit = func.next_block();

    func.entry_block = b_entry;

    // b_entry
    func.blocks.push(BasicBlock {
        id: b_entry,
        instructions: vec![
            Instruction::new(InstructionKind::ConstInt(v_0, 0), None),
            Instruction::new(InstructionKind::ConstInt(v_1, 1), None),
            Instruction::new(InstructionKind::ConstInt(v_10, 10), None),
            Instruction::new(InstructionKind::ConstInt(v_i_init, 0), None),
            Instruction::new(InstructionKind::Jump(b_header), None),
        ],
        predecessors: vec![],
        successors: vec![b_header],
    });

    // b_header
    let mut phi_ops = std::collections::HashMap::new();
    phi_ops.insert(b_entry, v_i_init);
    phi_ops.insert(b_if_merge, v_i_next);
    func.blocks.push(BasicBlock {
        id: b_header,
        instructions: vec![
            Instruction::new(InstructionKind::Phi(v_i_phi, phi_ops), None),
            Instruction::new(InstructionKind::SLt(v_cond_while, v_i_phi, v_n), None),
            Instruction::new(InstructionKind::Branch(v_cond_while, b_body, b_exit), None),
        ],
        predecessors: vec![b_entry, b_if_merge],
        successors: vec![b_body, b_exit],
    });

    // b_body
    func.blocks.push(BasicBlock {
        id: b_body,
        instructions: vec![
            Instruction::new(InstructionKind::SGt(v_cond_if, v_i_phi, v_0), None),
            Instruction::new(
                InstructionKind::Branch(v_cond_if, b_if_true, b_if_merge),
                None,
            ),
        ],
        predecessors: vec![b_header],
        successors: vec![b_if_true, b_if_merge],
    });

    // b_if_true
    func.blocks.push(BasicBlock {
        id: b_if_true,
        instructions: vec![
            Instruction::new(InstructionKind::SDiv(v_div, v_10, v_i_phi), None), // SAFE because v_i_phi > 0 here
            Instruction::new(InstructionKind::Jump(b_if_merge), None),
        ],
        predecessors: vec![b_body],
        successors: vec![b_if_merge],
    });

    // b_if_merge
    func.blocks.push(BasicBlock {
        id: b_if_merge,
        instructions: vec![
            Instruction::new(InstructionKind::Add(v_i_next, v_i_phi, v_1), None),
            Instruction::new(InstructionKind::Jump(b_header), None),
        ],
        predecessors: vec![b_body, b_if_true],
        successors: vec![b_header],
    });

    // b_exit
    func.blocks.push(BasicBlock {
        id: b_exit,
        instructions: vec![Instruction::new(InstructionKind::Return(None), None)],
        predecessors: vec![b_header],
        successors: vec![],
    });

    let result = verify(&func);
    assert!(
        result.is_ok(),
        "Should be safe: {:?}, IR: \n{}",
        result.err(),
        {
            func.dump();
            ""
        }
    );
}

#[test]
fn test_verify_div_unguarded_in_loop_fail() {
    let mut func = Function::new("unguarded_loop".to_string());
    func.arg_count = 1;

    // def unguarded_loop(n):
    //   i = 0
    //   while i < n:
    //     x = 10 / i  <-- UNSAFE on first iteration (i=0)
    //     i = i + 1

    let v_n = func.next_value();
    let v_0 = func.next_value();
    let v_1 = func.next_value();
    let v_10 = func.next_value();
    let v_i_init = func.next_value();
    let v_i_phi = func.next_value();
    let v_cond_while = func.next_value();
    let v_div = func.next_value();
    let v_i_next = func.next_value();

    let b_entry = func.next_block();
    let b_header = func.next_block();
    let b_body = func.next_block();
    let b_exit = func.next_block();

    func.entry_block = b_entry;

    // b_entry
    func.blocks.push(BasicBlock {
        id: b_entry,
        instructions: vec![
            Instruction::new(InstructionKind::ConstInt(v_0, 0), None),
            Instruction::new(InstructionKind::ConstInt(v_1, 1), None),
            Instruction::new(InstructionKind::ConstInt(v_10, 10), None),
            Instruction::new(InstructionKind::ConstInt(v_i_init, 0), None),
            Instruction::new(InstructionKind::Jump(b_header), None),
        ],
        predecessors: vec![],
        successors: vec![b_header],
    });

    // b_header
    let mut phi_ops = std::collections::HashMap::new();
    phi_ops.insert(b_entry, v_i_init);
    phi_ops.insert(b_body, v_i_next);
    func.blocks.push(BasicBlock {
        id: b_header,
        instructions: vec![
            Instruction::new(InstructionKind::Phi(v_i_phi, phi_ops), None),
            Instruction::new(InstructionKind::SLt(v_cond_while, v_i_phi, v_n), None),
            Instruction::new(InstructionKind::Branch(v_cond_while, b_body, b_exit), None),
        ],
        predecessors: vec![b_entry, b_body],
        successors: vec![b_body, b_exit],
    });

    // b_body
    func.blocks.push(BasicBlock {
        id: b_body,
        instructions: vec![
            Instruction::new(InstructionKind::SDiv(v_div, v_10, v_i_phi), None), // UNSAFE
            Instruction::new(InstructionKind::Add(v_i_next, v_i_phi, v_1), None),
            Instruction::new(InstructionKind::Jump(b_header), None),
        ],
        predecessors: vec![b_header],
        successors: vec![b_header],
    });

    // b_exit
    func.blocks.push(BasicBlock {
        id: b_exit,
        instructions: vec![Instruction::new(InstructionKind::Return(None), None)],
        predecessors: vec![b_header],
        successors: vec![],
    });

    let result = verify(&func);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Potential division by zero"));
}

#[test]
fn test_verify_use_after_move_fail() {
    let mut func = Function::new("move_test".to_string());
    func.arg_count = 1;

    // def move_test(x: Owned[i64]):
    //   bar(x)
    //   return x

    let v_x = func.next_value();
    func.set_type(v_x, Type::Owned(Box::new(Type::I64)));

    let v_res = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(
                InstructionKind::Call(v_res, "bar".to_string(), vec![v_x]),
                None,
            ),
            Instruction::new(InstructionKind::Return(Some(v_x)), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    let result = verify(&func);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Use-after-move"));
}

#[test]
fn test_verify_refined_missing_guard_fail() {
    let mut func = Function::new("refined_test".to_string());
    func.arg_count = 1;

    // def refined_test(x: Refined[i64, lambda x: x < 100]):
    //   y = x + 1
    //   z = 10 / (100 - y)  # UNSAFE if x = 99

    let v_x = func.next_value();
    func.set_type(v_x, Type::I64);
    func.set_refinement(v_x, "(< {v} 100)".to_string());

    let v_1 = func.next_value();
    let v_y = func.next_value();
    let v_100 = func.next_value();
    let v_diff = func.next_value();
    let v_10 = func.next_value();
    let v_z = func.next_value();

    let b0 = func.next_block();
    func.entry_block = b0;

    func.blocks.push(BasicBlock {
        id: b0,
        instructions: vec![
            Instruction::new(InstructionKind::ConstInt(v_1, 1), None),
            Instruction::new(InstructionKind::Add(v_y, v_x, v_1), None),
            Instruction::new(InstructionKind::ConstInt(v_100, 100), None),
            Instruction::new(InstructionKind::Sub(v_diff, v_100, v_y), None),
            Instruction::new(InstructionKind::ConstInt(v_10, 10), None),
            Instruction::new(InstructionKind::SDiv(v_z, v_10, v_diff), None),
            Instruction::new(InstructionKind::Return(None), None),
        ],
        predecessors: vec![],
        successors: vec![],
    });

    let result = verify(&func);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Potential division by zero"));
}
