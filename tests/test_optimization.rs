use lila_core::ssa::ir::InstructionKind;
use lila_core::ssa::transform;
use rustpython_parser::ast;
use rustpython_parser::Parse;
use std::collections::HashMap;

#[test]
fn test_constant_folding() {
    let source = "
def fold_test():
    x = 1 + 2 * 3
    return x
"
    .to_string();
    let suite = ast::Suite::parse(&source, "<test>").unwrap();
    let funcs = transform(
        "fold_test".to_string(),
        suite,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap();
    let func = funcs.last().unwrap();

    // Check if the calculation was folded
    // x = 1 + (2 * 3) => 1 + 6 => 7
    let mut found_const_7 = false;
    for block in &func.blocks {
        for inst in &block.instructions {
            if let InstructionKind::ConstInt(_, 7) = &inst.kind {
                found_const_7 = true;
            }
            // Ensure no multiplication or addition remains
            if matches!(
                inst.kind,
                InstructionKind::Add(_, _, _) | InstructionKind::Mul(_, _, _)
            ) {
                panic!("Unfolded instruction found: {:?}", inst);
            }
        }
    }
    assert!(found_const_7, "Constant folding failed to produce 7");
}

#[test]
fn test_dead_code_elimination() {
    let source = "
def dce_test():
    x = 10
    y = 20 # Dead
    z = x + 5
    return z
"
    .to_string();
    let suite = ast::Suite::parse(&source, "<test>").unwrap();
    let funcs = transform(
        "dce_test".to_string(),
        suite,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap();
    let func = funcs.last().unwrap();

    // y = 20 should be gone
    // 10 + 5 should be folded to 15 (if CF runs before DCE)

    let mut found_20 = false;
    let mut found_15 = false;
    for block in &func.blocks {
        for inst in &block.instructions {
            if let InstructionKind::ConstInt(_, 20) = &inst.kind {
                found_20 = true;
            }
            if let InstructionKind::ConstInt(_, 15) = &inst.kind {
                found_15 = true;
            }
        }
    }
    assert!(!found_20, "Dead code (y = 20) was not eliminated");
    assert!(found_15, "Result (z = 15) was not found or folded");
}

#[test]
fn test_type_propagation_loops() {
    let source = "
from lila import f64, i64
def loop_type_test(max_iter: i64) -> f64:
    z = 0.0
    for i in range(max_iter):
        z = z + 1.0
    return z
"
    .to_string();
    let suite = ast::Suite::parse(&source, "<test>").unwrap();
    let funcs = transform(
        "loop_type_test".to_string(),
        suite,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap();
    let func = funcs.last().unwrap();

    // Check if the loop variable 'z' was inferred as F64 and Add was converted to FAdd
    let mut found_fadd = false;
    for block in &func.blocks {
        for inst in &block.instructions {
            if matches!(inst.kind, InstructionKind::FAdd(_, _, _)) {
                found_fadd = true;
            }
        }
    }
    assert!(
        found_fadd,
        "Type propagation failed to convert Add to FAdd in loop"
    );
}
