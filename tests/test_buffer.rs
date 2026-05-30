use lila_core::bridge::verify_and_compile;
use lila_core::ssa::ir::InstructionKind;
use rustpython_parser::ast;
use rustpython_parser::Parse;
use std::collections::HashMap;

#[test]
fn test_buffer_interop_ir() {
    let source = "
from lila import Buffer, f64
def buffer_test(data: Buffer[f64], factor: f64) -> None:
    for i in range(len(data)):
        data[i] = data[i] * factor
"
    .to_string();

    let ast = ast::Suite::parse(&source, "<test>").unwrap();
    let funcs = lila_core::ssa::transform(
        "buffer_test".to_string(),
        ast,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap();
    let func = funcs.last().unwrap();

    // Verify IR contains BufferLoad/Store and BufferLen
    let mut found_buflen = false;
    let mut found_bufload = false;
    let mut found_bufstore = false;

    for block in &func.blocks {
        for inst in &block.instructions {
            match &inst.kind {
                InstructionKind::BufferLen(_, _) => found_buflen = true,
                InstructionKind::BufferLoad(_, _, _) => found_bufload = true,
                InstructionKind::BufferStore(_, _, _, _, _) => found_bufstore = true,
                _ => {}
            }
        }
    }

    assert!(found_buflen, "BufferLen instruction not found");
    assert!(found_bufload, "BufferLoad instruction not found");
    assert!(found_bufstore, "BufferStore instruction not found");
}

#[test]
fn test_buffer_out_of_bounds_fail() {
    let source = "
from lila import Buffer, i64
def oob_buffer(data: Buffer[i64]) -> i64:
    # Unsafe access without guard
    return data[10] 
"
    .to_string();

    let result = verify_and_compile(
        source,
        "oob_buffer".to_string(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    );

    assert!(
        result.is_err(),
        "Expected verification failure for OOB buffer access"
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Potential out-of-bounds buffer access"));
}
