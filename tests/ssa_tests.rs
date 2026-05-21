use lila_core::ssa::transform;
use rustpython_parser::ast;
use rustpython_parser::Parse;
use std::collections::HashMap;

#[test]
fn test_transform_basic_arithmetic() {
    let source = "
def test_arithmetic(a, b):
    x = a + b * 3
    return x
";
    let suite = ast::Suite::parse(source, "<test>").unwrap();
    let func = transform(
        "test_arithmetic".to_string(),
        suite,
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(func.name, "test_arithmetic");
    assert!(!func.blocks.is_empty());

    let block = &func.blocks[0];
    assert!(block.instructions.len() >= 3);
}

#[test]
fn test_transform_function_def() {
    let source = "
def foo(a, b):
    return a + b
";
    let suite = ast::Suite::parse(source, "<test>").unwrap();
    let func = transform("foo".to_string(), suite, HashMap::new(), HashMap::new()).unwrap();

    assert_eq!(func.name, "foo");
}

#[test]
fn test_transform_if_else() {
    let source_with_arg = "
def test_if_else(x):
    if x > 0:
        y = 1
    else:
        y = 2
";
    let suite = ast::Suite::parse(source_with_arg, "<test>").unwrap();
    let func = transform(
        "test_if_else".to_string(),
        suite,
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap();

    assert!(func.blocks.len() >= 4);
}

#[test]
fn test_transform_for_loop() {
    let source = "
def test_for(n):
    total = 0
    for i in range(n):
        total = total + i
    return total
";
    let suite = ast::Suite::parse(source, "<test>").unwrap();
    let func = transform(
        "test_for".to_string(),
        suite,
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(func.name, "test_for");
    assert!(func.blocks.len() >= 4);
}

#[test]
fn test_transform_break_continue() {
    let source = "
def test_break(n):
    i = 0
    while True:
        if i >= n:
            break
        i = i + 1
    return i
";
    let suite = ast::Suite::parse(source, "<test>").unwrap();
    let func = transform(
        "test_break".to_string(),
        suite,
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(func.name, "test_break");
}

#[test]
fn test_transform_short_circuit() {
    let source = "
def sc_test(a, b):
    if a and b:
        return 1
    return 0
";
    let suite = ast::Suite::parse(source, "<test>").unwrap();
    let func = transform("sc_test".to_string(), suite, HashMap::new(), HashMap::new()).unwrap();

    // Should have multiple blocks due to short-circuiting
    assert!(func.blocks.len() >= 3);
}

#[test]
fn test_transform_bitwise_and_shifts() {
    let source = "
def test_bitwise(a, b, c):
    x = (a << 2) | (b >> 1) ^ c
    return x
";
    let suite = ast::Suite::parse(source, "<test>").unwrap();
    let func = transform(
        "test_bitwise".to_string(),
        suite,
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap();

    // Check for Shl, AShr, Xor, Or instructions in the entry block
    let block = &func.blocks[0];
    let mut ops = Vec::new();
    for inst in &block.instructions {
        match &inst.kind {
            lila_core::ssa::ir::InstructionKind::Shl(_, _, _) => ops.push("shl"),
            lila_core::ssa::ir::InstructionKind::AShr(_, _, _) => ops.push("ashr"),
            lila_core::ssa::ir::InstructionKind::Xor(_, _, _) => ops.push("xor"),
            lila_core::ssa::ir::InstructionKind::Or(_, _, _) => ops.push("or"),
            _ => {}
        }
    }
    assert!(ops.contains(&"shl"));
    assert!(ops.contains(&"ashr"));
    assert!(ops.contains(&"xor"));
    assert!(ops.contains(&"or"));
}
