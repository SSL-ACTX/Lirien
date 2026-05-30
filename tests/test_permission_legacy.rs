use lila_core::bridge::verify_and_compile;

#[test]
#[ignore = "Z3 fractional permissions currently have a limitation with branching moves"]
fn test_reference_checker_branches() {
    let source = "
def branch_move(cond: bool, x: Owned[i64]) -> i64:
    if cond:
        # Move x here
        return consume_owned(x)
    else:
        # Move x here too - this is fine!
        return consume_owned(x)

def consume_owned(x: Owned[i64]) -> i64:
    return 1
"
    .to_string();
    let func_name = "branch_move".to_string();
    let result = verify_and_compile(
        source,
        func_name,
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    );
    assert!(result.is_ok(), "Expected OK, got {:?}", result.err());
}

#[test]
fn test_reference_checker_use_after_branch_move() {
    let source = "
def use_after_branch(cond: bool, x: Owned[i64]) -> i64:
    if cond:
        y = consume_owned(x)
    else:
        pass
    return consume_owned(x) # ERROR: x might have been moved

def consume_owned(x: Owned[i64]) -> i64:
    return 1
"
    .to_string();
    let func_name = "use_after_branch".to_string();
    let result = verify_and_compile(
        source,
        func_name,
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    );

    assert!(
        result.is_err(),
        "Expected error due to potential use-after-move"
    );
}

#[test]
fn test_reference_checker_loop_move() {
    let source = "
def loop_move(n: i64, x: Owned[i64]) -> i64:
    i = 0
    while i < n:
        # Move x in the first iteration
        y = consume_owned(x)
        i = i + 1
    return 1

def consume_owned(x: Owned[i64]) -> i64:
    return 1
"
    .to_string();
    let func_name = "loop_move".to_string();
    let result = verify_and_compile(
        source,
        func_name,
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    );

    assert!(result.is_err(), "Expected error due to move in loop");
}

#[test]
fn test_reference_checker_aliasing_multiple_refs_ok() {
    let source = "
def multiple_refs(x: i64) -> i64:
    r1 = Ref(x)
    r2 = Ref(x)
    consume_owned(r1)
    consume_owned(r2)
    return 1

def consume_owned(x: Owned[i64]) -> i64:
    return 1
"
    .to_string();
    let result = verify_and_compile(
        source,
        "multiple_refs".to_string(),
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    );
    assert!(
        result.is_ok(),
        "Expected OK for multiple Refs: {:?}",
        result.err()
    );
}

#[test]
fn test_reference_checker_aliasing_mut_and_ref_err() {
    let source = "
def mut_and_ref(x: i64) -> i64:
    m = Mut(x)
    r = Ref(x) # Error: x is already mutably referenceed
    consume_owned(m)
    return 1
"
    .to_string();
    let result = verify_and_compile(
        source,
        "mut_and_ref".to_string(),
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    );
    assert!(result.is_err(), "Expected error for Mut and Ref aliasing");
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Memory safety violation"));
}

#[test]
fn test_reference_checker_aliasing_mut_and_mut_err() {
    let source = "
def mut_and_mut(x: i64) -> i64:
    m1 = Mut(x)
    m2 = Mut(x) # Error: x is already mutably referenceed
    consume_owned(m1)
    return 1
"
    .to_string();
    let result = verify_and_compile(
        source,
        "mut_and_mut".to_string(),
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    );
    assert!(result.is_err(), "Expected error for Mut and Mut aliasing");
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Memory safety violation"));
}
