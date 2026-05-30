use lila_core::bridge::verify_and_compile;
use std::collections::HashMap;
use tracing_subscriber::EnvFilter;

fn init_logs() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
}

#[test]
fn test_perm_basic_ref_and_mut_conflict() {
    init_logs();
    let source = "
def conflict(x: i64) -> i64:
    m = Mut(x)
    r = Ref(x) 
    return consume_mut(m) + consume_ref(r)

def consume_mut(x: Mut[i64]) -> i64:
    return 1

def consume_ref(x: Ref[i64]) -> i64:
    return 1
"
    .to_string();

    let result = verify_and_compile(
        source,
        "conflict".to_string(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    );

    assert!(
        result.is_err(),
        "Expected error for Mut and Ref conflict, but got OK"
    );
}

#[test]
fn test_perm_multiple_refs_ok() {
    init_logs();
    let source = "
def multiple_refs(x: i64) -> i64:
    r1 = Ref(x)
    r2 = Ref(x)
    return 1
"
    .to_string();

    let result = verify_and_compile(
        source,
        "multiple_refs".to_string(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    );

    assert!(
        result.is_ok(),
        "Expected OK for multiple Refs, but got error: {:?}",
        result.err()
    );
}

#[test]
fn test_perm_double_move_err() {
    init_logs();
    let source = "
def double_move(x: Owned[i64]) -> i64:
    y = consume_owned(x)
    z = consume_owned(x) # Error: x moved twice
    return y + z

def consume_owned(x: Owned[i64]) -> i64:
    return 1
"
    .to_string();

    let result = verify_and_compile(
        source,
        "double_move".to_string(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    );

    assert!(
        result.is_err(),
        "Expected error for double move, but got OK"
    );
}
