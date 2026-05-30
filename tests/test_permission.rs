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
    m = Hand(x)
    r = Peek(x) 
    return consume_hand(m) + consume_peek(r)

def consume_hand(x: Hand[i64]) -> i64:
    return 1

def consume_peek(x: Peek[i64]) -> i64:
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
        "Expected error for Hand and Peek conflict, but got OK"
    );
}

#[test]
fn test_perm_multiple_refs_ok() {
    init_logs();
    let source = "
def multiple_refs(x: i64) -> i64:
    r1 = Peek(x)
    r2 = Peek(x)
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
        "Expected OK for multiple Peeks, but got error: {:?}",
        result.err()
    );
}

#[test]
fn test_perm_double_move_err() {
    init_logs();
    let source = "
def double_move(x: Held[i64]) -> i64:
    y = consume_held(x)
    z = consume_held(x) # Error: x moved twice
    return y + z

def consume_held(x: Held[i64]) -> i64:
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
