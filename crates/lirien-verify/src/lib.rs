//! # Lirien Verify
//!
//! This crate implements the formal verification engine for the Lirien compiler.
//! It translates the SSA IR control flow graph and refinement constraints into SMT-LIB formulas
//! and invokes Z3 to formally verify that the JIT-compiled functions satisfy their liquid type contracts
//! and are free of logical errors (e.g. out-of-bounds array access, null pointer dereferences).

pub mod backend;
pub mod refinement;
pub mod verifier;
pub mod z3_backend;

use self::backend::SolverBackend;
use self::verifier::verify_with_context;
use ::z3::{Context, Solver};
use lirien_ir::analysis::{interval, liveness};
use lirien_ir::ir::Function;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

static VERIFY_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Entry point to formally verify a Lirien JIT function using Z3.
///
/// Runs:
/// 1. Dataflow liveness analysis to minimize verification state.
/// 2. Numeric range/interval analysis to infer bounds (hints for Z3).
/// 3. Constraint-to-SMT mapping, asserting path conditions and checking refinement safety goals.
///
/// # Errors
/// Returns an error string if a verification contract is violated, or SMT solver fails/times out.
pub fn verify(func: &Function, timeout_ms: u32) -> Result<Option<String>, String> {
    info!(target: "lirien::verify", "Verifying function '{}'...", func.name);

    let uid = VERIFY_COUNT.fetch_add(1, Ordering::SeqCst);

    // 1. Liveness Analysis
    tracing::info!(target: "lirien::verify", "Running liveness analysis for '{}'...", func.name);
    let liveness = liveness::analyze_liveness(func);

    // 2. Interval Analysis (Hint for Z3)
    tracing::info!(target: "lirien::verify", "Running interval analysis for '{}'...", func.name);
    let analysis_results = interval::analyze(func);

    // 3. Logic Verification with Backend
    tracing::info!(target: "lirien::verify", "Starting verification for '{}'...", func.name);
    let ctx = Context::thread_local();
    let solver = Solver::new();

    let mut backend = z3_backend::Z3Backend::new(&ctx, &solver);

    backend.set_timeout(timeout_ms);

    verify_with_context(&mut backend, func, &analysis_results, liveness, uid)
}
