pub mod backend;
pub mod refinement_parser;
pub mod verifier;
pub mod z3_backend;

use self::backend::SolverBackend;
use self::verifier::verify_with_context;
use ::z3::{Context, Solver};
use lila_ir::analysis::{interval, liveness};
use lila_ir::ir::Function;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

static VERIFY_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn verify(func: &Function, timeout_ms: u32) -> Result<(), String> {
    info!(target: "lila::verify", "Verifying function '{}'...", func.name);

    let uid = VERIFY_COUNT.fetch_add(1, Ordering::SeqCst);

    // 1. Liveness Analysis
    tracing::info!(target: "lila::verify", "Running liveness analysis for '{}'...", func.name);
    let liveness = liveness::analyze_liveness(func);

    // 2. Interval Analysis (Hint for Z3)
    tracing::info!(target: "lila::verify", "Running interval analysis for '{}'...", func.name);
    let analysis_results = interval::analyze(func);

    // 3. Logic Verification with Backend
    tracing::info!(target: "lila::verify", "Starting verification for '{}'...", func.name);
    let ctx = Context::thread_local();
    let solver = Solver::new();

    let mut backend = z3_backend::Z3Backend::new(&ctx, &solver);

    backend.set_timeout(timeout_ms);

    verify_with_context(&mut backend, func, &analysis_results, liveness, uid)
}
