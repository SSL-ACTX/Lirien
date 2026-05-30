pub mod permissions;
pub mod refinement_parser;
pub mod z3;

use self::z3::verify_with_context;
use crate::ssa::analysis::interval;
use crate::ssa::analysis::liveness;
use crate::ssa::ir::Function;
use crate::verification::permissions::PermissionVerifier;
use ::z3::{Context, Params, Solver};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

static VERIFY_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn verify(func: &Function) -> Result<(), String> {
    info!(target: "lila::verify", "Verifying function '{}'...", func.name);

    let uid = VERIFY_COUNT.fetch_add(1, Ordering::SeqCst);

    // 1. Liveness Analysis for Fractional Permissions
    tracing::info!(target: "lila::verify", "Running liveness analysis for '{}'...", func.name);
    let liveness = liveness::analyze_liveness(func);

    // 2. Setup Fractional Permission Verifier
    let mut perm_verifier = PermissionVerifier::new(func);
    perm_verifier.set_uid(uid);

    // 3. Interval Analysis
    tracing::info!(target: "lila::verify", "Running interval analysis for '{}'...", func.name);
    let analysis_results = interval::analyze(func);

    // 4. Logic Verification with Z3
    tracing::info!(target: "lila::verify", "Starting Z3 verification for '{}'...", func.name);
    let ctx = Context::thread_local();
    let solver = Solver::new();

    // Set a 5-second timeout for the entire verification process
    let mut params = Params::new();
    params.set_u32("timeout", 5000);
    solver.set_params(&params);

    verify_with_context(
        &ctx,
        &solver,
        func,
        &analysis_results,
        liveness,
        perm_verifier,
        uid,
    )
}
