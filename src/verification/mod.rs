pub mod permissions;
pub mod refinement_parser;
pub mod z3;

use self::z3::verify_with_context;
use crate::ssa::analysis::interval;
use crate::ssa::analysis::liveness;
use crate::ssa::ir::Function;
use crate::verification::permissions::PermissionVerifier;
use ::z3::{Context, Solver};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

static VERIFY_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn verify(func: &Function) -> Result<(), String> {
    info!(target: "lila::verify", "Verifying function '{}'...", func.name);

    let uid = VERIFY_COUNT.fetch_add(1, Ordering::SeqCst);

    // 1. Liveness Analysis for Fractional Permissions
    let liveness = liveness::analyze_liveness(func);

    // 2. Setup Fractional Permission Verifier
    let mut perm_verifier = PermissionVerifier::new(func);
    perm_verifier.set_uid(uid);

    // 3. Interval Analysis
    let analysis_results = interval::analyze(func);

    // 4. Logic Verification with Z3
    let ctx = Context::thread_local();
    let solver = Solver::new();

    verify_with_context(
        &ctx,
        &solver,
        func,
        analysis_results,
        liveness,
        perm_verifier,
        uid,
    )
}
