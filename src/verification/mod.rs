pub mod borrow_checker;
pub mod refinement_parser;
pub mod z3;

use self::z3::verify_with_context;
use crate::ssa::analysis::interval;
use crate::ssa::ir::Function;
use ::z3::{Config, Context, Solver};
use tracing::info;

pub fn verify(func: &Function) -> Result<(), String> {
    info!(target: "lila::verify", "Verifying function '{}'...", func.name);

    // 1. Borrow Checking
    let borrow_checker = borrow_checker::BorrowChecker::new(func);
    borrow_checker.check()?;

    // 2. Interval Analysis
    let analysis_results = interval::analyze(func);

    // 3. Logic Verification with Z3
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);

    verify_with_context(&ctx, &solver, func, analysis_results)
}
