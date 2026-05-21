pub mod constant_folding;
pub mod dce;
pub mod type_propagation;

use super::ir::Function;
use tracing::info;

pub fn optimize(func: &mut Function) {
    info!(target: "lila::ssa::opt", "Optimizing IR for '{}'...", func.name);

    // Type Propagation Pass (crucial for loops)
    type_propagation::propagate_types(func);

    // Constant Folding Pass
    constant_folding::fold_constants(func);

    // Dead Code Elimination Pass
    dce::eliminate_dead_code(func);
}
