pub mod constant_folding;
pub mod dce;
pub mod fusion;
pub mod inference;
pub mod type_propagation;

use super::ir::Function;
use tracing::debug;

pub fn optimize(func: &mut Function) {
    debug!(target: "lirien::ssa::opt", "Optimizing IR for '{}'...", func.name);

    // Type Propagation
    type_propagation::propagate_types(func);

    // Constant Folding
    constant_folding::fold_constants(func);

    // Tensor Kernel Fusion
    fusion::fuse_tensor_kernels(func);

    // Dead Code Elimination
    dce::eliminate_dead_code(func);

    // Embed Static Analysis Results as Liquid Types
    inference::embed_intervals(func);
}
