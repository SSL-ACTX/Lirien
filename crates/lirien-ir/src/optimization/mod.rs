//! Optimization pipeline for Lirien JIT IR.
//!
//! This module sequences and runs optimization passes on the generated SSA Control Flow
//! Graph, improving execution speed and cleaning up redundancy before machine code lowering.

pub mod constant_folding;
pub mod dce;
pub mod fusion;
pub mod inference;
pub mod type_propagation;

use super::ir::Function;
use tracing::debug;

/// Runs the standard pipeline of JIT optimization passes on a function.
///
/// The current optimization pipeline includes:
/// 1. Type Propagation (resolving and propagating type annotations through the SSA graph).
/// 2. Constant Folding (evaluating compile-time constant arithmetic).
/// 3. Tensor Kernel Fusion (merging adjacent tensor operations into fused kernels).
/// 4. Dead Code Elimination (removing unused SSA values and unreachable blocks).
/// 5. Interval Inference/Embedding (incorporating static analysis invariants).
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

