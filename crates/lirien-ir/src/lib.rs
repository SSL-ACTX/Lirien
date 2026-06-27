//! # Lirien IR
//!
//! This crate defines the Intermediate Representation (IR) for Lirien,
//! which is structured as a strict Static Single Assignment (SSA) control flow graph.
//! It also provides the transformation pipeline from Python AST (rustpython_ast) to IR,
//! optimizations, static analysis passes, and a global function registry.

pub mod analysis;
pub mod builder;
pub mod ir;
pub mod optimization;
pub mod registry;

use self::builder::error::BuilderError;
use self::builder::CFGBuilder;
use self::ir::Function;
use rustpython_ast as ast;
use std::collections::HashMap;
use tracing::info;

/// Transforms a Python AST suite (representing a module or a function definition)
/// into a flattened collection of Lirien IR [`Function`]s.
///
/// This performs:
/// 1. Control Flow Graph (CFG) building via [`CFGBuilder`].
/// 2. Code flattening and SSA generation.
/// 3. Optimization passes (e.g. Dead Code Elimination, type propagation).
///
/// The resulting functions are sorted and returned, with the main entry-point
/// function guaranteed to be the last element of the returned vector.
///
/// # Errors
/// Returns a [`BuilderError`] if AST compilation or validation fails.
pub fn transform(
    name: String,
    suite: ast::Suite,
    struct_layouts: HashMap<String, Vec<(String, String)>>,
    enum_layouts: HashMap<String, Vec<(String, String)>>,
    type_aliases: HashMap<String, String>,
    named_tuple_layouts: HashMap<String, Vec<(String, String)>>,
    typed_dict_layouts: HashMap<String, Vec<(String, String)>>,
) -> Result<Vec<Function>, BuilderError> {
    info!(target: "lirien::ssa", "Transforming AST to IR for '{}'...", name);

    let mut builder = CFGBuilder::new(
        name,
        struct_layouts,
        enum_layouts,
        type_aliases,
        named_tuple_layouts,
        typed_dict_layouts,
    );
    builder.build(suite)?;

    let mut main_func = builder.func;
    let mut lambdas = builder.lambdas;

    // Optimization
    optimization::optimize(&mut main_func);
    for lambda in &mut lambdas {
        optimization::optimize(lambda);
    }

    let mut result = Vec::new();
    lambdas.reverse();
    result.extend(lambdas);
    result.push(main_func);

    for func in &result {
        func.dump();
    }

    Ok(result)
}
