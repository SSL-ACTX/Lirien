pub mod builder;
pub mod ir;
pub mod optimization;

use self::builder::CFGBuilder;
use self::ir::Function;
use rustpython_ast as ast;
use std::collections::HashMap;
use tracing::info;

pub fn transform(
    name: String,
    suite: ast::Suite,
    struct_layouts: HashMap<String, Vec<(String, String)>>,
    enum_layouts: HashMap<String, Vec<(String, String)>>,
    type_aliases: HashMap<String, String>,
) -> Result<Vec<Function>, String> {
    info!(target: "lila::ssa", "Transforming AST to IR for '{}'...", name);

    let mut builder = CFGBuilder::new(name, struct_layouts, enum_layouts, type_aliases);
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
pub mod analysis;
