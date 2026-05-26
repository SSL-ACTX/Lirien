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
    type_aliases: HashMap<String, String>,
) -> Result<Function, String> {
    info!(target: "lila::ssa", "Transforming AST to IR for '{}'...", name);

    let mut builder = CFGBuilder::new(name, struct_layouts, type_aliases);
    builder.build(suite)?;

    let mut func = builder.func;

    // Optimization Passes
    optimization::optimize(&mut func);

    func.dump();

    Ok(func)
}
pub mod analysis;
