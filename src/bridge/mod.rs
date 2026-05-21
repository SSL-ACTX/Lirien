pub mod registry;

use pyo3::prelude::*;
use registry::{FunctionSignature, GLOBAL_REGISTRY};
use rustpython_parser::ast;
use rustpython_parser::Parse;
use std::collections::HashMap;
use tracing::{debug, info};

#[pyfunction]
pub fn verify_and_compile(
    source: String,
    func_name: String,
    struct_layouts: HashMap<String, Vec<(String, String)>>,
    type_aliases: HashMap<String, String>,
) -> PyResult<usize> {
    info!(target: "lila::bridge", "Received source for '{}'", func_name);
    debug!(target: "lila::bridge", "Struct layouts: {:?}", struct_layouts);
    debug!(target: "lila::bridge", "Type aliases: {:?}", type_aliases);

    // Phase 1: Parsing
    let ast = ast::Suite::parse(&source, "<lila>").map_err(|e| {
        PyErr::new::<pyo3::exceptions::PySyntaxError, _>(format!("Parse error: {}", e))
    })?;

    debug!(target: "lila::bridge", "AST parsed successfully. Starting SSA transformation...");

    // Phase 2: SSA
    let ssa = crate::ssa::transform(func_name.clone(), ast, struct_layouts, type_aliases)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    info!(target: "lila::bridge", "SSA Transformation complete for '{}'", func_name);

    // Phase 3: Verification
    crate::verification::verify(&ssa).map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    info!(target: "lila::bridge", "Verification complete for '{}'", func_name);

    // Capture argument types before we move SSA
    let mut arg_types = Vec::new();
    for i in 0..ssa.arg_count {
        arg_types.push(ssa.get_type(crate::ssa::ir::Value(i)));
    }
    let return_type = ssa.return_type.clone();

    // Phase 4: Backend
    let code_ptr =
        crate::backend::compile(&ssa).map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    // Register in global registry
    {
        let mut registry = GLOBAL_REGISTRY.lock().unwrap();
        registry.register(FunctionSignature {
            name: func_name.clone(),
            arg_types,
            return_type,
            pointer: code_ptr,
        });
    }

    info!(
        target: "lila::bridge",
        "Backend compilation complete for '{}', ptr: {:x}",
        func_name, code_ptr
    );

    Ok(code_ptr)
}
