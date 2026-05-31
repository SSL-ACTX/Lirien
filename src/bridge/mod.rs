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
    enum_layouts: HashMap<String, Vec<(String, String)>>,
    type_aliases: HashMap<String, String>,
) -> PyResult<usize> {
    info!(target: "lila::bridge", "Received source for '{}'", func_name);
    debug!(target: "lila::bridge", "Struct layouts: {:?}", struct_layouts);
    debug!(target: "lila::bridge", "Enum layouts: {:?}", enum_layouts);
    debug!(target: "lila::bridge", "Type aliases: {:?}", type_aliases);

    let ast = ast::Suite::parse(&source, "<lila>").map_err(|e| {
        PyErr::new::<pyo3::exceptions::PySyntaxError, _>(format!("Parse error: {}", e))
    })?;

    debug!(target: "lila::bridge", "AST parsed successfully. Starting SSA transformation...");

    let ssa_list = crate::ssa::transform(
        func_name.clone(),
        ast,
        struct_layouts,
        enum_layouts,
        type_aliases,
    )
    .map_err(|e| {
        eprintln!(
            "[Lila Bridge Error] SSA Transform failed for {}: {}",
            func_name, e
        );
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)
    })?;

    let mut main_code_ptr = 0;

    for ssa in ssa_list {
        info!(target: "lila::bridge", "Processing SSA for '{}'...", ssa.name);

        crate::verification::verify(&ssa)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        info!(target: "lila::bridge", "Verification complete for '{}'", ssa.name);

        let mut arg_types = Vec::new();
        let mut arg_refinements = HashMap::new();
        for i in 0..ssa.arg_count {
            let v = crate::ssa::ir::Value(i);
            arg_types.push(ssa.get_type(v));
            if let Some(ref_str) = ssa.refinements.get(&v) {
                arg_refinements.insert(i, ref_str.clone());
            }
        }
        let return_type = ssa.return_type.clone();
        let return_refinement = ssa.ret_refinement.clone();

        let code_ptr = crate::backend::compile(&ssa)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        {
            let mut registry = GLOBAL_REGISTRY.lock().unwrap();
            registry.register(FunctionSignature {
                name: ssa.name.clone(),
                arg_types,
                arg_refinements,
                return_type,
                return_refinement,
                pointer: code_ptr,
            });
        }

        if ssa.name == func_name {
            main_code_ptr = code_ptr;
        }

        info!(
            target: "lila::bridge",
            "Backend compilation complete for '{}', ptr: {:x}",
            ssa.name, code_ptr
        );
    }

    Ok(main_code_ptr)
}
