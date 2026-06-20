use crate::cache;
use lirien_ir::registry::{FunctionSignature, GLOBAL_REGISTRY};
use pyo3::prelude::*;
use rustpython_parser::ast;
use rustpython_parser::Parse;
use std::collections::HashMap;
use tracing::{debug, info};

#[pyfunction]
#[pyo3(signature = (source, func_name, struct_layouts, enum_layouts, type_aliases, named_tuple_layouts=HashMap::new(), typed_dict_layouts=HashMap::new(), timeout_ms=5000, verify=true))]
#[allow(clippy::too_many_arguments)]
pub fn verify_and_compile(
    source: String,
    func_name: String,
    struct_layouts: HashMap<String, Vec<(String, String)>>,
    enum_layouts: HashMap<String, Vec<(String, String)>>,
    type_aliases: HashMap<String, String>,
    named_tuple_layouts: HashMap<String, Vec<(String, String)>>,
    typed_dict_layouts: HashMap<String, Vec<(String, String)>>,
    timeout_ms: u32,
    verify: bool,
) -> PyResult<usize> {
    info!(target: "lirien::bridge", "Received source for '{}' (verify={})", func_name, verify);
    debug!(target: "lirien::bridge", "Struct layouts: {:?}", struct_layouts);
    debug!(target: "lirien::bridge", "Enum layouts: {:?}", enum_layouts);
    debug!(target: "lirien::bridge", "Named Tuple layouts: {:?}", named_tuple_layouts);
    debug!(target: "lirien::bridge", "Typed Dict layouts: {:?}", typed_dict_layouts);
    debug!(target: "lirien::bridge", "Type aliases: {:?}", type_aliases);
    debug!(target: "lirien::bridge", "Timeout ms: {}", timeout_ms);

    let cache_hash = cache::compute_hash_full(
        &source,
        &func_name,
        &struct_layouts,
        &enum_layouts,
        &type_aliases,
        &named_tuple_layouts,
        &typed_dict_layouts,
    );

    // Incorporate the verify flag into the hash, so verified vs non-verified compilations don't conflict.
    use std::hash::{Hash, Hasher};
    let mut hasher = seahash::SeaHasher::new();
    cache_hash.hash(&mut hasher);
    verify.hash(&mut hasher);
    let cache_hash = hasher.finish();

    // Check L1 Native Code Cache first
    if let Some(entries) = cache::native_cache_lookup(cache_hash) {
        info!(target: "lirien::bridge", "L1 Native Cache HIT for '{}' (hash: {:016x}). Skipping compilation.", func_name, cache_hash);
        let mut main_code_ptr = 0;
        let mut registry = GLOBAL_REGISTRY.lock().unwrap();
        for entry in entries {
            registry.register(FunctionSignature {
                name: entry.name.clone(),
                arg_types: entry.arg_types,
                arg_refinements: entry.arg_refinements,
                return_type: entry.return_type,
                return_refinement: entry.return_refinement,
                pointer: entry.pointer,
            });
            if entry.name == func_name {
                main_code_ptr = entry.pointer;
            }
        }
        return Ok(main_code_ptr);
    }

    let ssa_list = if let Some(cached_funcs) = cache::load_ir(cache_hash) {
        info!(target: "lirien::bridge", "IR Cache HIT for '{}' (hash: {:016x}). Skipping AST & Z3 verification.", func_name, cache_hash);
        cached_funcs
    } else {
        info!(target: "lirien::bridge", "IR Cache MISS for '{}' (hash: {:016x}). Proceeding with full verification pipeline.", func_name, cache_hash);
        let ast = ast::Suite::parse(&source, "<lirien>").map_err(|e| {
            PyErr::new::<pyo3::exceptions::PySyntaxError, _>(format!("Parse error: {}", e))
        })?;

        debug!(target: "lirien::bridge", "AST parsed successfully. Starting SSA transformation...");

        let mut funcs = lirien_ir::transform(
            func_name.clone(),
            ast,
            struct_layouts,
            enum_layouts,
            type_aliases,
            named_tuple_layouts,
            typed_dict_layouts,
        )
        .map_err(|e| {
            eprintln!(
                "[Lirien Bridge Error] SSA Transform failed for {}: {}",
                func_name, e
            );
            cache::invalidate(cache_hash);
            cache::native_cache_invalidate(cache_hash);
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        if verify {
            for ssa in &mut funcs {
                info!(target: "lirien::bridge", "Processing SSA for '{}'...", ssa.name);
                match lirien_verify::verify(ssa, timeout_ms) {
                    Ok(inferred) => {
                        if let Some(inf) = inferred {
                            ssa.ret_refinement = Some(inf);
                        }
                    }
                    Err(e) => {
                        cache::invalidate(cache_hash);
                        cache::native_cache_invalidate(cache_hash);
                        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e));
                    }
                }
                info!(target: "lirien::bridge", "Verification complete for '{}'", ssa.name);
            }
        } else {
            info!(target: "lirien::bridge", "Skipping Z3 verification for '{}' because verify=false", func_name);
        }

        // Cache the verified IR
        cache::save_ir(cache_hash, &funcs);

        funcs
    };

    let dependencies = cache::collect_dependencies(&ssa_list);
    let mut main_code_ptr = 0;
    let mut native_entries = Vec::new();

    for ssa in ssa_list {
        let mut arg_types = Vec::new();
        let mut arg_refinements = HashMap::new();
        for i in 0..ssa.arg_count {
            let v = lirien_ir::ir::Value(i);
            arg_types.push(ssa.get_type(v));
            if let Some(ref_str) = ssa.refinements.get(&v) {
                arg_refinements.insert(i, ref_str.clone());
            }
        }
        let return_type = ssa.return_type.clone();
        let return_refinement = ssa.ret_refinement.clone();

        let code_ptr = match lirien_backend::compile(&ssa) {
            Ok(ptr) => ptr,
            Err(e) => {
                cache::invalidate(cache_hash);
                cache::native_cache_invalidate(cache_hash);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e));
            }
        };

        {
            let mut registry = GLOBAL_REGISTRY.lock().unwrap();
            registry.register(FunctionSignature {
                name: ssa.name.clone(),
                arg_types: arg_types.clone(),
                arg_refinements: arg_refinements.clone(),
                return_type: return_type.clone(),
                return_refinement: return_refinement.clone(),
                pointer: code_ptr,
            });
        }

        native_entries.push(cache::NativeCacheEntry {
            name: ssa.name.clone(),
            pointer: code_ptr,
            arg_types,
            arg_refinements,
            return_type,
            return_refinement,
        });

        if ssa.name == func_name {
            main_code_ptr = code_ptr;
        }

        info!(
            target: "lirien::bridge",
            "Backend compilation complete for '{}', ptr: {:x}",
            ssa.name, code_ptr
        );
    }

    cache::native_cache_store(cache_hash, native_entries, dependencies);

    Ok(main_code_ptr)
}
