//! # Lirien Bridge
//!
//! PyO3-based extension module providing Python bindings for the Lirien verifier and compiler.
//! Exposes APIs to trigger formal verification, compilation, runtime diagnostics configuration,
//! and CPU feature detection.

pub mod bridge;
pub mod cache;

use pyo3::prelude::*;

/// The raw PyO3 module initialization function.
#[pymodule]
fn lirien_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ::lirien_core::init_diagnostics();
    m.add_function(wrap_pyfunction!(bridge::verify_and_compile, m)?)?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(configure_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(get_cpu_info, m)?)?;
    Ok(())
}

/// Returns target CPU properties such as architecture, ISA name, and supported SIMD features.
#[pyfunction]
fn get_cpu_info() -> PyResult<std::collections::HashMap<String, String>> {
    use cranelift_codegen::settings::Configurable;
    let mut flag_builder = cranelift_codegen::settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    let isa_builder = cranelift_native::builder()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let isa = isa_builder
        .finish(cranelift_codegen::settings::Flags::new(flag_builder))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut info = std::collections::HashMap::new();
    info.insert("arch".to_string(), isa.triple().architecture.to_string());
    info.insert("isa".to_string(), isa.name().to_string());

    let combined_flags = format!(
        "{} {}",
        isa.flags(),
        isa.isa_flags()
            .iter()
            .map(|f| f.name)
            .collect::<Vec<_>>()
            .join(" ")
    );
    let features: Vec<String> = [
        "has_neon",
        "has_asimd",
        "has_simd",
        "has_avx",
        "has_sse",
        "has_sse2",
        "has_sse3",
        "has_ssse3",
        "has_sse41",
        "has_sse42",
        "has_lse",
    ]
    .iter()
    .filter(|&&f| combined_flags.contains(f))
    .map(|&f| f.trim_start_matches("has_").to_string())
    .collect();

    info.insert("features".to_string(), features.join(", "));

    Ok(info)
}

/// Sets the runtime logging level filter.
///
/// Under the hood, this reloads the global `tracing-subscriber::EnvFilter`.
#[pyfunction]
fn set_log_level(level: String) -> PyResult<()> {
    ::lirien_core::set_log_level(&level)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok(())
}

/// Configures tracing options dynamically by mapping components to target log levels.
#[pyfunction]
fn configure_tracing(config: std::collections::HashMap<String, String>) -> PyResult<()> {
    ::lirien_core::configure_tracing(config)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok(())
}
