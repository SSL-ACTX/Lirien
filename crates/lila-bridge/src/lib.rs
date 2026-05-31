pub mod bridge;
pub mod cache;

use pyo3::prelude::*;

#[pymodule]
fn lila_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ::lila_core::init_diagnostics();
    m.add_function(wrap_pyfunction!(bridge::verify_and_compile, m)?)?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(configure_tracing, m)?)?;
    Ok(())
}

#[pyfunction]
fn set_log_level(level: String) -> PyResult<()> {
    ::lila_core::set_log_level(&level)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok(())
}

#[pyfunction]
fn configure_tracing(config: std::collections::HashMap<String, String>) -> PyResult<()> {
    ::lila_core::configure_tracing(config)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok(())
}
