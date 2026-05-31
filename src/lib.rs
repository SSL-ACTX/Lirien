pub mod backend;
pub mod bridge;
pub mod diagnostics;
pub mod ssa;
pub mod verification;

use pyo3::prelude::*;

#[pymodule]
fn lila_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    diagnostics::init_diagnostics();
    m.add_function(wrap_pyfunction!(bridge::verify_and_compile, m)?)?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    Ok(())
}

#[pyfunction]
fn set_log_level(level: String) -> PyResult<()> {
    diagnostics::set_log_level(&level)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok(())
}
