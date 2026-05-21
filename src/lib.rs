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
    Ok(())
}
