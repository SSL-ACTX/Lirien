//! # Lirien Backend
//!
//! This crate implements the JIT compiler machine code backend using Cranelift.
//! It lowers the SSA IR graph into native target assembly instructions and returns execution pointers.

pub mod cranelift;

use lirien_ir::ir::Function as SsaFunction;

/// Compiles a Lirien JIT function IR structure into machine code via Cranelift.
///
/// Returns a raw memory pointer (`usize`) to the generated executable function.
///
/// # Errors
/// Returns an error string if machine lowering or symbol resolution fails.
pub fn compile(ssa_func: &SsaFunction) -> Result<usize, String> {
    cranelift::compile(ssa_func)
}
