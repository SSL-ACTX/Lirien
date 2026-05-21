pub mod cranelift;

use crate::ssa::ir::Function as SsaFunction;

pub fn compile(ssa_func: &SsaFunction) -> Result<usize, String> {
    cranelift::compile(ssa_func)
}
