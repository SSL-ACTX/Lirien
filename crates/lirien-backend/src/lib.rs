pub mod cranelift;

use lirien_ir::ir::Function as SsaFunction;

pub fn compile(ssa_func: &SsaFunction) -> Result<usize, String> {
    cranelift::compile(ssa_func)
}
