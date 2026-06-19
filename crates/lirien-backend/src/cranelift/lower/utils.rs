use cranelift::prelude::*;
use cranelift_module::Module;
use lirien_ir::ir::Value as SsaValue;
use super::CodegenContext;

pub fn get_val(values: &std::collections::HashMap<SsaValue, Value>, val: &SsaValue) -> Value {
    *values
        .get(val)
        .unwrap_or_else(|| panic!("Value v{} not found", val.0))
}

pub fn get_all_cl_values<M: Module>(ctx: &CodegenContext<M>, val: &SsaValue) -> Vec<Value> {
    let ty = ctx.ssa_func.get_type(*val);
    match ty {
        lirien_ir::ir::Type::NamedTuple(_) | lirien_ir::ir::Type::Tuple(_) => ctx
            .unpacked_values
            .get(val)
            .cloned()
            .unwrap_or_else(|| vec![get_val(&ctx.values, val)]),
        lirien_ir::ir::Type::Buffer(_) => vec![
            get_val(&ctx.values, val),
            *ctx.buffer_lengths
                .get(val)
                .unwrap_or_else(|| panic!("Length for v{} not found", val.0)),
        ],
        lirien_ir::ir::Type::Tensor(_, ref _dims) => {
            let mut res = vec![get_val(&ctx.values, val)];
            res.extend(
                ctx.tensor_dims
                    .get(val)
                    .unwrap_or_else(|| panic!("Dims for v{} not found", val.0)),
            );
            res
        }
        _ => vec![get_val(&ctx.values, val)],
    }
}

pub fn get_len(lengths: &std::collections::HashMap<SsaValue, Value>, val: &SsaValue) -> Value {
    *lengths
        .get(val)
        .unwrap_or_else(|| panic!("Length for v{} not found", val.0))
}
