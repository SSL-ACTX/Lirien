use super::{get_all_cl_values, CodegenContext, LoweringError};
use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};
use lirien_ir::ir::{Type as SsaType, Value};

pub fn lower<M: Module>(
    ctx: &mut CodegenContext<M>,
    dest: Value,
    func: &str,
    args: &[Value],
) -> Result<(), LoweringError> {
    let is_math_intrinsic = matches!(
        func,
        "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "exp"
            | "log"
            | "log10"
            | "pow"
            | "floor"
            | "ceil"
            | "trunc"
            | "nearest"
    );

    if is_math_intrinsic {
        let mut sig = ctx.module.make_signature();
        for _ in args {
            sig.params.push(AbiParam::new(types::F64));
        }
        let ret_ty = ctx.ssa_func.get_type(dest);
        if ret_ty.is_float() {
            sig.returns.push(AbiParam::new(types::F64));
        } else {
            sig.returns
                .push(AbiParam::new(super::super::translate_type(&ret_ty)));
        }

        let callee = ctx.module.declare_function(func, Linkage::Import, &sig)?;
        let local_callee = ctx.module.declare_func_in_func(callee, ctx.builder.func);

        let mut arg_vals = Vec::new();
        for arg in args {
            let cl_val = get_all_cl_values(ctx, arg)[0];
            let arg_ty = ctx.ssa_func.get_type(*arg);
            if arg_ty.is_float32() {
                arg_vals.push(ctx.builder.ins().fpromote(types::F64, cl_val));
            } else {
                arg_vals.push(cl_val);
            }
        }

        let call = ctx.builder.ins().call(local_callee, &arg_vals);
        let res = ctx.builder.inst_results(call)[0];

        if ret_ty.is_float32() {
            let demoted = ctx.builder.ins().fdemote(types::F32, res);
            ctx.values.insert(dest, demoted);
        } else {
            ctx.values.insert(dest, res);
        }

        return Ok(());
    }

    let mut arg_types = Vec::new();
    for arg in args {
        arg_types.push(ctx.ssa_func.get_type(*arg));
    }
    let ret_ty = ctx.ssa_func.get_type(dest);

    let (cl_sig, is_sret, is_register_composite_ret) =
        super::build_cranelift_signature(ctx.ssa_func, &arg_types, &ret_ty, false, ctx.module);

    let callee = ctx
        .module
        .declare_function(func, Linkage::Import, &cl_sig)?;
    let local_callee = ctx.module.declare_func_in_func(callee, ctx.builder.func);

    let mut arg_vals = Vec::new();
    let mut sret_slot = None;

    if is_sret {
        let size = ret_ty.size(&ctx.ssa_func.struct_layouts);
        let slot = ctx
            .builder
            .create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size as u32));
        let addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
        arg_vals.push(addr);
        sret_slot = Some(slot);
    }

    for arg in args {
        arg_vals.extend(get_all_cl_values(ctx, arg));
    }

    let call = ctx.builder.ins().call(local_callee, &arg_vals);

    if is_sret {
        let addr = ctx
            .builder
            .ins()
            .stack_addr(types::I64, sret_slot.unwrap(), 0);
        ctx.values.insert(dest, addr);
    } else if is_register_composite_ret {
        let res_vals = ctx.builder.inst_results(call).to_vec();
        ctx.unpacked_values.insert(dest, res_vals);
    } else if ret_ty != SsaType::Unknown {
        let res = ctx.builder.inst_results(call)[0];
        ctx.values.insert(dest, res);
    }

    Ok(())
}
