use super::{get_all_cl_values, CodegenContext};
use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};
use lila_ir::ir::{Type as SsaType, Value};

pub fn lower<M: Module>(
    ctx: &mut CodegenContext<M>,
    dest: Value,
    func: &str,
    args: &[Value],
) -> Result<(), String> {
    let mut arg_types = Vec::new();
    for arg in args {
        arg_types.push(ctx.ssa_func.get_type(*arg));
    }
    let ret_ty = ctx.ssa_func.get_type(dest);

    let (cl_sig, is_sret, is_register_composite_ret) = super::build_cranelift_signature(
        ctx.ssa_func,
        &arg_types,
        &ret_ty,
        false,
        ctx.module,
    );

    let callee = ctx
        .module
        .declare_function(func, Linkage::Import, &cl_sig)
        .map_err(|e| e.to_string())?;
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
        let addr = ctx.builder.ins().stack_addr(types::I64, sret_slot.unwrap(), 0);
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
