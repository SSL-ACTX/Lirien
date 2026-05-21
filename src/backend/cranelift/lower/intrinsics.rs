use super::{get_val, translate_type, CodegenContext};
use crate::ssa::ir::{Type as SsaType, Value as SsaValue};
use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};

pub fn lower<M: Module>(
    ctx: &mut CodegenContext<M>,
    dest: SsaValue,
    func: &str,
    args: &[SsaValue],
) -> Result<(), String> {
    if func == "make_tuple" {
        let elt_types = match ctx.ssa_func.get_type(dest) {
            SsaType::Tuple(t) => t,
            _ => panic!("make_tuple must return a Tuple type"),
        };
        let total_size = ctx
            .ssa_func
            .get_type(dest)
            .size(&ctx.ssa_func.struct_layouts);
        let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            total_size as u32,
        ));
        let base_ptr = ctx.builder.ins().stack_addr(types::I64, slot, 0);

        let mut offset = 0;
        for (arg, f_ty) in args.iter().zip(elt_types.iter()) {
            let align = f_ty.align(&ctx.ssa_func.struct_layouts);
            offset = (offset + align - 1) & !(align - 1);

            let arg_val = get_val(&ctx.values, arg);
            let cl_ty = translate_type(f_ty);

            let val_to_store = if ctx.builder.func.dfg.value_type(arg_val) != cl_ty {
                if cl_ty.is_int() && ctx.builder.func.dfg.value_type(arg_val).is_int() {
                    ctx.builder.ins().ireduce(cl_ty, arg_val)
                } else {
                    arg_val
                }
            } else {
                arg_val
            };

            ctx.builder
                .ins()
                .store(MemFlags::new(), val_to_store, base_ptr, offset as i32);
            offset += f_ty.size(&ctx.ssa_func.struct_layouts);
        }
        ctx.values.insert(dest, base_ptr);
        return Ok(());
    }

    if func == "sin" || func == "cos" || func == "pow" {
        let mut sig = ctx.module.make_signature();
        if func == "pow" {
            sig.params.push(AbiParam::new(types::F64));
            sig.params.push(AbiParam::new(types::F64));
        } else {
            sig.params.push(AbiParam::new(types::F64));
        }
        sig.returns.push(AbiParam::new(types::F64));

        let callee = ctx
            .module
            .declare_function(func, Linkage::Import, &sig)
            .unwrap();
        let local_callee = ctx.module.declare_func_in_func(callee, ctx.builder.func);

        let mut arg_vals = Vec::new();
        for arg in args {
            arg_vals.push(get_val(&ctx.values, arg));
        }

        let call = ctx.builder.ins().call(local_callee, &arg_vals);
        let res = ctx.builder.inst_results(call)[0];
        ctx.values.insert(dest, res);
        return Ok(());
    }

    let mut cl_sig = ctx.module.make_signature();
    let registry = crate::bridge::registry::GLOBAL_REGISTRY.lock().unwrap();

    if let Some(sig) = registry.get(func) {
        for arg_ty in &sig.arg_types {
            cl_sig.params.push(AbiParam::new(translate_type(arg_ty)));
            if let SsaType::Buffer(_) = arg_ty {
                cl_sig.params.push(AbiParam::new(types::I64));
            }
        }
        cl_sig
            .returns
            .push(AbiParam::new(translate_type(&sig.return_type)));
    } else {
        for arg in args {
            let ty = ctx.ssa_func.get_type(*arg);
            cl_sig.params.push(AbiParam::new(translate_type(&ty)));
            if let SsaType::Buffer(_) = ty {
                cl_sig.params.push(AbiParam::new(types::I64));
            }
        }
        let ret_ty = ctx.ssa_func.get_type(dest);
        cl_sig.returns.push(AbiParam::new(translate_type(&ret_ty)));
    }

    let callee = ctx
        .module
        .declare_function(func, Linkage::Import, &cl_sig)
        .map_err(|e| e.to_string())?;
    let local_callee = ctx.module.declare_func_in_func(callee, ctx.builder.func);

    let mut arg_vals = Vec::new();
    for arg in args {
        let arg_val = get_val(&ctx.values, arg);
        arg_vals.push(arg_val);
        if let Some(len) = ctx.buffer_lengths.get(arg) {
            arg_vals.push(*len);
        }
    }

    let call = ctx.builder.ins().call(local_callee, &arg_vals);
    let res = ctx.builder.inst_results(call)[0];
    ctx.values.insert(dest, res);
    Ok(())
}
