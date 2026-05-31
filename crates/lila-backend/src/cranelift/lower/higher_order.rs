use super::{get_val, translate_type, CodegenContext};
use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};
use lila_ir::ir::{Type as SsaType, Value as SsaValue};

pub fn lower<M: Module>(
    ctx: &mut CodegenContext<M>,
    dest: SsaValue,
    fn_ptr: SsaValue,
    args: &[SsaValue],
) -> Result<(), String> {
    let fn_ty = ctx.ssa_func.get_type(fn_ptr);
    let (arg_types, ret_ty, is_closure) = match fn_ty {
        SsaType::FnPointer(ref args, ref ret) => (args.clone(), (**ret).clone(), false),
        SsaType::Closure(_, ref args, ref ret) => (args.clone(), (**ret).clone(), true),
        _ => {
            // Fallback for unknown types (treat as i64 args and i64 return)
            let mut fallback_args = Vec::new();
            for _ in args {
                fallback_args.push(SsaType::I64);
            }
            (fallback_args, SsaType::I64, false)
        }
    };

    let mut sig = ctx.module.make_signature();
    let mut is_sret = false;

    if let SsaType::Tuple(_) | SsaType::Struct(_) = ret_ty {
        sig.params.push(AbiParam::new(types::I64)); // sret pointer
        is_sret = true;
    }

    if is_closure {
        sig.params.push(AbiParam::new(types::I64)); // context pointer (closure itself)
    }

    for arg_ty in &arg_types {
        sig.params.push(AbiParam::new(translate_type(arg_ty)));
        if let SsaType::Buffer(_) = arg_ty {
            sig.params.push(AbiParam::new(types::I64));
        }
    }

    if !is_sret && ret_ty != SsaType::Unknown {
        sig.returns.push(AbiParam::new(translate_type(&ret_ty)));
    }

    let cl_fn_val = get_val(&ctx.values, &fn_ptr);
    let cl_fn_ptr = if is_closure {
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::new(), cl_fn_val, 0)
    } else {
        cl_fn_val
    };

    let mut arg_vals = Vec::new();
    let mut sret_addr = None;

    if is_sret {
        let size = ret_ty.size(&ctx.ssa_func.struct_layouts);
        let slot = ctx
            .builder
            .create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size as u32));
        let addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
        arg_vals.push(addr);
        sret_addr = Some(addr);
    }

    if is_closure {
        arg_vals.push(cl_fn_val); // Pass closure ptr as context
    }

    for arg in args {
        let arg_val = get_val(&ctx.values, arg);
        arg_vals.push(arg_val);
        if let Some(len) = ctx.buffer_lengths.get(arg) {
            arg_vals.push(*len);
        }
    }

    let sig_ref = ctx.builder.import_signature(sig);
    let call = ctx
        .builder
        .ins()
        .call_indirect(sig_ref, cl_fn_ptr, &arg_vals);

    if is_sret {
        ctx.values.insert(dest, sret_addr.unwrap());
    } else if ret_ty != SsaType::Unknown {
        let res = ctx.builder.inst_results(call)[0];
        ctx.values.insert(dest, res);
    } else {
        // Void-like return
        let zero = ctx.builder.ins().iconst(types::I64, 0);
        ctx.values.insert(dest, zero);
    }

    Ok(())
}

pub fn lower_lambda<M: Module>(
    ctx: &mut CodegenContext<M>,
    dest: SsaValue,
    func_name: &str,
    captures: &[SsaValue],
) -> Result<(), String> {
    let mut sig = ctx.module.make_signature();
    let fn_ty = ctx.ssa_func.get_type(dest);

    let (arg_types, ret_ty) = if let SsaType::Closure(_, args, ret) = fn_ty {
        (args, *ret)
    } else {
        return Err("Lambda must result in a Closure type".to_string());
    };

    // Closure Signature: (ctx_ptr, ...args) -> ret
    sig.params.push(AbiParam::new(types::I64)); // ctx_ptr
    for arg_ty in &arg_types {
        sig.params.push(AbiParam::new(translate_type(arg_ty)));
        if let SsaType::Buffer(_) = arg_ty {
            sig.params.push(AbiParam::new(types::I64));
        }
    }
    if ret_ty != SsaType::Unknown {
        sig.returns.push(AbiParam::new(translate_type(&ret_ty)));
    }
    ctx.builder.import_signature(sig.clone());
    let callee = ctx
        .module
        .declare_function(func_name, Linkage::Import, &sig)
        .unwrap();
    let local_callee = ctx.module.declare_func_in_func(callee, ctx.builder.func);
    let fn_ptr = ctx.builder.ins().func_addr(types::I64, local_callee);

    // Closure Layout: [0..8]: fn_ptr, [8..N]: captures
    let mut total_size = 8;
    let mut capture_offsets = Vec::new();
    for capture in captures {
        let ty = ctx.ssa_func.get_type(*capture);
        let align = ty.align(&ctx.ssa_func.struct_layouts);
        total_size = (total_size + align - 1) & !(align - 1);
        capture_offsets.push(total_size);
        total_size += ty.size(&ctx.ssa_func.struct_layouts);
    }

    // Use malloc for closure allocation
    let mut malloc_sig = ctx.module.make_signature();
    malloc_sig.params.push(AbiParam::new(types::I64)); // size
    malloc_sig.returns.push(AbiParam::new(types::I64)); // ptr
    ctx.builder.import_signature(malloc_sig.clone());
    let malloc_func = ctx
        .module
        .declare_function("malloc", Linkage::Import, &malloc_sig)
        .unwrap();
    let local_malloc = ctx
        .module
        .declare_func_in_func(malloc_func, ctx.builder.func);

    let size_val = ctx.builder.ins().iconst(types::I64, total_size as i64);
    let malloc_call = ctx.builder.ins().call(local_malloc, &[size_val]);
    let closure_ptr = ctx.builder.inst_results(malloc_call)[0];

    // Store fn_ptr
    ctx.builder
        .ins()
        .store(MemFlags::new(), fn_ptr, closure_ptr, 0);

    // Store captures
    for (i, capture) in captures.iter().enumerate() {
        let val = get_val(&ctx.values, capture);
        let offset = capture_offsets[i];
        let ty = ctx.ssa_func.get_type(*capture);

        if ty.is_composite() {
            let dest_with_offset = ctx.builder.ins().iadd_imm(closure_ptr, offset as i64);
            super::copy_memory(
                &mut ctx.builder,
                val,
                dest_with_offset,
                ty.size(&ctx.ssa_func.struct_layouts),
            );
        } else {
            ctx.builder
                .ins()
                .store(MemFlags::new(), val, closure_ptr, offset as i32);
        }
    }

    ctx.values.insert(dest, closure_ptr);
    Ok(())
}
