use crate::ssa::ir::{
    BlockId as SsaBlockId, Function as SsaFunction, InstructionKind, Type as SsaType,
    Value as SsaValue,
};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::{HashMap, HashSet};
use tracing::info;

pub mod lower;

pub struct CodegenContext<'a, M: Module> {
    pub builder: FunctionBuilder<'a>,
    pub module: &'a mut M,
    pub ssa_func: &'a SsaFunction,
    pub blocks: HashMap<SsaBlockId, Block>,
    pub values: HashMap<SsaValue, Value>,
    pub buffer_lengths: HashMap<SsaValue, Value>,
    pub is_tuple_return: bool,
    pub sret_ptr: Option<Value>,
}

pub fn translate_type(ty: &SsaType) -> types::Type {
    match ty {
        SsaType::I8 | SsaType::U8 | SsaType::Bool => types::I8,
        SsaType::I16 | SsaType::U16 => types::I16,
        SsaType::I32 | SsaType::U32 => types::I32,
        SsaType::I64
        | SsaType::U64
        | SsaType::Owned(_)
        | SsaType::Ref(_)
        | SsaType::Mut(_)
        | SsaType::FnPointer(_, _)
        | SsaType::Closure(_, _, _) => types::I64,
        SsaType::F32 => types::F32,
        SsaType::F64 => types::F64,
        SsaType::Array(_, _)
        | SsaType::Buffer(_)
        | SsaType::Struct(_)
        | SsaType::Enum(_)
        | SsaType::Tuple(_) => {
            types::I64 // Pointer
        }
        SsaType::Unknown => types::I64,
    }
}

pub fn compile(ssa_func: &SsaFunction) -> Result<usize, String> {
    info!(target: "lila::jit", "Compiling SSA to Machine Code via Cranelift for '{}'...", ssa_func.name);

    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| e.to_string())?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // Register all previously compiled Lila functions
    {
        let registry = crate::bridge::registry::GLOBAL_REGISTRY.lock().unwrap();
        for (name, sig) in registry.functions.iter() {
            jit_builder.symbol(name, sig.pointer as *const u8);
        }
    }

    // Special mock for tests
    extern "C" fn consume_owned_mock(_val: i64) -> i64 {
        1
    }
    jit_builder.symbol("consume_owned", consume_owned_mock as *const u8);

    // Link math intrinsics
    extern "C" {
        fn malloc(size: usize) -> *mut u8;
        fn sin(x: f64) -> f64;
        fn cos(x: f64) -> f64;
        fn pow(x: f64, y: f64) -> f64;
    }
    jit_builder.symbol("malloc", malloc as *const u8);
    jit_builder.symbol("sin", sin as *const u8);
    jit_builder.symbol("cos", cos as *const u8);
    jit_builder.symbol("pow", pow as *const u8);

    let mut module = JITModule::new(jit_builder);
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    let mut sig = module.make_signature();
    let mut is_buffer = Vec::new();
    let mut is_tuple_return = false;

    if let SsaType::Tuple(_) = ssa_func.return_type {
        sig.params.push(AbiParam::new(types::I64));
        is_tuple_return = true;
    }

    for i in 0..ssa_func.arg_count {
        let ty = ssa_func.get_type(SsaValue(i));
        if let SsaType::Buffer(_) = ty {
            sig.params.push(AbiParam::new(types::I64)); // Ptr
            sig.params.push(AbiParam::new(types::I64)); // Len
            is_buffer.push(true);
        } else {
            sig.params.push(AbiParam::new(translate_type(&ty)));
            is_buffer.push(false);
        }
    }

    if ssa_func.return_type != SsaType::Unknown && !is_tuple_return {
        sig.returns
            .push(AbiParam::new(translate_type(&ssa_func.return_type)));
    }

    let func_id = module
        .declare_function(&ssa_func.name, Linkage::Export, &sig)
        .map_err(|e| e.to_string())?;

    ctx.func.signature = sig;

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let mut blocks: HashMap<SsaBlockId, Block> = HashMap::new();

        for ssa_block in &ssa_func.blocks {
            blocks.insert(ssa_block.id, builder.create_block());
        }

        let entry_block = blocks[&ssa_func.entry_block];
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        let mut values = HashMap::new();
        let mut buffer_lengths = HashMap::new();

        // 1. Pre-declare all values to handle forward references within blocks
        // and cross-block references that aren't Phis (though valid SSA should handle this via dominance).
        // Since we are lowering to Cranelift which doesn't like forward refs for non-Phis,
        // we'll rely on RPO to visit definitions before uses.
        // However, we still need to pre-declare Phis.

        let mut param_idx = 0;
        let sret_ptr = if is_tuple_return {
            let ptr = builder.block_params(entry_block)[param_idx];
            param_idx += 1;
            Some(ptr)
        } else {
            None
        };

        for (i, &is_buf) in is_buffer.iter().enumerate() {
            let val = SsaValue(i);
            if is_buf {
                values.insert(val, builder.block_params(entry_block)[param_idx]);
                buffer_lengths.insert(val, builder.block_params(entry_block)[param_idx + 1]);
                param_idx += 2;
            } else {
                values.insert(val, builder.block_params(entry_block)[param_idx]);
                param_idx += 1;
            }
        }

        let mut cg_ctx = CodegenContext {
            builder,
            module: &mut module,
            ssa_func,
            blocks,
            values,
            buffer_lengths,
            is_tuple_return,
            sret_ptr,
        };

        // 2. Pre-declare all Phis across all blocks
        for ssa_block in &ssa_func.blocks {
            let current_cl_block = cg_ctx.blocks[&ssa_block.id];
            cg_ctx.builder.switch_to_block(current_cl_block);

            for inst in &ssa_block.instructions {
                if let InstructionKind::Phi(dest, _) = &inst.kind {
                    let ty = ssa_func.get_type(*dest);
                    let cl_ty = translate_type(&ty);
                    let cl_val = cg_ctx.builder.append_block_param(current_cl_block, cl_ty);
                    cg_ctx.values.insert(*dest, cl_val);
                    if let SsaType::Buffer(_) = ty {
                        let cl_len = cg_ctx
                            .builder
                            .append_block_param(current_cl_block, types::I64);
                        cg_ctx.buffer_lengths.insert(*dest, cl_len);
                    }
                }
            }
        }

        // 3. Compute Reverse Post-Order (RPO) to visit definitions before uses
        let mut rpo = Vec::new();
        let mut visited = HashSet::new();
        fn visit(
            block_id: SsaBlockId,
            func: &SsaFunction,
            visited: &mut HashSet<SsaBlockId>,
            rpo: &mut Vec<SsaBlockId>,
        ) {
            if visited.contains(&block_id) {
                return;
            }
            visited.insert(block_id);
            if let Some(block) = func.blocks.iter().find(|b| b.id == block_id) {
                for &succ in &block.successors {
                    visit(succ, func, visited, rpo);
                }
            }
            rpo.push(block_id);
        }
        visit(ssa_func.entry_block, ssa_func, &mut visited, &mut rpo);
        rpo.reverse();

        // 4. Lower instructions in RPO
        for block_id in rpo {
            let ssa_block = ssa_func
                .blocks
                .iter()
                .find(|b| b.id == block_id)
                .expect("Block not found");
            let current_cl_block = cg_ctx.blocks[&ssa_block.id];
            cg_ctx.builder.switch_to_block(current_cl_block);
            for inst in &ssa_block.instructions {
                lower::lower_instruction(&mut cg_ctx, inst, ssa_block.id)?;
            }
        }

        cg_ctx.builder.seal_all_blocks();
        cg_ctx.builder.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| e.to_string())?;
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let code = module.get_finalized_function(func_id);
    Ok(code as usize)
}
