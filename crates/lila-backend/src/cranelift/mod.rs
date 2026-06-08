use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use lila_ir::ir::{
    BlockId as SsaBlockId, Function as SsaFunction, InstructionKind, Type as SsaType,
    Value as SsaValue,
};
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
        SsaType::I64 | SsaType::U64 | SsaType::FnPointer(_, _) | SsaType::Closure(_, _, _) => {
            types::I64
        }
        SsaType::F32 => types::F32,
        SsaType::F64 => types::F64,
        SsaType::F32X4 => types::F32X4,
        SsaType::I32X4 => types::I32X4,
        SsaType::F64X2 => types::F64X2,
        SsaType::I64X2 => types::I64X2,
        SsaType::Array(_, _)
        | SsaType::Buffer(_)
        | SsaType::Struct(_)
        | SsaType::Enum(_)
        | SsaType::Pointer(_)
        | SsaType::Tuple(_) => {
            types::I64 // Pointer
        }
        SsaType::Unknown => types::I64,
        SsaType::Refined(inner, _) | SsaType::Literal(inner, _) => translate_type(inner),
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

    let mut jit_builder =
        JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names());

    // Register all previously compiled Lila functions
    {
        let registry = lila_ir::registry::GLOBAL_REGISTRY.lock().unwrap();
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
        fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
    }
    extern "C" fn lila_sin(x: f64) -> f64 { x.sin() }
    extern "C" fn lila_cos(x: f64) -> f64 { x.cos() }
    extern "C" fn lila_pow(x: f64, y: f64) -> f64 { x.powf(y) }

    jit_builder.symbol("malloc", malloc as *const u8);
    jit_builder.symbol("memcpy", memcpy as *const u8);
    jit_builder.symbol("sin", lila_sin as *const u8);
    jit_builder.symbol("cos", lila_cos as *const u8);
    jit_builder.symbol("pow", lila_pow as *const u8);

    let mut module = JITModule::new(jit_builder);
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    let mut sig = module.make_signature();
    let mut is_buffer = Vec::new();
    let mut is_simd_arg = Vec::new();
    let mut is_ptr_return = false;

    if let SsaType::Tuple(_) = ssa_func.return_type {
        sig.params.push(AbiParam::new(types::I64));
        is_ptr_return = true;
    } else if ssa_func.return_type.is_simd() {
        sig.params.push(AbiParam::new(types::I64));
        is_ptr_return = true;
    }

    for i in 0..ssa_func.arg_count {
        let ty = ssa_func.get_type(SsaValue(i));
        if let SsaType::Buffer(_) = ty {
            sig.params.push(AbiParam::new(types::I64)); // Ptr
            sig.params.push(AbiParam::new(types::I64)); // Len
            is_buffer.push(true);
            is_simd_arg.push(false);
        } else if ty.is_simd() {
            sig.params.push(AbiParam::new(types::I64)); // Pass by pointer for interop
            is_buffer.push(false);
            is_simd_arg.push(true);
        } else {
            sig.params.push(AbiParam::new(translate_type(&ty)));
            is_buffer.push(false);
            is_simd_arg.push(false);
        }
    }

    if ssa_func.return_type != SsaType::Unknown && !is_ptr_return {
        sig.returns
            .push(AbiParam::new(translate_type(&ssa_func.return_type)));
    }

    let func_id = module
        .declare_function(&ssa_func.name, Linkage::Export, &sig)
        .map_err(|e| e.to_string())?;

    ctx.func.signature = sig;

    // One-time CPU feature logging
    static CPU_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !CPU_LOGGED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        let mut combined_flags = isa.flags().to_string();
        for val in isa.isa_flags() {
            combined_flags.push(' ');
            combined_flags.push_str(val.name);
        }

        let mut features: Vec<String> = [
            "has_neon",
            "has_asimd",
            "has_simd",
            "has_avx",
            "has_sse",
            "has_sse2",
            "has_sse3",
            "has_ssse3",
            "has_sse41",
            "has_sse42",
        ]
        .iter()
        .filter(|&&f| combined_flags.contains(f))
        .map(|&f| f.trim_start_matches("has_").to_string())
        .collect();

        // Handle baselines for major architectures
        let arch = isa.triple().architecture.to_string().to_lowercase();
        if arch.contains("aarch64")
            && !features.contains(&"neon".to_string())
            && !features.contains(&"asimd".to_string())
        {
            features.insert(0, "Neon".to_string());
        } else if arch.contains("x86_64") && !features.contains(&"sse2".to_string()) {
            features.insert(0, "SSE2".to_string());
        }

        let simd_info = if features.is_empty() {
            "Software"
        } else {
            &features.join("+")
        };

        info!(target: "lila::jit", "JIT initialized for {} ({}) [SIMD: {}]", isa.name(), isa.triple().architecture, simd_info);
    }

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let mut blocks: HashMap<SsaBlockId, Block> = HashMap::new();

        for ssa_block in &ssa_func.blocks {
            blocks.insert(ssa_block.id, builder.create_block());
        }

        let entry_block = blocks[&ssa_func.entry_block];
        builder.append_block_params_for_function_params(entry_block);

        let values = HashMap::new();
        let buffer_lengths = HashMap::new();

        let mut cg_ctx = CodegenContext {
            builder,
            module: &mut module,
            ssa_func,
            blocks,
            values,
            buffer_lengths,
            is_tuple_return: matches!(ssa_func.return_type, SsaType::Tuple(_)),
            sret_ptr: None,
        };

        let mut param_idx = 0;
        let sret_ptr = if is_ptr_return {
            let ptr = cg_ctx.builder.block_params(entry_block)[param_idx];
            param_idx += 1;
            Some(ptr)
        } else {
            None
        };
        cg_ctx.sret_ptr = sret_ptr;

        // 2. Pre-declare all Phis across all blocks
        for ssa_block in &ssa_func.blocks {
            let current_cl_block = cg_ctx.blocks[&ssa_block.id];

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

            if block_id == ssa_func.entry_block {
                // 4.1. Initialize arguments in the entry block
                let mut p_idx = param_idx; // SRet already handled
                for (i, (&is_buf, &is_simd)) in is_buffer.iter().zip(is_simd_arg.iter()).enumerate()
                {
                    let val = SsaValue(i);
                    let ty = ssa_func.get_type(val);
                    if is_buf {
                        cg_ctx
                            .values
                            .insert(val, cg_ctx.builder.block_params(entry_block)[p_idx]);
                        cg_ctx
                            .buffer_lengths
                            .insert(val, cg_ctx.builder.block_params(entry_block)[p_idx + 1]);
                        p_idx += 2;
                    } else if is_simd {
                        let ptr = cg_ctx.builder.block_params(entry_block)[p_idx];
                        p_idx += 1;
                        let cl_ty = translate_type(&ty);
                        let vec_val = cg_ctx.builder.ins().load(cl_ty, MemFlags::new(), ptr, 0);
                        cg_ctx.values.insert(val, vec_val);
                    } else {
                        cg_ctx
                            .values
                            .insert(val, cg_ctx.builder.block_params(entry_block)[p_idx]);
                        p_idx += 1;
                    }
                }
            }

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
