//! Cranelift compilation coordinator.
//!
//! Translates Lirien IR types and basic block flows into Cranelift IR,
//! links native math helper symbols and memory routines, compiles, and registers target function pointers.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use lirien_ir::ir::{
    BlockId as SsaBlockId, Function as SsaFunction, InstructionKind, Type as SsaType,
    Value as SsaValue,
};
use std::collections::{HashMap, HashSet};
use tracing::info;

pub mod lower;

/// Structure representing active Cranelift compilation state mappings.
pub struct CodegenContext<'a, M: Module> {
    /// Cranelift function builder helper.
    pub builder: FunctionBuilder<'a>,
    /// Cranelift target module compilation unit.
    pub module: &'a mut M,
    /// Reference to the source SSA function.
    pub ssa_func: &'a SsaFunction,
    /// Maps Lirien block IDs to Cranelift blocks.
    pub blocks: HashMap<SsaBlockId, Block>,
    /// Maps Lirien values to Cranelift values.
    pub values: HashMap<SsaValue, Value>,
    /// Maps Lirien composite/unpacked aggregates to their flattened Cranelift values.
    pub unpacked_values: HashMap<SsaValue, Vec<Value>>,
    /// Maps buffer references to their Cranelift integer lengths.
    pub buffer_lengths: HashMap<SsaValue, Value>,
    /// Maps tensor references to their Cranelift integer dimension variables.
    pub tensor_dims: HashMap<SsaValue, Vec<Value>>,
    /// Maps sliced array SSA values to their step Cranelift value (for strided indexing).
    pub array_strides: HashMap<SsaValue, Value>,
    /// True if the function returns a multi-register or memory-flushed tuple aggregate.
    pub is_tuple_return: bool,
    /// Pointer to the pre-allocated struct return value buffer (sret).
    pub sret_ptr: Option<Value>,
}

/// Translates a Lirien IR type to its corresponding Cranelift machine type.
pub fn translate_type(ty: &SsaType) -> types::Type {
    match ty {
        SsaType::I8 | SsaType::U8 | SsaType::Bool => types::I8,
        SsaType::I16 | SsaType::U16 => types::I16,
        SsaType::I32 | SsaType::U32 => types::I32,
        SsaType::I64 | SsaType::U64 | SsaType::FnPointer(..) | SsaType::Closure(..) => {
            types::I64
        }
        SsaType::F32 => types::F32,
        SsaType::F64 => types::F64,
        SsaType::F32X4 => types::F32X4,
        SsaType::I32X4 => types::I32X4,
        SsaType::F64X2 => types::F64X2,
        SsaType::I64X2 => types::I64X2,
        SsaType::I8X16 | SsaType::U8X16 => types::I8X16,
        SsaType::I16X8 | SsaType::U16X8 => types::I16X8,
        SsaType::Array(_, _)
        | SsaType::Buffer(_)
        | SsaType::Tensor(_, _)
        | SsaType::Struct(_)
        | SsaType::TypedDict(_)
        | SsaType::NamedTuple(_)
        | SsaType::Enum(_)
        | SsaType::Pointer(_)
        | SsaType::NullablePointer(_)
        | SsaType::Optional(_)
        | SsaType::Tuple(_) => {
            types::I64 // Pointer
        }
        SsaType::Unknown => types::I64,
        SsaType::Refined(inner, _) | SsaType::Literal(inner, _) => translate_type(inner),
    }
}

/// Compiles a Lirien JIT function IR structure into machine code via Cranelift.
///
/// Under the hood, this sets up target architecture attributes, maps external dynamic library symbols
/// (like sin, cos, malloc, memcpy), lower instructions using the [`lower`] sub-module, compiles to
/// executable memory pages, and registers the compiled signature in the [`lirien_ir::registry`].
///
/// # Errors
/// Returns an error string if machine lowering, parsing, or JIT linking fails.
pub fn compile(ssa_func: &SsaFunction) -> Result<usize, String> {

    info!(target: "lirien::jit", "Compiling SSA to Machine Code via Cranelift for '{}'...", ssa_func.name);

    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| e.to_string())?;

    let mut jit_builder =
        JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names());

    // Register all previously compiled Lirien functions
    {
        let registry = lirien_ir::registry::GLOBAL_REGISTRY.lock().unwrap();
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
    extern "C" fn lirien_sin(x: f64) -> f64 { x.sin() }
    extern "C" fn lirien_cos(x: f64) -> f64 { x.cos() }
    extern "C" fn lirien_tan(x: f64) -> f64 { x.tan() }
    extern "C" fn lirien_asin(x: f64) -> f64 { x.asin() }
    extern "C" fn lirien_acos(x: f64) -> f64 { x.acos() }
    extern "C" fn lirien_atan(x: f64) -> f64 { x.atan() }
    extern "C" fn lirien_exp(x: f64) -> f64 { x.exp() }
    extern "C" fn lirien_log(x: f64) -> f64 { x.ln() }
    extern "C" fn lirien_log10(x: f64) -> f64 { x.log10() }
    extern "C" fn lirien_pow(x: f64, y: f64) -> f64 { x.powf(y) }
    extern "C" fn lirien_floor(x: f64) -> f64 { x.floor() }
    extern "C" fn lirien_ceil(x: f64) -> f64 { x.ceil() }
    extern "C" fn lirien_trunc(x: f64) -> f64 { x.trunc() }
    extern "C" fn lirien_nearest(x: f64) -> f64 { x.round() }
    
    // Naive math kernel for testing dependent types execution
    extern "C" fn lirien_matmul_alloc_f32(a: *const f32, b: *const f32, m: usize, n: usize, k: usize) -> *mut f32 {
        unsafe {
            let c = malloc(m * k * 4) as *mut f32;
            for i in 0..m {
                for j in 0..k {
                    let mut sum = 0.0;
                    for l in 0..n {
                        sum += (*a.add(i * n + l)) * (*b.add(l * k + j));
                    }
                    *c.add(i * k + j) = sum;
                }
            }
            c
        }
    }

    extern "C" fn lirien_tensor_arith_f32(a: *const f32, b: *const f32, size: usize, op: u8) -> *mut f32 {
        unsafe {
            let c = malloc(size * 4) as *mut f32;
            for i in 0..size {
                let va = *a.add(i);
                let vb = *b.add(i);
                *c.add(i) = match op {
                    0 => va + vb,
                    1 => va - vb,
                    2 => va * vb,
                    3 => va / vb,
                    _ => 0.0,
                };
            }
            c
        }
    }

    extern "C" fn lirien_tensor_reduce_f32(a: *const f32, size: usize, op: u8) -> f32 {
        unsafe {
            if size == 0 { return 0.0; }
            let mut res = *a;
            for i in 1..size {
                let v = *a.add(i);
                res = match op {
                    0 => res + v, // Sum
                    1 => if v > res { v } else { res }, // Max
                    2 => if v < res { v } else { res }, // Min
                    _ => res,
                };
            }
            res
        }
    }

    extern "C" fn lirien_tensor_scalar_arith_f32(a: *const f32, b: f32, size: usize, op: u8) -> *mut f32 {
        unsafe {
            let c = malloc(size * 4) as *mut f32;
            for i in 0..size {
                let va = *a.add(i);
                *c.add(i) = match op {
                    0 => va + b,
                    1 => va - b,
                    2 => va * b,
                    3 => va / b,
                    _ => 0.0,
                };
            }
            c
        }
    }

    extern "C" fn lirien_tensor_broadcast_f32(
        src_ptr: *const f32,
        src_dims: *const usize,
        src_rank: usize,
        target_dims: *const usize,
        target_rank: usize,
    ) -> *mut f32 {
        unsafe {
            let mut total_size = 1;
            let target_dims_slice = std::slice::from_raw_parts(target_dims, target_rank);
            for &d in target_dims_slice {
                total_size *= d;
            }

            let dest_ptr = malloc(total_size * 4) as *mut f32;
            let src_dims_slice = std::slice::from_raw_parts(src_dims, src_rank);

            let mut src_strides = vec![1; src_rank];
            for i in (0..src_rank as i64 - 1).rev() {
                src_strides[i as usize] = src_strides[i as usize + 1] * src_dims_slice[i as usize + 1];
            }

            let mut broadcast_strides = vec![0; target_rank];
            for i in 0..target_rank {
                let target_idx = target_rank as i64 - 1 - i as i64;
                let src_idx = src_rank as i64 - 1 - i as i64;
                if src_idx >= 0 {
                    if src_dims_slice[src_idx as usize] == target_dims_slice[target_idx as usize] {
                        broadcast_strides[target_idx as usize] = src_strides[src_idx as usize];
                    } else if src_dims_slice[src_idx as usize] == 1 {
                        broadcast_strides[target_idx as usize] = 0;
                    }
                }
            }

            for i in 0..total_size {
                let mut curr_idx = i;
                let mut src_offset = 0;
                let mut stride_in_dest = total_size;
                for j in 0..target_rank {
                    stride_in_dest /= target_dims_slice[j];
                    let dim_idx = curr_idx / stride_in_dest;
                    src_offset += dim_idx * broadcast_strides[j];
                    curr_idx %= stride_in_dest;
                }
                *dest_ptr.add(i) = *src_ptr.add(src_offset);
            }

            dest_ptr
        }
    }

    jit_builder.symbol("malloc", malloc as *const u8);
    jit_builder.symbol("memcpy", memcpy as *const u8);
    jit_builder.symbol("sin", lirien_sin as *const u8);
    jit_builder.symbol("cos", lirien_cos as *const u8);
    jit_builder.symbol("tan", lirien_tan as *const u8);
    jit_builder.symbol("asin", lirien_asin as *const u8);
    jit_builder.symbol("acos", lirien_acos as *const u8);
    jit_builder.symbol("atan", lirien_atan as *const u8);
    jit_builder.symbol("exp", lirien_exp as *const u8);
    jit_builder.symbol("log", lirien_log as *const u8);
    jit_builder.symbol("log10", lirien_log10 as *const u8);
    jit_builder.symbol("pow", lirien_pow as *const u8);
    jit_builder.symbol("floor", lirien_floor as *const u8);
    jit_builder.symbol("ceil", lirien_ceil as *const u8);
    jit_builder.symbol("trunc", lirien_trunc as *const u8);
    jit_builder.symbol("nearest", lirien_nearest as *const u8);
    jit_builder.symbol("lirien_matmul_alloc_f32", lirien_matmul_alloc_f32 as *const u8);
    jit_builder.symbol("lirien_tensor_arith_f32", lirien_tensor_arith_f32 as *const u8);
    jit_builder.symbol("lirien_tensor_reduce_f32", lirien_tensor_reduce_f32 as *const u8);
    jit_builder.symbol("lirien_tensor_scalar_arith_f32", lirien_tensor_scalar_arith_f32 as *const u8);
    jit_builder.symbol("lirien_tensor_broadcast_f32", lirien_tensor_broadcast_f32 as *const u8);

    let mut module = JITModule::new(jit_builder);
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    let mut arg_types = Vec::new();
    for i in 0..ssa_func.arg_count {
        arg_types.push(ssa_func.get_type(SsaValue(i)));
    }

    let (sig, is_ptr_return, _) = lower::build_cranelift_signature(
        ssa_func,
        &arg_types,
        &ssa_func.return_type,
        false,
        &module,
    );

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

        info!(target: "lirien::jit", "JIT initialized for {} ({}) [SIMD: {}]", isa.name(), isa.triple().architecture, simd_info);
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
        let unpacked_values = HashMap::new();
        let buffer_lengths = HashMap::new();

        let mut cg_ctx = CodegenContext {
            builder,
            module: &mut module,
            ssa_func,
            blocks,
            values,
            unpacked_values,
            buffer_lengths,
            tensor_dims: HashMap::new(),
            array_strides: HashMap::new(),
            is_tuple_return: is_ptr_return,
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
                    if let SsaType::NamedTuple(_) = ty {
                        let cl_types = lower::get_flattened_types(ssa_func, &ty);
                        let mut field_vals = Vec::new();
                        for cl_ty in cl_types {
                            field_vals.push(cg_ctx.builder.append_block_param(current_cl_block, cl_ty));
                        }
                        cg_ctx.unpacked_values.insert(*dest, field_vals);
                    } else {
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
                for i in 0..ssa_func.arg_count {
                    let val = SsaValue(i);
                    let ty = ssa_func.get_type(val);
                    match ty {
                        SsaType::Buffer(_) => {
                            cg_ctx.values.insert(val, cg_ctx.builder.block_params(entry_block)[p_idx]);
                            cg_ctx.buffer_lengths.insert(val, cg_ctx.builder.block_params(entry_block)[p_idx + 1]);
                            p_idx += 2;
                        }
                        SsaType::Tensor(_, ref dims) => {
                            cg_ctx.values.insert(val, cg_ctx.builder.block_params(entry_block)[p_idx]);
                            let mut dim_vals = Vec::new();
                            for j in 0..dims.len() {
                                dim_vals.push(cg_ctx.builder.block_params(entry_block)[p_idx + 1 + j]);
                            }
                            cg_ctx.tensor_dims.insert(val, dim_vals);
                            p_idx += 1 + dims.len();
                        }
                        SsaType::NamedTuple(_) | SsaType::Tuple(_) => {
                            let cl_types = lower::get_flattened_types(ssa_func, &ty);
                            let mut field_vals = Vec::new();
                            for _ in cl_types {
                                field_vals.push(cg_ctx.builder.block_params(entry_block)[p_idx]);
                                p_idx += 1;
                            }
                            cg_ctx.unpacked_values.insert(val, field_vals);
                        }
                        _ if ty.is_simd() => {
                            let ptr = cg_ctx.builder.block_params(entry_block)[p_idx];
                            p_idx += 1;
                            let cl_ty = translate_type(&ty);
                            let vec_val = cg_ctx.builder.ins().load(cl_ty, MemFlags::new(), ptr, 0);
                            cg_ctx.values.insert(val, vec_val);
                        }
                        _ => {
                            cg_ctx.values.insert(val, cg_ctx.builder.block_params(entry_block)[p_idx]);
                            p_idx += 1;
                        }
                    }
                }
            }

            for inst in &ssa_block.instructions {
                lower::lower_instruction(&mut cg_ctx, inst, ssa_block.id)
                    .map_err(|e| e.to_string())?;
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
