use crate::backend::SolverBackend;
use lila_ir::analysis::interval::{Bound, IntervalAnalysisResults};
use lila_ir::ir::{BlockId, Function, InstructionKind, Type, Value};
use std::collections::HashMap;

pub mod arithmetic;
pub mod calls;
pub mod control_flow;
pub mod memory;
pub mod tuples;

pub fn get_leaf_offsets(
    ty: &Type,
    struct_layouts: &HashMap<String, Vec<(String, Type)>>,
    base_offset: usize,
    offsets: &mut Vec<(usize, Type)>,
) {
    match ty {
        Type::Struct(name) => {
            if let Some(fields) = struct_layouts.get(name) {
                let mut offset = 0;
                for (_, f_ty) in fields {
                    let align = f_ty.align(struct_layouts);
                    offset = (offset + align - 1) & !(align - 1);
                    get_leaf_offsets(f_ty, struct_layouts, base_offset + offset, offsets);
                    offset += f_ty.size(struct_layouts);
                }
            }
        }
        Type::Tuple(types) => {
            let mut offset = 0;
            for f_ty in types {
                let align = f_ty.align(struct_layouts);
                offset = (offset + align - 1) & !(align - 1);
                get_leaf_offsets(f_ty, struct_layouts, base_offset + offset, offsets);
                offset += f_ty.size(struct_layouts);
            }
        }
        Type::Literal(inner, _) | Type::Refined(inner, _) => {
            get_leaf_offsets(inner, struct_layouts, base_offset, offsets);
        }
        _ => {
            offsets.push((base_offset, ty.clone()));
        }
    }
}

pub fn copy_composite<
    'a,
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'a, B>,
    current_state: B::Array,
    src_val: Value,
    src_ty: &Type,
    dest_base_offset: i64,
) -> B::Array {
    let mut new_state = current_state;
    let mut leaves = Vec::new();
    get_leaf_offsets(src_ty, &t_ctx.func.struct_layouts, 0, &mut leaves);

    if let Some(src_array) = t_ctx.z3_arrays.get(&src_val).cloned() {
        for (offset, _leaf_ty) in leaves {
            let src_offset_z3 = t_ctx.backend.int_from_i64(offset as i64);
            let dest_offset_z3 = t_ctx.backend.int_from_i64(dest_base_offset + offset as i64);
            let val = t_ctx.backend.array_select_bv(&src_array, &src_offset_z3);
            new_state = t_ctx
                .backend
                .array_store_bv(&new_state, &dest_offset_z3, &val);
        }
    }
    new_state
}

pub struct TranslationContext<'a, B: SolverBackend> {
    pub backend: &'a mut B,
    pub func: &'a Function,
    pub analysis: &'a IntervalAnalysisResults,
    pub uid: usize,
    pub z3_ints: HashMap<Value, B::Int>,
    pub z3_floats: HashMap<Value, B::Float>,
    pub z3_bvs: HashMap<Value, B::BV>,
    pub z3_arrays: HashMap<Value, B::Array>,
    pub z3_tensor_dims: HashMap<Value, Vec<B::Int>>,
    pub tuple_mappings: HashMap<Value, Vec<Value>>,
    pub block_conditions: HashMap<BlockId, B::Bool>,
    pub edge_conditions: HashMap<(BlockId, BlockId), B::Bool>,
    pub has_refinements: bool,
}

pub fn verify_with_context<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    backend: &mut B,
    func: &Function,
    analysis: &IntervalAnalysisResults,
    _liveness: lila_ir::analysis::liveness::LivenessAnalysisResults,
    uid: usize,
) -> Result<(), String> {
    let mut t_ctx = TranslationContext {
        backend,
        func,
        analysis,
        uid,
        z3_ints: HashMap::new(),
        z3_floats: HashMap::new(),
        z3_bvs: HashMap::new(),
        z3_arrays: HashMap::new(),
        z3_tensor_dims: HashMap::new(),
        tuple_mappings: HashMap::new(),
        block_conditions: HashMap::new(),
        edge_conditions: HashMap::new(),
        has_refinements: func.ret_refinement.is_some(),
    };

    // 1. Initialize Z3 values for all SSA values
    memory::init_values(&mut t_ctx)?;

    assert_cfg_constraints(&mut t_ctx);

    translate_instructions(&mut t_ctx)?;

    assert_derived_intervals(&mut t_ctx);

    verify_return_refinements(&mut t_ctx)?;

    verify_call_arguments(&mut t_ctx)?;

    // 11. Final Consistency Check
    if !t_ctx.has_refinements {
        tracing::info!(target: "lila::verify::verifier", "Skipping final consistency check for '{}' (no refinements).", func.name);
    } else {
        tracing::info!(target: "lila::verify::verifier", "Performing final consistency check for '{}'...", func.name);
        if !t_ctx.backend.check()? {
            return Err("Formal verification failed: Logical contradiction detected.".to_string());
        }
    }

    tracing::info!(target: "lila::verify::verifier", "Proof successful for '{}'", func.name);
    Ok(())
}

fn assert_derived_intervals<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'_, B>,
) {
    // 6. Assert derived intervals and refinements
    for (val, interval) in &t_ctx.analysis.intervals {
        if let Some(ty) = t_ctx.func.value_types.get(val) {
            if let Some(z3_bv) = t_ctx.z3_bvs.get(val) {
                if let Some(bit_width) = ty.int_bit_width() {
                    let is_signed = ty.is_signed();
                    if let Bound::Finite(low) = interval.low {
                        let low_bv = t_ctx.backend.bv_from_i64(low as i64, bit_width);
                        if is_signed {
                            let __tmp = t_ctx.backend.bv_sge(z3_bv, &low_bv);
                            t_ctx.backend.assert(&__tmp);
                        } else {
                            let __tmp = t_ctx.backend.bv_uge(z3_bv, &low_bv);
                            t_ctx.backend.assert(&__tmp);
                        }
                    }
                    if let Bound::Finite(high) = interval.high {
                        let high_bv = t_ctx.backend.bv_from_i64(high as i64, bit_width);
                        if is_signed {
                            let __tmp = t_ctx.backend.bv_sle(z3_bv, &high_bv);
                            t_ctx.backend.assert(&__tmp);
                        } else {
                            let __tmp = t_ctx.backend.bv_ule(z3_bv, &high_bv);
                            t_ctx.backend.assert(&__tmp);
                        }
                    }
                }
            } else if let Some(z3_float) = t_ctx.z3_floats.get(val) {
                if let Bound::Finite(low) = interval.low {
                    let low_float = if ty.is_float32() {
                        t_ctx.backend.float_from_f32(low as f32)
                    } else {
                        t_ctx.backend.float_from_f64(low)
                    };
                    let __tmp = t_ctx.backend.float_ge(z3_float, &low_float);
                    t_ctx.backend.assert(&__tmp);
                }
                if let Bound::Finite(high) = interval.high {
                    let high_float = if ty.is_float32() {
                        t_ctx.backend.float_from_f32(high as f32)
                    } else {
                        t_ctx.backend.float_from_f64(high)
                    };
                    let __tmp = t_ctx.backend.float_le(z3_float, &high_float);
                    t_ctx.backend.assert(&__tmp);
                }
            }
        }
    }

    for ((val, b_id), interval) in &t_ctx.analysis.block_narrowing {
        if let Some(path_cond) = t_ctx.block_conditions.get(b_id) {
            if let Some(ty) = t_ctx.func.value_types.get(val) {
                if let Some(z3_bv) = t_ctx.z3_bvs.get(val) {
                    if let Some(bit_width) = ty.int_bit_width() {
                        let is_signed = ty.is_signed();
                        if let Bound::Finite(low) = interval.low {
                            let low_bv = t_ctx.backend.bv_from_i64(low as i64, bit_width);
                            if is_signed {
                                let __inner = t_ctx.backend.bv_sge(z3_bv, &low_bv);
                                let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                                t_ctx.backend.assert(&__tmp);
                            } else {
                                let __inner = t_ctx.backend.bv_uge(z3_bv, &low_bv);
                                let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                                t_ctx.backend.assert(&__tmp);
                            }
                        }
                        if let Bound::Finite(high) = interval.high {
                            let high_bv = t_ctx.backend.bv_from_i64(high as i64, bit_width);
                            if is_signed {
                                let __inner = t_ctx.backend.bv_sle(z3_bv, &high_bv);
                                let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                                t_ctx.backend.assert(&__tmp);
                            } else {
                                let __inner = t_ctx.backend.bv_ule(z3_bv, &high_bv);
                                let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                                t_ctx.backend.assert(&__tmp);
                            }
                        }
                    }
                } else if let Some(z3_float) = t_ctx.z3_floats.get(val) {
                    if let Bound::Finite(low) = interval.low {
                        let low_float = if ty.is_float32() {
                            t_ctx.backend.float_from_f32(low as f32)
                        } else {
                            t_ctx.backend.float_from_f64(low)
                        };
                        let __inner = t_ctx.backend.float_ge(z3_float, &low_float);
                        let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                        t_ctx.backend.assert(&__tmp);
                    }
                    if let Bound::Finite(high) = interval.high {
                        let high_float = if ty.is_float32() {
                            t_ctx.backend.float_from_f32(high as f32)
                        } else {
                            t_ctx.backend.float_from_f64(high)
                        };
                        let __inner = t_ctx.backend.float_le(z3_float, &high_float);
                        let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                        t_ctx.backend.assert(&__tmp);
                    }
                }
            }
        }
    }
}

fn assert_cfg_constraints<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'_, B>,
) {
    // 3. Declare Booleans for all blocks and known edges
    for block in &t_ctx.func.blocks {
        let b_cond = t_ctx.backend.bool_const(&format!(
            "{}_block_{}_{}",
            t_ctx.func.name, block.id.0, t_ctx.uid
        ));
        t_ctx.block_conditions.insert(block.id, b_cond);

        if let Some(last_inst) = block.instructions.last() {
            match &last_inst.kind {
                InstructionKind::Jump(target) => {
                    let e_cond = t_ctx.backend.bool_const(&format!(
                        "{}_edge_{}_{}_{}",
                        t_ctx.func.name, block.id.0, target.0, t_ctx.uid
                    ));
                    t_ctx.edge_conditions.insert((block.id, *target), e_cond);
                }
                InstructionKind::Branch(_, t_block, f_block) => {
                    let et_cond = t_ctx.backend.bool_const(&format!(
                        "{}_edge_{}_{}_{}",
                        t_ctx.func.name, block.id.0, t_block.0, t_ctx.uid
                    ));
                    t_ctx.edge_conditions.insert((block.id, *t_block), et_cond);
                    let ef_cond = t_ctx.backend.bool_const(&format!(
                        "{}_edge_{}_{}_{}",
                        t_ctx.func.name, block.id.0, f_block.0, t_ctx.uid
                    ));
                    t_ctx.edge_conditions.insert((block.id, *f_block), ef_cond);
                }
                InstructionKind::Match(_, cases, default, _) => {
                    for target in cases.values() {
                        let e_cond = t_ctx.backend.bool_const(&format!(
                            "{}_edge_{}_{}_{}",
                            t_ctx.func.name, block.id.0, target.0, t_ctx.uid
                        ));
                        t_ctx.edge_conditions.insert((block.id, *target), e_cond);
                    }
                    let e_cond_default = t_ctx.backend.bool_const(&format!(
                        "{}_edge_{}_{}_{}",
                        t_ctx.func.name, block.id.0, default.0, t_ctx.uid
                    ));
                    t_ctx
                        .edge_conditions
                        .insert((block.id, *default), e_cond_default);
                }
                _ => {}
            }
        }

        // Handle implicit edges from ParallelFor
        for inst in &block.instructions {
            if let InstructionKind::ParallelFor(_, _, _, _, body_block, ..) = &inst.kind {
                let e_cond = t_ctx.backend.bool_const(&format!(
                    "{}_edge_{}_{}_{}",
                    t_ctx.func.name, block.id.0, body_block.0, t_ctx.uid
                ));
                t_ctx
                    .edge_conditions
                    .insert((block.id, *body_block), e_cond);
            }
        }
    }

    // 4. Assert Structural CFG Constraints
    let true_cond = t_ctx.backend.bool_from_bool(true);
    let false_cond = t_ctx.backend.bool_from_bool(false);
    if let Some(entry_cond) = t_ctx.block_conditions.get(&t_ctx.func.entry_block) {
        let __tmp = t_ctx.backend.bool_eq(entry_cond, &true_cond);
        t_ctx.backend.assert(&__tmp);
    }
    for block in &t_ctx.func.blocks {
        let path_cond = t_ctx.block_conditions.get(&block.id).unwrap().clone();
        if block.id != t_ctx.func.entry_block {
            let mut incoming_edges = Vec::new();
            for (edge_src, edge_dst) in t_ctx.edge_conditions.keys() {
                if *edge_dst == block.id {
                    incoming_edges
                        .push(t_ctx.edge_conditions.get(&(*edge_src, *edge_dst)).unwrap());
                }
            }
            if incoming_edges.is_empty() {
                let __tmp = t_ctx.backend.bool_eq(&path_cond, &false_cond);
                t_ctx.backend.assert(&__tmp);
            } else {
                let or_expr = t_ctx.backend.bool_or(&incoming_edges);
                let __tmp = t_ctx.backend.bool_eq(&path_cond, &or_expr);
                t_ctx.backend.assert(&__tmp);
            }
        }
    }
}

fn translate_instructions<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'_, B>,
) -> Result<(), String> {
    // 5. Translate Instructions (Arithmetic, Memory, Control Flow)
    tracing::info!(target: "lila::verify::verifier", "Translating instructions for '{}'...", t_ctx.func.name);
    for block in &t_ctx.func.blocks {
        let path_cond = t_ctx.block_conditions.get(&block.id).unwrap().clone();
        for inst in &block.instructions {
            match &inst.kind {
                InstructionKind::Add(..)
                | InstructionKind::Sub(..)
                | InstructionKind::Mul(..)
                | InstructionKind::SDiv(..)
                | InstructionKind::UDiv(..)
                | InstructionKind::SRem(..)
                | InstructionKind::URem(..)
                | InstructionKind::MatMult(..)
                | InstructionKind::FAdd(..)
                | InstructionKind::FSub(..)
                | InstructionKind::FMul(..)
                | InstructionKind::FDiv(..)
                | InstructionKind::FSqrt(..)
                | InstructionKind::FSin(..)
                | InstructionKind::FCos(..)
                | InstructionKind::FPow(..)
                | InstructionKind::ConstInt(..)
                | InstructionKind::ConstFloat(..)
                | InstructionKind::Assign(..)
                | InstructionKind::Eq(..)
                | InstructionKind::Ne(..)
                | InstructionKind::SLt(..)
                | InstructionKind::SLe(..)
                | InstructionKind::SGt(..)
                | InstructionKind::SGe(..)
                | InstructionKind::ULt(..)
                | InstructionKind::ULe(..)
                | InstructionKind::UGt(..)
                | InstructionKind::UGe(..)
                | InstructionKind::FLt(..)
                | InstructionKind::FLe(..)
                | InstructionKind::FGt(..)
                | InstructionKind::FGe(..)
                | InstructionKind::And(..)
                | InstructionKind::Or(..)
                | InstructionKind::Xor(..)
                | InstructionKind::Shl(..)
                | InstructionKind::LShr(..)
                | InstructionKind::AShr(..)
                | InstructionKind::IToF(..)
                | InstructionKind::FToI(..)
                | InstructionKind::FConv(..)
                | InstructionKind::Not(..)
                | InstructionKind::Abs(..)
                | InstructionKind::Neg(..)
                | InstructionKind::Min(..)
                | InstructionKind::Max(..)
                | InstructionKind::Avg(..)
                | InstructionKind::SIMDSplat(..)
                | InstructionKind::SIMDExtractLane(..)
                | InstructionKind::SIMDInsertLane(..) => {

                    arithmetic::translate(t_ctx, inst, &path_cond)?;
                }
                InstructionKind::Jump(_)
                | InstructionKind::Branch(..)
                | InstructionKind::Match(..)
                | InstructionKind::Phi(..) => {
                    control_flow::translate(t_ctx, inst, &path_cond, block.id)?;
                }
                InstructionKind::ArrayLoad(..)
                | InstructionKind::ArrayStore(..)
                | InstructionKind::BufferLoad(..)
                | InstructionKind::BufferStore(..)
                | InstructionKind::TensorLoad(..)
                | InstructionKind::TensorStore(..)
                | InstructionKind::BufferLen(..)
                | InstructionKind::TensorAdd(..)
                | InstructionKind::TensorSub(..)
                | InstructionKind::TensorMul(..)
                | InstructionKind::TensorDiv(..)
                | InstructionKind::TensorScalarAdd(..)
                | InstructionKind::TensorScalarSub(..)
                | InstructionKind::TensorScalarMul(..)
                | InstructionKind::TensorScalarDiv(..)
                | InstructionKind::TensorSum(..)
                | InstructionKind::TensorMax(..)
                | InstructionKind::TensorMin(..)
                | InstructionKind::StructCreate(..)
                | InstructionKind::StructLoad(..)
                | InstructionKind::StructOffset(..)
                | InstructionKind::StructSet(..)
                | InstructionKind::EnumCreate(..)
                | InstructionKind::EnumGetTag(..)
                | InstructionKind::EnumIsVariant(..)
                | InstructionKind::EnumAsVariant(..)
                | InstructionKind::EnumExtract(..)
                | InstructionKind::Alloc(..)

                | InstructionKind::PointerLoad(..)
                | InstructionKind::PointerStore(..) => {
                    memory::translate(t_ctx, inst, &path_cond)?;
                }
                InstructionKind::TupleCreate(..) | InstructionKind::TupleExtract(..) => {
                    tuples::translate(t_ctx, inst, &path_cond)?;
                }
                InstructionKind::Call(..) => {
                    calls::translate(t_ctx, inst, &path_cond)?;
                }
                InstructionKind::IndirectCall(..)
                | InstructionKind::Lambda(..)
                | InstructionKind::Return(..) => {}
                InstructionKind::ParallelFor(
                    index_var,
                    start,
                    stop,
                    _step,
                    body_block,
                    _exit_block,
                    _captures,
                ) => {
                    if let Some(edge_p) = t_ctx.edge_conditions.get(&(block.id, *body_block)) {
                        let __tmp = t_ctx.backend.bool_eq(edge_p, &path_cond);
                        t_ctx.backend.assert(&__tmp);
                    }

                    if let Some(body_cond) = t_ctx.block_conditions.get(body_block) {
                        let idx_int = if let Some(bv) = t_ctx.z3_bvs.get(index_var) {
                            t_ctx.backend.bv_to_int(bv, true)
                        } else {
                            t_ctx.z3_ints.get(index_var).unwrap().clone()
                        };
                        let start_int = if let Some(bv) = t_ctx.z3_bvs.get(start) {
                            t_ctx.backend.bv_to_int(bv, true)
                        } else {
                            t_ctx.z3_ints.get(start).unwrap().clone()
                        };
                        let stop_int = if let Some(bv) = t_ctx.z3_bvs.get(stop) {
                            t_ctx.backend.bv_to_int(bv, true)
                        } else {
                            t_ctx.z3_ints.get(stop).unwrap().clone()
                        };

                        // In the loop body, start <= index < stop
                        let __inner1 = t_ctx.backend.int_ge(&idx_int, &start_int);
                        let __tmp1 = t_ctx.backend.bool_implies(body_cond, &__inner1);
                        t_ctx.backend.assert(&__tmp1);

                        let __inner2 = t_ctx.backend.int_lt(&idx_int, &stop_int);
                        let __tmp2 = t_ctx.backend.bool_implies(body_cond, &__inner2);
                        t_ctx.backend.assert(&__tmp2);
                    }
                }
                InstructionKind::Nop() => {}
            }

            // Translate logical constraints attached to the instruction
            translate_constraints(t_ctx, inst, &path_cond)?;
        }
    }
    Ok(())
}

fn translate_constraints<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'_, B>,
    inst: &lila_ir::ir::Instruction,
    path_cond: &B::Bool,
) -> Result<(), String> {
    use crate::refinement_parser::{parse_bool_expr_with_resolver, Resolver};

    if inst.constraints.is_empty() {
        return Ok(());
    }

    let resolver = Resolver {
        ints: &t_ctx.z3_ints,
        floats: &t_ctx.z3_floats,
        bvs: &t_ctx.z3_bvs,
        arrays: &t_ctx.z3_arrays,
    };

    for constraint in &inst.constraints {
        let z3_constraint = parse_bool_expr_with_resolver(constraint, &resolver)?;
        let __tmp = t_ctx.backend.bool_implies(path_cond, &z3_constraint);
        t_ctx.backend.assert(&__tmp);
    }

    Ok(())
}

fn verify_return_refinements<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'_, B>,
) -> Result<(), String> {
    // 8. Verify Return Refinements & Shapes
    let ret_ty = t_ctx.func.return_type.clone();
    for block in &t_ctx.func.blocks {
        let path_cond = t_ctx.block_conditions.get(&block.id).unwrap();
        for inst in &block.instructions {
            if let InstructionKind::Return(Some(ret_val)) = &inst.kind {
                let actual_ty = t_ctx.func.get_type(*ret_val);
                
                // Shape verification
                if let (Type::Tensor(_, src_dims), Type::Tensor(_, target_dims)) = (&actual_ty, &ret_ty) {
                    if src_dims.len() != target_dims.len() {
                        let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                        return Err(format!("Tensor rank mismatch in return: expected {} dims, got {}{}", target_dims.len(), src_dims.len(), loc_info));
                    }
                    if let Some(src_z3_dims) = t_ctx.z3_tensor_dims.get(ret_val) {
                        for (dim_idx, target_dim_name) in target_dims.iter().enumerate() {
                            let target_z3_dim = t_ctx.backend.int_const(target_dim_name);
                            t_ctx.backend.push();
                            t_ctx.backend.assert(path_cond);
                            let eq = t_ctx.backend.int_eq(&src_z3_dims[dim_idx], &target_z3_dim);
                            let not_eq = t_ctx.backend.bool_not(&eq);
                            t_ctx.backend.assert(&not_eq);
                            
                            if t_ctx.backend.check()? {
                                let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                return Err(format!("Tensor shape mismatch in return: dimension '{}' (idx {}) does not match{}", target_dim_name, dim_idx, loc_info));
                            }
                            t_ctx.backend.pop(1);
                        }
                    }
                }

                if let Some(ret_ref) = &t_ctx.func.ret_refinement {
                    let ty = t_ctx.func.get_type(*ret_val);
                    let res = if let Some(z3_bv) = t_ctx.z3_bvs.get(ret_val) {
                        let bv_int = t_ctx.backend.bv_to_int(z3_bv, ty.is_signed());
                        crate::refinement_parser::parse_refinement(ret_ref, &bv_int, Some(z3_bv))
                    } else if let Some(z3_int) = t_ctx.z3_ints.get(ret_val) {
                        crate::refinement_parser::parse_refinement(ret_ref, z3_int, None)
                    } else if let Some(z3_float) = t_ctx.z3_floats.get(ret_val) {
                        crate::refinement_parser::parse_float_refinement(ret_ref, z3_float)
                    } else {
                        continue;
                    };

                    if let Ok(expr) = res {
                        t_ctx.backend.push();
                        t_ctx.backend.assert(path_cond);
                        let __tmp = t_ctx.backend.bool_not(&expr);
                        t_ctx.backend.assert(&__tmp);
                        if t_ctx.backend.check()? {
                            let loc_info = inst
                                .location
                                .map(|l| format!(" at {}", l))
                                .unwrap_or_default();
                            return Err(format!(
                                "Return refinement violation: value of {:?} does not satisfy '{}' and may be violated on some reachable path{}.",
                                ret_val, ret_ref, loc_info
                            ));
                        }
                        t_ctx.backend.pop(1);
                    }
                }
            }
        }
    }

    Ok(())
}

fn verify_call_arguments<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'_, B>,
) -> Result<(), String> {
    // 10. Verify Call Arguments
    for block in &t_ctx.func.blocks {
        let path_cond = t_ctx.block_conditions.get(&block.id).unwrap();
        for inst in &block.instructions {
            if let InstructionKind::Call(_, target_name, args) = &inst.kind {
                let registry = lila_ir::registry::GLOBAL_REGISTRY.lock().unwrap();
                let sig = if target_name == &t_ctx.func.name {
                    // Recursive call: use current function's signature
                    let mut arg_types = Vec::new();
                    let mut arg_refinements = HashMap::new();
                    for i in 0..t_ctx.func.arg_count {
                        let v = Value(i);
                        arg_types.push(t_ctx.func.get_type(v));
                        if let Some(ref_str) = t_ctx.func.refinements.get(&v) {
                            arg_refinements.insert(i, ref_str.clone());
                        }
                    }
                    Some(lila_ir::registry::FunctionSignature {
                        name: target_name.clone(),
                        arg_types,
                        arg_refinements,
                        return_type: t_ctx.func.return_type.clone(),
                        return_refinement: t_ctx.func.ret_refinement.clone(),
                        pointer: 0,
                    })
                } else {
                    registry.get(target_name).cloned()
                };

                if let Some(sig) = sig {
                    let mut call_dim_map: std::collections::HashMap<String, B::Int> = std::collections::HashMap::new();
                    for (i, arg_val) in args.iter().enumerate() {
                        let arg_ty = t_ctx.func.get_type(*arg_val);
                        if i < sig.arg_types.len() {
                            let target_ty = &sig.arg_types[i];
                            if let (Type::Tensor(_, src_dims), Type::Tensor(_, target_dims)) = (&arg_ty, target_ty) {
                                if src_dims.len() != target_dims.len() {
                                    let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                    return Err(format!("Tensor rank mismatch in call to '{}': expected {} dims, got {}{}", target_name, target_dims.len(), src_dims.len(), loc_info));
                                }
                                if let Some(src_z3_dims) = t_ctx.z3_tensor_dims.get(arg_val) {
                                    for (dim_idx, target_dim_name) in target_dims.iter().enumerate() {
                                        let src_z3_dim = &src_z3_dims[dim_idx];
                                        if let Some(bound_z3_dim) = call_dim_map.get(target_dim_name) {
                                            t_ctx.backend.push();
                                            t_ctx.backend.assert(path_cond);
                                            let eq = t_ctx.backend.int_eq(src_z3_dim, bound_z3_dim);
                                            let not_eq = t_ctx.backend.bool_not(&eq);
                                            t_ctx.backend.assert(&not_eq);
                                            
                                            if t_ctx.backend.check()? {
                                                let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                                return Err(format!("Tensor shape mismatch in call to '{}': dimension '{}' (idx {}) does not match previously bound value{}", target_name, target_dim_name, dim_idx, loc_info));
                                            }
                                            t_ctx.backend.pop(1);
                                        } else {
                                            call_dim_map.insert(target_dim_name.clone(), src_z3_dim.clone());
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(ref_str) = sig.arg_refinements.get(&i) {
                            let arg_ty = t_ctx.func.get_type(*arg_val);
                            let res = if let Some(z3_bv) = t_ctx.z3_bvs.get(arg_val) {
                                let bv_int = t_ctx.backend.bv_to_int(z3_bv, arg_ty.is_signed());
                                crate::refinement_parser::parse_refinement(
                                    ref_str,
                                    &bv_int,
                                    Some(z3_bv),
                                )
                            } else if let Some(z3_int) = t_ctx.z3_ints.get(arg_val) {
                                crate::refinement_parser::parse_refinement(ref_str, z3_int, None)
                            } else if let Some(z3_float) = t_ctx.z3_floats.get(arg_val) {
                                crate::refinement_parser::parse_float_refinement(ref_str, z3_float)
                            } else {
                                continue;
                            };

                            if let Ok(expr) = res {
                                t_ctx.backend.push();
                                t_ctx.backend.assert(path_cond);
                                let __tmp = t_ctx.backend.bool_not(&expr);
                                t_ctx.backend.assert(&__tmp);
                                if t_ctx.backend.check()? {
                                    let loc_info = inst
                                        .location
                                        .map(|l| format!(" at {}", l))
                                        .unwrap_or_default();
                                    return Err(format!(
                                        "Argument refinement violation for function '{}' (arg {}): value of {:?} does not satisfy '{}' and may be violated on some reachable path{}.",
                                        target_name, i, arg_val, ref_str, loc_info
                                    ));
                                }
                                t_ctx.backend.pop(1);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
