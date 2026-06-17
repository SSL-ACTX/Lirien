use crate::backend::SolverBackend;
use lila_ir::analysis::interval::IntervalAnalysisResults;
use lila_ir::ir::{BlockId, Function, InstructionKind, Type, Value};
use std::collections::HashMap;

pub mod arithmetic;
pub mod calls;
pub mod cfg;
pub mod control_flow;
pub mod intervals;
pub mod memory;
pub mod returns;
pub mod tuples;

pub fn get_leaf_offsets(
    ty: &Type,
    struct_layouts: &HashMap<String, Vec<(String, Type)>>,
    base_offset: usize,
    offsets: &mut Vec<(usize, Type)>,
) {
    match ty {
        Type::Struct(name) | Type::TypedDict(name) | Type::NamedTuple(name) => {
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
    pub array_offsets: HashMap<Value, B::Int>,
    pub tuple_mappings: HashMap<Value, Vec<Value>>,
    pub block_conditions: HashMap<BlockId, B::Bool>,
    pub edge_conditions: HashMap<(BlockId, BlockId), B::Bool>,
    pub has_refinements: bool,
}

impl<'a, B: SolverBackend> TranslationContext<'a, B> {
    pub fn get_dim_var(&mut self, dim_name: &str) -> B::Int {
        self.backend.int_const(dim_name)
    }
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
) -> Result<Option<String>, String> {
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
        array_offsets: HashMap::new(),
        tuple_mappings: HashMap::new(),
        block_conditions: HashMap::new(),
        edge_conditions: HashMap::new(),
        has_refinements: func.ret_refinement.is_some(),
    };

    memory::init_values(&mut t_ctx)?;

    cfg::assert_cfg_constraints(&mut t_ctx);

    translate_instructions(&mut t_ctx)?;

    intervals::assert_derived_intervals(&mut t_ctx);

    returns::verify_return_refinements(&mut t_ctx)?;

    calls::verify_call_arguments(&mut t_ctx)?;

    if !t_ctx.has_refinements {
        tracing::info!(target: "lila::verify::verifier", "Skipping final consistency check for '{}' (no refinements).", func.name);
    } else {
        tracing::info!(target: "lila::verify::verifier", "Performing final consistency check for '{}'...", func.name);
        if !t_ctx.backend.check()? {
            return Err("Formal verification failed: Logical contradiction detected.".to_string());
        }
    }

    let inferred = if func.ret_refinement == Some("...".to_string()) {
        returns::infer_return_refinement(&t_ctx)?
    } else {
        None
    };

    tracing::info!(target: "lila::verify::verifier", "Proof successful for '{}' (inferred: {:?})", func.name, inferred);
    Ok(inferred)
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
                | InstructionKind::FTan(..)
                | InstructionKind::FAsin(..)
                | InstructionKind::FAcos(..)
                | InstructionKind::FAtan(..)
                | InstructionKind::FExp(..)
                | InstructionKind::FLog(..)
                | InstructionKind::FLog10(..)
                | InstructionKind::FPow(..)
                | InstructionKind::FFloor(..)
                | InstructionKind::FCeil(..)
                | InstructionKind::FTrunc(..)
                | InstructionKind::FNearest(..)
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
                | InstructionKind::ArraySlice(..)
                | InstructionKind::BufferLoad(..)
                | InstructionKind::BufferStore(..)
                | InstructionKind::TensorLoad(..)
                | InstructionKind::TensorStore(..)
                | InstructionKind::TensorDim(..)
                | InstructionKind::TensorBroadcast(..)
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
                    tuples::translate(t_ctx, inst, &path_cond);
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
    use crate::refinement::{parse_bool_expr_with_resolver, Resolver};

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
