use lila_ir::analysis::interval::{Bound, IntervalAnalysisResults};
use lila_ir::ir::{BlockId, Function, InstructionKind, Type, Value};
use std::collections::HashMap;

use z3::ast::{Array, Bool, Float, Int, BV};
use z3::{Context, Solver};

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
        _ => {
            offsets.push((base_offset, ty.clone()));
        }
    }
}

pub fn copy_composite_z3<'a>(
    t_ctx: &TranslationContext<'a>,
    current_state: Array,
    src_val: Value,
    src_ty: &Type,
    dest_base_offset: i64,
) -> Array {
    let mut new_state = current_state;
    let mut leaves = Vec::new();
    get_leaf_offsets(src_ty, &t_ctx.func.struct_layouts, 0, &mut leaves);

    if let Some(src_array) = t_ctx.z3_arrays.get(&src_val) {
        for (offset, _leaf_ty) in leaves {
            let src_offset_z3 = Int::from_i64(offset as i64);
            let dest_offset_z3 = Int::from_i64(dest_base_offset + offset as i64);
            let val = src_array.select(&src_offset_z3);
            new_state = new_state.store(&dest_offset_z3, &val);
        }
    }
    new_state
}

pub struct TranslationContext<'a> {
    pub ctx: &'a Context,
    pub solver: &'a Solver,
    pub func: &'a Function,
    pub analysis: &'a IntervalAnalysisResults,
    pub uid: usize,
    pub z3_ints: HashMap<Value, Int>, // Kept for refinement parsing
    pub z3_floats: HashMap<Value, Float>,
    pub z3_bvs: HashMap<Value, BV>,
    pub z3_arrays: HashMap<Value, Array>,
    pub z3_perms: HashMap<Value, z3::ast::Real>,
    pub tuple_mappings: HashMap<Value, Vec<Value>>,
    pub block_conditions: HashMap<BlockId, Bool>,
    pub edge_conditions: HashMap<(BlockId, BlockId), Bool>,
    pub has_refinements: bool,
}

pub fn verify_with_context(
    ctx: &Context,
    solver: &Solver,
    func: &Function,
    analysis: &IntervalAnalysisResults,
    liveness: lila_ir::analysis::liveness::LivenessAnalysisResults,
    perm_verifier: crate::permissions::PermissionVerifier,
    uid: usize,
) -> Result<(), String> {
    let mut t_ctx = TranslationContext {
        ctx,
        solver,
        func,
        analysis,
        uid,
        z3_ints: HashMap::new(),
        z3_floats: HashMap::new(),
        z3_bvs: HashMap::new(),
        z3_arrays: HashMap::new(),
        z3_perms: HashMap::new(),
        tuple_mappings: HashMap::new(),
        block_conditions: HashMap::new(),
        edge_conditions: HashMap::new(),
        has_refinements: func.ret_refinement.is_some(),
    };

    // 1. Initialize Z3 values for all SSA values
    memory::init_values(&mut t_ctx)?;

    init_permissions(&mut t_ctx);

    assert_cfg_constraints(&mut t_ctx, solver);

    translate_instructions(&mut t_ctx)?;

    assert_derived_intervals(&t_ctx, solver);

    // 7. Generate Fractional Permission Assertions (Must be AFTER path constraints and instructions are translated)
    perm_verifier.generate_assertions(
        solver,
        &liveness,
        &t_ctx.z3_perms,
        &t_ctx.block_conditions,
    )?;

    verify_return_refinements(&t_ctx, solver)?;

    verify_call_arguments(&t_ctx, solver)?;

    // 11. Final Consistency Check
    if !t_ctx.has_refinements && t_ctx.z3_perms.is_empty() {
        tracing::info!(target: "lila::verify::z3", "Skipping final consistency check for '{}' (no refinements or pointers).", func.name);
    } else {
        tracing::info!(target: "lila::verify::z3", "Performing final consistency check for '{}'...", func.name);
        if solver.check() == z3::SatResult::Unsat {
            return Err(
                "Formal verification failed: Logical contradiction or permission conflict detected."
                    .to_string(),
            );
        }
    }

    tracing::info!(target: "lila::verify::z3", "Proof successful for '{}'", func.name);
    Ok(())
}

fn assert_derived_intervals(t_ctx: &TranslationContext, solver: &Solver) {
    // 6. Assert derived intervals and refinements
    for (val, interval) in &t_ctx.analysis.intervals {
        if let Some(ty) = t_ctx.func.value_types.get(val) {
            if let Some(z3_bv) = t_ctx.z3_bvs.get(val) {
                if let Some(bit_width) = ty.int_bit_width() {
                    let is_signed = ty.is_signed();
                    if let Bound::Finite(low) = interval.low {
                        let low_bv = BV::from_i64(low as i64, bit_width);
                        if is_signed {
                            solver.assert(z3_bv.bvsge(&low_bv));
                        } else {
                            solver.assert(z3_bv.bvuge(&low_bv));
                        }
                    }
                    if let Bound::Finite(high) = interval.high {
                        let high_bv = BV::from_i64(high as i64, bit_width);
                        if is_signed {
                            solver.assert(z3_bv.bvsle(&high_bv));
                        } else {
                            solver.assert(z3_bv.bvule(&high_bv));
                        }
                    }
                }
            } else if let Some(z3_float) = t_ctx.z3_floats.get(val) {
                if let Bound::Finite(low) = interval.low {
                    let low_float = if matches!(ty, Type::F32) {
                        Float::from_f32(low as f32)
                    } else {
                        Float::from_f64(low)
                    };
                    solver.assert(z3_float.ge(&low_float));
                }
                if let Bound::Finite(high) = interval.high {
                    let high_float = if matches!(ty, Type::F32) {
                        Float::from_f32(high as f32)
                    } else {
                        Float::from_f64(high)
                    };
                    solver.assert(z3_float.le(&high_float));
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
                            let low_bv = BV::from_i64(low as i64, bit_width);
                            if is_signed {
                                solver.assert(path_cond.implies(z3_bv.bvsge(&low_bv)));
                            } else {
                                solver.assert(path_cond.implies(z3_bv.bvuge(&low_bv)));
                            }
                        }
                        if let Bound::Finite(high) = interval.high {
                            let high_bv = BV::from_i64(high as i64, bit_width);
                            if is_signed {
                                solver.assert(path_cond.implies(z3_bv.bvsle(&high_bv)));
                            } else {
                                solver.assert(path_cond.implies(z3_bv.bvule(&high_bv)));
                            }
                        }
                    }
                } else if let Some(z3_float) = t_ctx.z3_floats.get(val) {
                    if let Bound::Finite(low) = interval.low {
                        let low_float = if matches!(ty, Type::F32) {
                            Float::from_f32(low as f32)
                        } else {
                            Float::from_f64(low)
                        };
                        solver.assert(path_cond.implies(z3_float.ge(&low_float)));
                    }
                    if let Bound::Finite(high) = interval.high {
                        let high_float = if matches!(ty, Type::F32) {
                            Float::from_f32(high as f32)
                        } else {
                            Float::from_f64(high)
                        };
                        solver.assert(path_cond.implies(z3_float.le(&high_float)));
                    }
                }
            }
        }
    }
}

fn init_permissions(t_ctx: &mut TranslationContext) {
    let has_parallel = t_ctx.func.blocks.iter().any(|b| {
        b.instructions
            .iter()
            .any(|i| matches!(i.kind, InstructionKind::ParallelFor { .. }))
    });

    if !has_parallel {
        return;
    }

    // 2. Initialize Permission Variables for Fractional Permission Model
    for i in 0..t_ctx.func.value_count {
        let v = Value(i);
        let ty = t_ctx.func.get_type(v);
        if ty.is_pointer_like() {
            let p_var =
                z3::ast::Real::new_const(format!("{}_perm_v{}_{}", t_ctx.func.name, i, t_ctx.uid));
            t_ctx.z3_perms.insert(v, p_var);
        }
    }
}

fn assert_cfg_constraints(t_ctx: &mut TranslationContext, solver: &Solver) {
    // 3. Declare Booleans for all blocks and known edges
    for block in &t_ctx.func.blocks {
        let b_cond = Bool::new_const(format!(
            "{}_block_{}_{}",
            t_ctx.func.name, block.id.0, t_ctx.uid
        ));
        t_ctx.block_conditions.insert(block.id, b_cond);

        if let Some(last_inst) = block.instructions.last() {
            match &last_inst.kind {
                InstructionKind::Jump(target) => {
                    let e_cond = Bool::new_const(format!(
                        "{}_edge_{}_{}_{}",
                        t_ctx.func.name, block.id.0, target.0, t_ctx.uid
                    ));
                    t_ctx.edge_conditions.insert((block.id, *target), e_cond);
                }
                InstructionKind::Branch(_, t_block, f_block) => {
                    let et_cond = Bool::new_const(format!(
                        "{}_edge_{}_{}_{}",
                        t_ctx.func.name, block.id.0, t_block.0, t_ctx.uid
                    ));
                    t_ctx.edge_conditions.insert((block.id, *t_block), et_cond);
                    let ef_cond = Bool::new_const(format!(
                        "{}_edge_{}_{}_{}",
                        t_ctx.func.name, block.id.0, f_block.0, t_ctx.uid
                    ));
                    t_ctx.edge_conditions.insert((block.id, *f_block), ef_cond);
                }
                _ => {}
            }
        }

        // Handle implicit edges from ParallelFor
        for inst in &block.instructions {
            if let InstructionKind::ParallelFor { body_block, .. } = &inst.kind {
                let e_cond = Bool::new_const(format!(
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
    let true_cond = Bool::from_bool(true);
    let false_cond = Bool::from_bool(false);
    if let Some(entry_cond) = t_ctx.block_conditions.get(&t_ctx.func.entry_block) {
        solver.assert(entry_cond.eq(&true_cond));
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
                solver.assert(path_cond.eq(&false_cond));
            } else {
                let or_expr = Bool::or(&incoming_edges.to_vec());
                solver.assert(path_cond.eq(&or_expr));
            }
        }
    }
}

fn translate_instructions(t_ctx: &mut TranslationContext) -> Result<(), String> {
    // 5. Translate Instructions (Arithmetic, Memory, Control Flow)
    tracing::info!(target: "lila::verify::z3", "Translating instructions for '{}'...", t_ctx.func.name);
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
                | InstructionKind::Not(..) => {
                    arithmetic::translate(t_ctx, inst, &path_cond)?;
                }
                InstructionKind::Jump(_)
                | InstructionKind::Branch(..)
                | InstructionKind::Phi(..) => {
                    control_flow::translate(t_ctx, inst, &path_cond, block.id)?;
                }
                InstructionKind::ArrayLoad(..)
                | InstructionKind::ArrayStore(..)
                | InstructionKind::BufferLoad(..)
                | InstructionKind::BufferStore(..)
                | InstructionKind::BufferLen(..)
                | InstructionKind::StructCreate(..)
                | InstructionKind::StructLoad(..)
                | InstructionKind::StructOffset(..)
                | InstructionKind::StructSet(..)
                | InstructionKind::EnumCreate(..)
                | InstructionKind::EnumIsVariant(..)
                | InstructionKind::EnumExtract(..)
                | InstructionKind::Peek(..)
                | InstructionKind::Hand(..) => {
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
                InstructionKind::ParallelFor {
                    index_var,
                    start,
                    stop,
                    body_block,
                    ..
                } => {
                    if let Some(edge_p) = t_ctx.edge_conditions.get(&(block.id, *body_block)) {
                        t_ctx.solver.assert(edge_p.eq(&path_cond));
                    }

                    if let Some(body_cond) = t_ctx.block_conditions.get(body_block) {
                        let idx_int = if let Some(bv) = t_ctx.z3_bvs.get(index_var) {
                            bv.to_int(true)
                        } else {
                            t_ctx.z3_ints.get(index_var).unwrap().clone()
                        };
                        let start_int = if let Some(bv) = t_ctx.z3_bvs.get(start) {
                            bv.to_int(true)
                        } else {
                            t_ctx.z3_ints.get(start).unwrap().clone()
                        };
                        let stop_int = if let Some(bv) = t_ctx.z3_bvs.get(stop) {
                            bv.to_int(true)
                        } else {
                            t_ctx.z3_ints.get(stop).unwrap().clone()
                        };

                        // In the loop body, start <= index < stop
                        t_ctx
                            .solver
                            .assert(body_cond.implies(idx_int.ge(&start_int)));
                        t_ctx
                            .solver
                            .assert(body_cond.implies(idx_int.lt(&stop_int)));
                    }
                }
                InstructionKind::Nop | InstructionKind::Release(_) => {}
            }

            // Translate logical constraints attached to the instruction
            translate_constraints(t_ctx, inst, &path_cond)?;
        }
    }
    Ok(())
}

fn translate_constraints(
    t_ctx: &mut TranslationContext,
    inst: &lila_ir::ir::Instruction,
    path_cond: &Bool,
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
        t_ctx.solver.assert(path_cond.implies(&z3_constraint));
    }

    Ok(())
}

fn verify_return_refinements(t_ctx: &TranslationContext, solver: &Solver) -> Result<(), String> {
    // 8. Verify Return Refinements
    if let Some(ret_ref) = &t_ctx.func.ret_refinement {
        for block in &t_ctx.func.blocks {
            let path_cond = t_ctx.block_conditions.get(&block.id).unwrap();
            for inst in &block.instructions {
                if let InstructionKind::Return(Some(ret_val)) = &inst.kind {
                    let ty = t_ctx.func.get_type(*ret_val);
                    let res = if let Some(z3_bv) = t_ctx.z3_bvs.get(ret_val) {
                        crate::refinement_parser::parse_refinement(
                            ret_ref,
                            &z3_bv.to_int(ty.is_signed()),
                            Some(z3_bv),
                        )
                    } else if let Some(z3_int) = t_ctx.z3_ints.get(ret_val) {
                        crate::refinement_parser::parse_refinement(ret_ref, z3_int, None)
                    } else if let Some(z3_float) = t_ctx.z3_floats.get(ret_val) {
                        crate::refinement_parser::parse_float_refinement(ret_ref, z3_float)
                    } else {
                        continue;
                    };

                    if let Ok(expr) = res {
                        solver.push();
                        solver.assert(path_cond);
                        solver.assert(expr.not());
                        if solver.check() != z3::SatResult::Unsat {
                            let loc_info = inst
                                .location
                                .map(|l| format!(" at {}", l))
                                .unwrap_or_default();
                            return Err(format!(
                                "Return refinement violation: value of {:?} does not satisfy '{}' and may be violated on some reachable path{}.",
                                ret_val, ret_ref, loc_info
                            ));
                        }
                        solver.pop(1);
                    }
                }
            }
        }
    }

    Ok(())
}

fn verify_call_arguments(t_ctx: &TranslationContext, solver: &Solver) -> Result<(), String> {
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
                    for (i, arg_val) in args.iter().enumerate() {
                        if let Some(ref_str) = sig.arg_refinements.get(&i) {
                            let arg_ty = t_ctx.func.get_type(*arg_val);
                            let res = if let Some(z3_bv) = t_ctx.z3_bvs.get(arg_val) {
                                crate::refinement_parser::parse_refinement(
                                    ref_str,
                                    &z3_bv.to_int(arg_ty.is_signed()),
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
                                solver.push();
                                solver.assert(path_cond);
                                solver.assert(expr.not());
                                if solver.check() != z3::SatResult::Unsat {
                                    let loc_info = inst
                                        .location
                                        .map(|l| format!(" at {}", l))
                                        .unwrap_or_default();
                                    return Err(format!(
                                        "Argument refinement violation for function '{}' (arg {}): value of {:?} does not satisfy '{}' and may be violated on some reachable path{}.",
                                        target_name, i, arg_val, ref_str, loc_info
                                    ));
                                }
                                solver.pop(1);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
