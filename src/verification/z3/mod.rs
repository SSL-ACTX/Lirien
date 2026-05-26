use crate::ssa::analysis::interval::{Bound, IntervalAnalysisResults};
use crate::ssa::ir::{BlockId, Function, InstructionKind, Type, Value};
use std::collections::HashMap;

use z3::ast::{Array, Bool, Float, Int, BV};
use z3::{Context, Solver};

pub mod arithmetic;
pub mod control_flow;
pub mod memory;
pub mod tuples;

pub struct TranslationContext<'a> {
    pub ctx: &'a Context,
    pub solver: &'a Solver,
    pub func: &'a Function,
    pub uid: usize,
    pub z3_ints: HashMap<Value, Int>, // Kept for refinement parsing
    pub z3_floats: HashMap<Value, Float>,
    pub z3_bvs: HashMap<Value, BV>,
    pub z3_arrays: HashMap<Value, Array>,
    pub z3_perms: HashMap<Value, z3::ast::Real>,
    pub tuple_mappings: HashMap<Value, Vec<Value>>,
    pub block_conditions: HashMap<BlockId, Bool>,
    pub edge_conditions: HashMap<(BlockId, BlockId), Bool>,
}

pub fn verify_with_context(
    ctx: &Context,
    solver: &Solver,
    func: &Function,
    analysis: IntervalAnalysisResults,
    liveness: crate::ssa::analysis::liveness::LivenessAnalysisResults,
    perm_verifier: crate::verification::permissions::PermissionVerifier,
    uid: usize,
) -> Result<(), String> {
    let mut t_ctx = TranslationContext {
        ctx,
        solver,
        func,
        uid,
        z3_ints: HashMap::new(),
        z3_floats: HashMap::new(),
        z3_bvs: HashMap::new(),
        z3_arrays: HashMap::new(),
        z3_perms: HashMap::new(),
        tuple_mappings: HashMap::new(),
        block_conditions: HashMap::new(),
        edge_conditions: HashMap::new(),
    };

    // 1. Initialize Z3 values for all SSA values
    memory::init_values(&mut t_ctx)?;

    // 2. Initialize Permission Variables for Fractional Permission Model
    for i in 0..func.value_count {
        let v = Value(i);
        let ty = func.get_type(v);
        if matches!(
            ty,
            Type::Ref(_) | Type::Mut(_) | Type::Owned(_) | Type::Buffer(_) | Type::Array(_, None)
        ) {
            let p_var = z3::ast::Real::new_const(format!("{}_perm_v{}_{}", func.name, i, uid));
            t_ctx.z3_perms.insert(v, p_var);
        }
    }

    // 3. Declare Booleans for all blocks and known edges
    for block in &func.blocks {
        let b_cond = Bool::new_const(format!("{}_block_{}_{}", func.name, block.id.0, uid));
        t_ctx.block_conditions.insert(block.id, b_cond);

        if let Some(last_inst) = block.instructions.last() {
            match &last_inst.kind {
                InstructionKind::Jump(target) => {
                    let e_cond = Bool::new_const(format!(
                        "{}_edge_{}_{}_{}",
                        func.name, block.id.0, target.0, uid
                    ));
                    t_ctx.edge_conditions.insert((block.id, *target), e_cond);
                }
                InstructionKind::Branch(_, t_block, f_block) => {
                    let et_cond = Bool::new_const(format!(
                        "{}_edge_{}_{}_{}",
                        func.name, block.id.0, t_block.0, uid
                    ));
                    t_ctx.edge_conditions.insert((block.id, *t_block), et_cond);
                    let ef_cond = Bool::new_const(format!(
                        "{}_edge_{}_{}_{}",
                        func.name, block.id.0, f_block.0, uid
                    ));
                    t_ctx.edge_conditions.insert((block.id, *f_block), ef_cond);
                }
                _ => {}
            }
        }
    }

    // 4. Assert Structural CFG Constraints
    let true_cond = Bool::from_bool(true);
    let false_cond = Bool::from_bool(false);
    if let Some(entry_cond) = t_ctx.block_conditions.get(&func.entry_block) {
        solver.assert(entry_cond.eq(&true_cond));
    }
    for block in &func.blocks {
        let path_cond = t_ctx.block_conditions.get(&block.id).unwrap().clone();
        if block.id != func.entry_block {
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

    // 5. Translate Instructions (Arithmetic, Memory, Control Flow)
    for block in &func.blocks {
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
                    arithmetic::translate(&mut t_ctx, inst, &path_cond)?;
                }
                InstructionKind::Jump(_)
                | InstructionKind::Branch(..)
                | InstructionKind::Phi(..) => {
                    control_flow::translate(&mut t_ctx, inst, &path_cond, block.id)?;
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
                | InstructionKind::Reference(..)
                | InstructionKind::MutReference(..) => {
                    memory::translate(&mut t_ctx, inst, &path_cond)?;
                }
                InstructionKind::TupleCreate(..) | InstructionKind::TupleExtract(..) => {
                    tuples::translate(&mut t_ctx, inst, &path_cond)?;
                }
                _ => {}
            }
        }
    }

    // 6. Assert derived intervals and refinements
    for (val, interval) in analysis.intervals {
        if let (Some(z3_bv), Some(ty)) = (t_ctx.z3_bvs.get(&val), t_ctx.func.value_types.get(&val))
        {
            if let Some(bit_width) = ty.int_bit_width() {
                if let Bound::Finite(low) = interval.low {
                    solver.assert(z3_bv.bvsge(BV::from_i64(low as i64, bit_width)));
                }
                if let Bound::Finite(high) = interval.high {
                    solver.assert(z3_bv.bvsle(BV::from_i64(high as i64, bit_width)));
                }
            }
        }
    }
    for ((val, b_id), interval) in &analysis.block_narrowing {
        if let Some(path_cond) = t_ctx.block_conditions.get(b_id) {
            if let (Some(z3_bv), Some(ty)) =
                (t_ctx.z3_bvs.get(val), t_ctx.func.value_types.get(val))
            {
                if let Some(bit_width) = ty.int_bit_width() {
                    if let Bound::Finite(low) = interval.low {
                        solver.assert(
                            path_cond.implies(z3_bv.bvsge(BV::from_i64(low as i64, bit_width))),
                        );
                    }
                    if let Bound::Finite(high) = interval.high {
                        solver.assert(
                            path_cond.implies(z3_bv.bvsle(BV::from_i64(high as i64, bit_width))),
                        );
                    }
                }
            }
        }
    }

    // 7. Generate Fractional Permission Assertions (Must be AFTER path constraints and instructions are translated)
    perm_verifier.generate_assertions(
        solver,
        &liveness,
        &t_ctx.z3_perms,
        &t_ctx.block_conditions,
    )?;

    // 8. Final Consistency Check
    if solver.check() == z3::SatResult::Unsat {
        return Err(
            "Formal verification failed: Logical contradiction or permission conflict detected."
                .to_string(),
        );
    }

    tracing::info!(target: "lila::verify::z3", "Proof successful for '{}'", func.name);
    Ok(())
}
