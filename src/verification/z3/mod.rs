use crate::ssa::analysis::interval::{Bound, IntervalAnalysisResults};
use crate::ssa::ir::{BlockId, Function, InstructionKind, Value};
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
    pub z3_ints: HashMap<Value, Int>, // Kept for refinement parsing
    pub z3_floats: HashMap<Value, Float>,
    pub z3_bvs: HashMap<Value, BV>,
    pub z3_arrays: HashMap<Value, Array>,
    pub tuple_mappings: HashMap<Value, Vec<Value>>,
    pub block_conditions: HashMap<BlockId, Bool>,
    pub edge_conditions: HashMap<(BlockId, BlockId), Bool>,
}

pub fn verify_with_context(
    ctx: &Context,
    solver: &Solver,
    func: &Function,
    analysis: IntervalAnalysisResults,
) -> Result<(), String> {
    let mut t_ctx = TranslationContext {
        ctx,
        solver,
        func,
        z3_ints: HashMap::new(),
        z3_floats: HashMap::new(),
        z3_bvs: HashMap::new(),
        z3_arrays: HashMap::new(),
        tuple_mappings: HashMap::new(),
        block_conditions: HashMap::new(),
        edge_conditions: HashMap::new(),
    };

    // Initialize Z3 values for all SSA values
    memory::init_values(&mut t_ctx)?;

    // Assert derived intervals
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

    // Declare Booleans for all blocks and known edges
    for block in &func.blocks {
        let b_cond = Bool::new_const(format!("block_{}", block.id.0));
        t_ctx.block_conditions.insert(block.id, b_cond);

        // Find outgoing edges
        if let Some(last_inst) = block.instructions.last() {
            match &last_inst.kind {
                InstructionKind::Jump(target) => {
                    let e_cond = Bool::new_const(format!("edge_{}_{}", block.id.0, target.0));
                    t_ctx.edge_conditions.insert((block.id, *target), e_cond);
                }
                InstructionKind::Branch(_, t_block, f_block) => {
                    let et_cond = Bool::new_const(format!("edge_{}_{}", block.id.0, t_block.0));
                    t_ctx.edge_conditions.insert((block.id, *t_block), et_cond);
                    let ef_cond = Bool::new_const(format!("edge_{}_{}", block.id.0, f_block.0));
                    t_ctx.edge_conditions.insert((block.id, *f_block), ef_cond);
                }
                _ => {}
            }
        }
    }

    // Assert Structural CFG Constraints
    let true_cond = Bool::from_bool(true);
    let false_cond = Bool::from_bool(false);

    // Entry block is always true
    if let Some(entry_cond) = t_ctx.block_conditions.get(&func.entry_block) {
        solver.assert(entry_cond.eq(&true_cond));
    }

    for block in &func.blocks {
        let path_cond = t_ctx.block_conditions.get(&block.id).unwrap().clone();

        // Block condition == OR of incoming edges (except entry block)
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

    // Assert block-specific narrowing
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

    // Translate Instructions
    for block in &func.blocks {
        let path_cond = t_ctx.block_conditions.get(&block.id).unwrap().clone();

        for inst in &block.instructions {
            match &inst.kind {
                crate::ssa::ir::InstructionKind::Add(_, _, _)
                | crate::ssa::ir::InstructionKind::Sub(_, _, _)
                | crate::ssa::ir::InstructionKind::Mul(_, _, _)
                | crate::ssa::ir::InstructionKind::SDiv(_, _, _)
                | crate::ssa::ir::InstructionKind::UDiv(_, _, _)
                | crate::ssa::ir::InstructionKind::SRem(_, _, _)
                | crate::ssa::ir::InstructionKind::URem(_, _, _)
                | crate::ssa::ir::InstructionKind::FAdd(_, _, _)
                | crate::ssa::ir::InstructionKind::FSub(_, _, _)
                | crate::ssa::ir::InstructionKind::FMul(_, _, _)
                | crate::ssa::ir::InstructionKind::FDiv(_, _, _)
                | crate::ssa::ir::InstructionKind::FSqrt(_, _)
                | crate::ssa::ir::InstructionKind::FSin(_, _)
                | crate::ssa::ir::InstructionKind::FCos(_, _)
                | crate::ssa::ir::InstructionKind::FPow(_, _, _)
                | crate::ssa::ir::InstructionKind::ConstInt(_, _)
                | crate::ssa::ir::InstructionKind::ConstFloat(_, _)
                | crate::ssa::ir::InstructionKind::Eq(_, _, _)
                | crate::ssa::ir::InstructionKind::Ne(_, _, _)
                | crate::ssa::ir::InstructionKind::SLt(_, _, _)
                | crate::ssa::ir::InstructionKind::SLe(_, _, _)
                | crate::ssa::ir::InstructionKind::SGt(_, _, _)
                | crate::ssa::ir::InstructionKind::SGe(_, _, _)
                | crate::ssa::ir::InstructionKind::ULt(_, _, _)
                | crate::ssa::ir::InstructionKind::ULe(_, _, _)
                | crate::ssa::ir::InstructionKind::UGt(_, _, _)
                | crate::ssa::ir::InstructionKind::UGe(_, _, _)
                | crate::ssa::ir::InstructionKind::FLt(_, _, _)
                | crate::ssa::ir::InstructionKind::FLe(_, _, _)
                | crate::ssa::ir::InstructionKind::FGt(_, _, _)
                | crate::ssa::ir::InstructionKind::FGe(_, _, _)
                | crate::ssa::ir::InstructionKind::And(_, _, _)
                | crate::ssa::ir::InstructionKind::Or(_, _, _)
                | crate::ssa::ir::InstructionKind::Xor(_, _, _)
                | crate::ssa::ir::InstructionKind::Shl(_, _, _)
                | crate::ssa::ir::InstructionKind::LShr(_, _, _)
                | crate::ssa::ir::InstructionKind::AShr(_, _, _)
                | crate::ssa::ir::InstructionKind::IToF(_, _, _)
                | crate::ssa::ir::InstructionKind::FToI(_, _, _)
                | crate::ssa::ir::InstructionKind::Not(_, _) => {
                    arithmetic::translate(&mut t_ctx, inst, &path_cond)?;
                }

                crate::ssa::ir::InstructionKind::Jump(_)
                | crate::ssa::ir::InstructionKind::Branch(_, _, _)
                | crate::ssa::ir::InstructionKind::Phi(_, _) => {
                    control_flow::translate(&mut t_ctx, inst, &path_cond, block.id)?;
                }

                crate::ssa::ir::InstructionKind::ArrayLoad(_, _, _)
                | crate::ssa::ir::InstructionKind::ArrayStore(_, _, _, _, _)
                | crate::ssa::ir::InstructionKind::BufferLoad(_, _, _)
                | crate::ssa::ir::InstructionKind::BufferStore(_, _, _, _, _)
                | crate::ssa::ir::InstructionKind::BufferLen(_, _)
                | crate::ssa::ir::InstructionKind::StructCreate(_, _, _)
                | crate::ssa::ir::InstructionKind::StructLoad(_, _, _)
                | crate::ssa::ir::InstructionKind::StructOffset(_, _, _)
                | crate::ssa::ir::InstructionKind::StructSet(_, _, _, _, _)
                | crate::ssa::ir::InstructionKind::EnumCreate(_, _, _, _)
                | crate::ssa::ir::InstructionKind::EnumIsVariant(_, _, _)
                | crate::ssa::ir::InstructionKind::EnumExtract(_, _, _)
                | crate::ssa::ir::InstructionKind::Borrow(_, _)
                | crate::ssa::ir::InstructionKind::MutBorrow(_, _) => {
                    memory::translate(&mut t_ctx, inst, &path_cond)?;
                }
                crate::ssa::ir::InstructionKind::TupleCreate(_, _)
                | crate::ssa::ir::InstructionKind::TupleExtract(_, _, _) => {
                    tuples::translate(&mut t_ctx, inst, &path_cond)?;
                }
                _ => {}
            }
        }
    }

    tracing::info!(target: "lila::verify::z3", "Proof successful for '{}'", func.name);
    Ok(())
}
