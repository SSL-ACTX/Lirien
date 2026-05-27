use super::TranslationContext;
use crate::ssa::ir::{BlockId, Instruction, InstructionKind};
use std::collections::VecDeque;
use z3::ast::{Bool, Int, BV};

pub fn translate(
    ctx: &mut TranslationContext,
    inst: &Instruction,
    path_cond: &Bool,
    current_block_id: BlockId,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::Branch(cond, t_block, f_block) => {
            let cond_is_true = if let Some(z3_bv_cond) = ctx.z3_bvs.get(cond) {
                let bit_width = ctx.func.get_type(*cond).int_bit_width().unwrap_or(64);
                let zero = BV::from_i64(0, bit_width);
                z3_bv_cond.eq(&zero).not()
            } else if let Some(z3_int_cond) = ctx.z3_ints.get(cond) {
                let zero = Int::from_i64(0);
                z3_int_cond.eq(&zero).not()
            } else {
                return Err(format!(
                    "Value {} not modeled in Z3 (required for branch)",
                    cond
                ));
            };

            let true_cond = Bool::and(&[path_cond, &cond_is_true]);
            let false_cond = Bool::and(&[path_cond, &cond_is_true.not()]);

            if let Some(edge_t) = ctx.edge_conditions.get(&(current_block_id, *t_block)) {
                ctx.solver.assert(edge_t.eq(&true_cond));
            }
            if let Some(edge_f) = ctx.edge_conditions.get(&(current_block_id, *f_block)) {
                ctx.solver.assert(edge_f.eq(&false_cond));
            }
        }
        InstructionKind::Jump(target) => {
            if let Some(edge_j) = ctx.edge_conditions.get(&(current_block_id, *target)) {
                ctx.solver.assert(edge_j.eq(path_cond));
            }
        }
        InstructionKind::Phi(dest, mappings) => {
            for (pred_id, src_val) in mappings {
                // To avoid logical contradictions like 'v = v + 1' in loop back-edges,
                // we only assert equality for forward edges. Back-edges are handled
                // by the inductive interval analysis results.
                if is_reachable(ctx, current_block_id, *pred_id) {
                    continue; // Skip back-edge equality
                }

                if let Some(edge_cond) = ctx.edge_conditions.get(&(*pred_id, current_block_id)) {
                    if let Some(z3_dest) = ctx.z3_bvs.get(dest) {
                        if let Some(z3_src) = ctx.z3_bvs.get(src_val) {
                            ctx.solver.assert(edge_cond.implies(z3_dest.eq(z3_src)));
                        }
                    } else if let Some(z3_dest) = ctx.z3_ints.get(dest) {
                        if let Some(z3_src) = ctx.z3_ints.get(src_val) {
                            ctx.solver.assert(edge_cond.implies(z3_dest.eq(z3_src)));
                        }
                    } else if let Some(z3_dest) = ctx.z3_floats.get(dest) {
                        if let Some(z3_src) = ctx.z3_floats.get(src_val) {
                            ctx.solver.assert(edge_cond.implies(z3_dest.eq(z3_src)));
                        }
                    } else if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                        if let Some(z3_src) = ctx.z3_arrays.get(src_val) {
                            ctx.solver.assert(edge_cond.implies(z3_dest.eq(z3_src)));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_reachable(ctx: &TranslationContext, start: BlockId, target: BlockId) -> bool {
    let mut visited = std::collections::HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == target {
            return true;
        }
        if visited.contains(&current) {
            continue;
        }
        visited.insert(current);

        // Find all blocks reachable from 'current'
        for block in &ctx.func.blocks {
            if block.id == current {
                if let Some(last_inst) = block.instructions.last() {
                    match &last_inst.kind {
                        InstructionKind::Jump(t) => {
                            queue.push_back(*t);
                        }
                        InstructionKind::Branch(_, t, f) => {
                            queue.push_back(*t);
                            queue.push_back(*f);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    false
}
