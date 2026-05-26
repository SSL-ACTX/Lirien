use super::TranslationContext;
use crate::ssa::ir::{BlockId, Instruction, InstructionKind};
use z3::ast::{Ast, Bool, Int, BV};

pub fn translate<'ctx>(
    ctx: &mut TranslationContext<'ctx>,
    inst: &Instruction,
    path_cond: &Bool<'ctx>,
    current_block_id: BlockId,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::Branch(cond, t_block, f_block) => {
            let cond_is_true = if let Some(z3_bv_cond) = ctx.z3_bvs.get(cond) {
                let bit_width = ctx.func.get_type(*cond).int_bit_width().unwrap_or(64);
                let zero = BV::from_i64(ctx.ctx, 0, bit_width);
                z3_bv_cond._eq(&zero).not()
            } else if let Some(z3_int_cond) = ctx.z3_ints.get(cond) {
                let zero = Int::from_i64(ctx.ctx, 0);
                z3_int_cond._eq(&zero).not()
            } else {
                return Err(format!(
                    "Value {} not modeled in Z3 (required for branch)",
                    cond
                ));
            };

            let true_cond = Bool::and(ctx.ctx, &[path_cond, &cond_is_true]);
            let false_cond = Bool::and(ctx.ctx, &[path_cond, &cond_is_true.not()]);

            if let Some(edge_t) = ctx.edge_conditions.get(&(current_block_id, *t_block)) {
                ctx.solver.assert(&edge_t._eq(&true_cond));
            }
            if let Some(edge_f) = ctx.edge_conditions.get(&(current_block_id, *f_block)) {
                ctx.solver.assert(&edge_f._eq(&false_cond));
            }
        }
        InstructionKind::Jump(target) => {
            if let Some(edge_j) = ctx.edge_conditions.get(&(current_block_id, *target)) {
                ctx.solver.assert(&edge_j._eq(path_cond));
            }
        }
        InstructionKind::Phi(dest, mappings) => {
            if let Some(z3_dest) = ctx.z3_bvs.get(dest) {
                for (pred_id, src_val) in mappings {
                    if let Some(edge_cond) = ctx.edge_conditions.get(&(*pred_id, current_block_id))
                    {
                        if let Some(z3_src) = ctx.z3_bvs.get(src_val) {
                            ctx.solver.assert(&edge_cond.implies(&z3_dest._eq(z3_src)));
                        }
                    }
                }
            } else if let Some(z3_dest) = ctx.z3_ints.get(dest) {
                for (pred_id, src_val) in mappings {
                    if let Some(edge_cond) = ctx.edge_conditions.get(&(*pred_id, current_block_id))
                    {
                        if let Some(z3_src) = ctx.z3_ints.get(src_val) {
                            ctx.solver.assert(&edge_cond.implies(&z3_dest._eq(z3_src)));
                        }
                    }
                }
            } else if let Some(z3_dest) = ctx.z3_reals.get(dest) {
                for (pred_id, src_val) in mappings {
                    if let Some(edge_cond) = ctx.edge_conditions.get(&(*pred_id, current_block_id))
                    {
                        if let Some(z3_src) = ctx.z3_reals.get(src_val) {
                            ctx.solver.assert(&edge_cond.implies(&z3_dest._eq(z3_src)));
                        }
                    }
                }
            } else if let Some(z3_dest) = ctx.z3_arrays.get(dest) {
                for (pred_id, src_val) in mappings {
                    if let Some(edge_cond) = ctx.edge_conditions.get(&(*pred_id, current_block_id))
                    {
                        if let Some(z3_src) = ctx.z3_arrays.get(src_val) {
                            ctx.solver.assert(&edge_cond.implies(&z3_dest._eq(z3_src)));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}
