use super::{update_block_condition, TranslationContext};
use crate::ssa::ir::{BlockId, Instruction, InstructionKind};
use z3::ast::{Ast, Bool, Int};

pub fn translate<'ctx>(
    ctx: &mut TranslationContext<'ctx>,
    inst: &Instruction,
    path_cond: &Bool<'ctx>,
    current_block_id: BlockId,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::Branch(cond, t_block, f_block) => {
            let z3_cond = ctx
                .z3_ints
                .get(cond)
                .ok_or_else(|| format!("Value {} not modeled in Z3 (required for branch)", cond))?;
            let zero = Int::from_i64(ctx.ctx, 0);
            let cond_is_true = z3_cond._eq(&zero).not();

            let true_cond = Bool::and(ctx.ctx, &[path_cond, &cond_is_true]);
            let false_cond = Bool::and(ctx.ctx, &[path_cond, &cond_is_true.not()]);

            update_block_condition(
                ctx.ctx,
                &mut ctx.block_conditions,
                *t_block,
                true_cond.clone(),
            );
            update_block_condition(
                ctx.ctx,
                &mut ctx.block_conditions,
                *f_block,
                false_cond.clone(),
            );
            ctx.edge_conditions
                .insert((current_block_id, *t_block), true_cond);
            ctx.edge_conditions
                .insert((current_block_id, *f_block), false_cond);
        }
        InstructionKind::Jump(target) => {
            update_block_condition(
                ctx.ctx,
                &mut ctx.block_conditions,
                *target,
                path_cond.clone(),
            );
            ctx.edge_conditions
                .insert((current_block_id, *target), path_cond.clone());
        }
        InstructionKind::Phi(dest, mappings) => {
            if let Some(z3_dest) = ctx.z3_ints.get(dest) {
                for (pred_id, src_val) in mappings {
                    if let Some(edge_cond) = ctx.edge_conditions.get(&(*pred_id, current_block_id))
                    {
                        if let Some(z3_src) = ctx.z3_ints.get(src_val) {
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
