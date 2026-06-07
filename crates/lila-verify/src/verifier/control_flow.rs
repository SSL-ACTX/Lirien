use super::TranslationContext;
use lila_ir::ir::{BlockId, Instruction, InstructionKind};
use std::collections::VecDeque;

pub fn translate<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &mut TranslationContext<'_, B>,
    inst: &Instruction,
    path_cond: &B::Bool,
    current_block_id: BlockId,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::Branch(cond, t_block, f_block) => {
            if let Some(z3_cond) = ctx.z3_bvs.get(cond).cloned() {
                let bit_width = ctx.func.get_type(*cond).int_bit_width().unwrap_or(1);
                let one = ctx.backend.bv_from_i64(1, bit_width);
                let cond_is_true = ctx.backend.bv_eq(&z3_cond, &one);

                let true_cond = ctx.backend.bool_and(&[path_cond, &cond_is_true]);
                let edge_t = ctx
                    .edge_conditions
                    .get(&(current_block_id, *t_block))
                    .unwrap()
                    .clone();
                let __tmp_t = ctx.backend.bool_eq(&edge_t, &true_cond);
                ctx.backend.assert(&__tmp_t);

                let cond_is_false = ctx.backend.bool_not(&cond_is_true);
                let false_cond = ctx.backend.bool_and(&[path_cond, &cond_is_false]);
                let edge_f = ctx
                    .edge_conditions
                    .get(&(current_block_id, *f_block))
                    .unwrap()
                    .clone();
                let __tmp_f = ctx.backend.bool_eq(&edge_f, &false_cond);
                ctx.backend.assert(&__tmp_f);
            }
        }
        InstructionKind::Jump(target) => {
            let edge_j = ctx
                .edge_conditions
                .get(&(current_block_id, *target))
                .unwrap()
                .clone();
            let __tmp = ctx.backend.bool_eq(&edge_j, path_cond);
            ctx.backend.assert(&__tmp);
        }
        InstructionKind::Match(selector, cases, default, is_strict) => {
            if let Some(z3_selector) = ctx.z3_bvs.get(selector).cloned() {
                let mut handled_conds = Vec::new();

                for (tag, target) in cases {
                    let expected_tag = ctx.backend.bv_from_i64(*tag as i64, 8);
                    let is_match = ctx.backend.bv_eq(&z3_selector, &expected_tag);
                    handled_conds.push(is_match.clone());

                    let true_cond = ctx.backend.bool_and(&[path_cond, &is_match]);
                    let edge_t = ctx
                        .edge_conditions
                        .get(&(current_block_id, *target))
                        .unwrap()
                        .clone();
                    let __tmp = ctx.backend.bool_eq(&edge_t, &true_cond);
                    ctx.backend.assert(&__tmp);
                }

                let any_matched = ctx
                    .backend
                    .bool_or(&handled_conds.iter().collect::<Vec<_>>());
                let none_matched = ctx.backend.bool_not(&any_matched);
                let default_cond = ctx.backend.bool_and(&[path_cond, &none_matched]);

                if *is_strict {
                    // Prove that none_matched is impossible under path_cond
                    ctx.backend.push();
                    ctx.backend.assert(&default_cond);
                    if ctx.backend.check() != Ok(false) {
                        let loc_info = inst
                            .location
                            .map(|l| format!(" at {}", l))
                            .unwrap_or_default();
                        return Err(format!(
                            "Non-exhaustive match detected: some ADT variants are not handled{}",
                            loc_info
                        ));
                    }
                    ctx.backend.pop(1);
                }

                let edge_d = ctx
                    .edge_conditions
                    .get(&(current_block_id, *default))
                    .unwrap()
                    .clone();
                let __tmp = ctx.backend.bool_eq(&edge_d, &default_cond);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::Phi(dest, incoming) => {
            for (incoming_block, incoming_val) in incoming {
                if !is_reachable(ctx, *incoming_block, current_block_id) {
                    continue; // Skip mathematically unreachable paths
                }

                let edge_cond = ctx
                    .edge_conditions
                    .get(&(*incoming_block, current_block_id))
                    .unwrap()
                    .clone();

                if let (Some(z3_dest), Some(z3_src)) = (
                    ctx.z3_bvs.get(dest).cloned(),
                    ctx.z3_bvs.get(incoming_val).cloned(),
                ) {
                    let __inner = ctx.backend.bv_eq(&z3_dest, &z3_src);
                    let __tmp = ctx.backend.bool_implies(&edge_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                } else if let (Some(z3_dest), Some(z3_src)) = (
                    ctx.z3_floats.get(dest).cloned(),
                    ctx.z3_floats.get(incoming_val).cloned(),
                ) {
                    let __inner = ctx.backend.float_eq(&z3_dest, &z3_src);
                    let __tmp = ctx.backend.bool_implies(&edge_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                } else if let (Some(z3_dest), Some(z3_src)) = (
                    ctx.z3_ints.get(dest).cloned(),
                    ctx.z3_ints.get(incoming_val).cloned(),
                ) {
                    let __inner = ctx.backend.int_eq(&z3_dest, &z3_src);
                    let __tmp = ctx.backend.bool_implies(&edge_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                } else if let (Some(z3_dest), Some(z3_src)) = (
                    ctx.z3_arrays.get(dest).cloned(),
                    ctx.z3_arrays.get(incoming_val).cloned(),
                ) {
                    let __inner = ctx.backend.array_eq(&z3_dest, &z3_src);
                    let __tmp = ctx.backend.bool_implies(&edge_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_reachable<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &TranslationContext<'_, B>,
    start: BlockId,
    target: BlockId,
) -> bool {
    let mut visited = std::collections::HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == target {
            return true;
        }
        if !visited.insert(current) {
            continue;
        }

        // Add all successors of current
        for (edge_src, edge_dst) in ctx.edge_conditions.keys() {
            if *edge_src == current {
                queue.push_back(*edge_dst);
            }
        }
    }

    false
}
