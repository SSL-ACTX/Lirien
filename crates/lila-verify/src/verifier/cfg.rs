use crate::backend::SolverBackend;
use crate::verifier::TranslationContext;
use lila_ir::ir::InstructionKind;

pub fn assert_cfg_constraints<
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
    // 1. Declare Booleans for all blocks and known edges
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

    // 2. Assert Structural CFG Constraints
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
