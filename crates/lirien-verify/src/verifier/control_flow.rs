use super::TranslationContext;
use lirien_ir::ir::{BlockId, Instruction, InstructionKind, Type, Value};
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
                    if ctx.backend.check()? {
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
            let is_loop_header = incoming.iter().any(|(pred, _)| {
                is_reachable(ctx, current_block_id, *pred)
            });

            if is_loop_header {
                // This is a loop header!
                // 1. Separate incoming edges into entries and backedges.
                let mut entries = Vec::new();
                let mut backedges = Vec::new();
                for (pred, val) in incoming {
                    if is_reachable(ctx, current_block_id, *pred) {
                        backedges.push((*pred, *val));
                    } else {
                        entries.push((*pred, *val));
                    }
                }

                // 2. Assert that program flow reaching the loop header must have entered via an entry edge
                let mut entry_conds = Vec::new();
                for (incoming_block, _) in &entries {
                    let edge_cond = ctx
                        .edge_conditions
                        .get(&(*incoming_block, current_block_id))
                        .unwrap()
                        .clone();
                    entry_conds.push(edge_cond);
                }
                if !entry_conds.is_empty() {
                    let entry_refs: Vec<&z3::ast::Bool> = entry_conds.iter().collect();
                    let or_entries = ctx.backend.bool_or(&entry_refs);
                    let __tmp = ctx.backend.bool_implies(path_cond, &or_entries);
                    ctx.backend.assert(&__tmp);
                }

                // 3. Translate entry edges as usual (these initialize the loop variables).
                for (incoming_block, incoming_val) in &entries {
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

                // 3. For back-edges: do NOT assert the direct dest == incoming_val equality.
                // Instead, we will assert loop invariants and check their preservation.
                let dest_ty = ctx.func.get_type(*dest);
                let unwrapped_ty = unwrap_refined(&dest_ty);
                let is_buffer_or_array = matches!(unwrapped_ty, Type::Buffer(_) | Type::Array(..));

                if is_buffer_or_array {
                    // For buffers/arrays, the length is invariant.
                    // Assert: at the loop header, the length equals the initial length from the entry.
                    if let Some((_, init_val)) = entries.first() {
                        tracing::debug!(target: "lirien::verify", "Found buffer/array loop variable v{} with init_val v{}", dest.0, init_val.0);
                        if let (Some(z3_dest), Some(z3_init)) = (
                            ctx.z3_bvs.get(dest).cloned(),
                            ctx.z3_bvs.get(init_val).cloned(),
                        ) {
                            tracing::debug!(target: "lirien::verify", "Asserting length invariant: len(v{}) == len(v{})", dest.0, init_val.0);
                            let __inner = ctx.backend.bv_eq(&z3_dest, &z3_init);
                            let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                            ctx.backend.assert(&__tmp);
                        } else {
                            tracing::debug!(target: "lirien::verify", "  Failed to find z3_bvs for length invariant: dest v{} has len? {}, init v{} has len? {}",
                                dest.0, ctx.z3_bvs.contains_key(dest), init_val.0, ctx.z3_bvs.contains_key(init_val));
                        }
                    }
                }

                // Let's find other Phi nodes in this block to detect equality relations.
                let block = ctx.func.blocks.iter().find(|b| b.id == current_block_id).unwrap();
                for other_inst in &block.instructions {
                    if let InstructionKind::Phi(other_dest, other_incoming) = &other_inst.kind {
                        if other_dest.0 < dest.0
                            && detect_equality(ctx, current_block_id, *dest, incoming, *other_dest, other_incoming)
                        {
                            // We proved dest == other_dest is an invariant!
                            if let (Some(z3_dest), Some(z3_other)) = (
                                ctx.z3_bvs.get(dest).cloned(),
                                ctx.z3_bvs.get(other_dest).cloned(),
                            ) {
                                let __inner = ctx.backend.bv_eq(&z3_dest, &z3_other);
                                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                                ctx.backend.assert(&__tmp);

                                // Push safety check for each back-edge: next_val == next_other_val
                                for (back_pred, next_val) in &backedges {
                                    if let Some(next_other_val) = other_incoming.get(back_pred) {
                                        if let (Some(z3_next), Some(z3_next_other)) = (
                                            ctx.z3_bvs.get(next_val).cloned(),
                                            ctx.z3_bvs.get(next_other_val).cloned(),
                                        ) {
                                            let back_edge_cond = ctx
                                                .edge_conditions
                                                .get(&(*back_pred, current_block_id))
                                                .unwrap()
                                                .clone();
                                            let eq = ctx.backend.bv_eq(&z3_next, &z3_next_other);
                                            let violation = ctx.backend.bool_not(&eq);
                                            ctx.safety_checks.push(crate::verifier::SafetyCheck {
                                                path_cond: back_edge_cond,
                                                violation_cond: violation,
                                                error_message: format!(
                                                    "Loop invariant (v{} == v{}) not preserved on back-edge from b{}",
                                                    dest.0, other_dest.0, back_pred.0
                                                ),
                                                location: inst.location,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Not a loop header, process all incoming edges normally.
                for (incoming_block, incoming_val) in incoming {
                    if !is_reachable(ctx, *incoming_block, current_block_id) {
                        continue;
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
        }
        _ => {}
    }
    Ok(())
}

fn unwrap_refined(ty: &Type) -> &Type {
    let mut curr = ty;
    while let Type::Refined(inner, _) = curr {
        curr = inner;
    }
    curr
}

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static INSTRUCTION_CACHE: RefCell<HashMap<Value, InstructionKind>> = RefCell::new(HashMap::new());
    static CACHED_FUNC_NAME: RefCell<String> = const { RefCell::new(String::new()) };

    static CFG_SUCCESSORS: RefCell<HashMap<BlockId, Vec<BlockId>>> = RefCell::new(HashMap::new());
    static REACHABILITY_CACHE: RefCell<HashMap<(BlockId, BlockId), bool>> = RefCell::new(HashMap::new());
    static CACHED_FUNC_NAME_REACH: RefCell<String> = const { RefCell::new(String::new()) };
}

fn get_cached_instruction<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &TranslationContext<'_, B>,
    val: Value,
) -> Option<InstructionKind> {
    INSTRUCTION_CACHE.with(|cache| {
        CACHED_FUNC_NAME.with(|cached_name| {
            let mut cache = cache.borrow_mut();
            let mut cached_name = cached_name.borrow_mut();

            if *cached_name != ctx.func.name {
                cache.clear();
                *cached_name = ctx.func.name.clone();
                for block in &ctx.func.blocks {
                    for inst in &block.instructions {
                        if let Some(def) = inst.get_def() {
                            cache.insert(def, inst.kind.clone());
                        }
                    }
                }
            }

            cache.get(&val).cloned()
        })
    })
}

fn get_const_int<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &TranslationContext<'_, B>,
    val: Value,
) -> Option<i64> {
    if let Some(InstructionKind::ConstInt(_, v)) = get_cached_instruction(ctx, val) {
        Some(v)
    } else {
        None
    }
}

fn get_instruction<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &TranslationContext<'_, B>,
    val: Value,
) -> Option<InstructionKind> {
    get_cached_instruction(ctx, val)
}

fn detect_equality<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &TranslationContext<'_, B>,
    current_block_id: BlockId,
    dest_a: Value,
    incoming_a: &std::collections::HashMap<BlockId, Value>,
    dest_b: Value,
    incoming_b: &std::collections::HashMap<BlockId, Value>,
) -> bool {
    tracing::debug!(target: "lirien::verify", "detect_equality comparing v{} and v{}", dest_a.0, dest_b.0);
    if incoming_a.len() != incoming_b.len() {
        tracing::debug!(target: "lirien::verify", "  Length mismatch: {} vs {}", incoming_a.len(), incoming_b.len());
        return false;
    }

    let mut entries_a = Vec::new();
    let mut backedges_a = Vec::new();
    for (pred, val) in incoming_a {
        if is_reachable(ctx, current_block_id, *pred) {
            backedges_a.push((*pred, *val));
        } else {
            entries_a.push((*pred, *val));
        }
    }

    let mut entries_b = Vec::new();
    let mut backedges_b = Vec::new();
    for (pred, val) in incoming_b {
        if is_reachable(ctx, current_block_id, *pred) {
            backedges_b.push((*pred, *val));
        } else {
            entries_b.push((*pred, *val));
        }
    }

    if backedges_a.is_empty() {
        tracing::debug!(target: "lirien::verify", "  No backedges found for v{}", dest_a.0);
        return false;
    }

    for (pred_a, val_a) in &entries_a {
        if let Some((_, val_b)) = entries_b.iter().find(|(pred_b, _)| pred_b == pred_a) {
            if *val_a == *val_b {
                continue;
            }
            if let (Some(c_a), Some(c_b)) = (get_const_int(ctx, *val_a), get_const_int(ctx, *val_b)) {
                if c_a == c_b {
                    continue;
                }
            }
            tracing::debug!(target: "lirien::verify", "  Entry mismatch on b{}: v{} vs v{}", pred_a.0, val_a.0, val_b.0);
            return false;
        } else {
            tracing::debug!(target: "lirien::verify", "  Missing entry predecessor b{} in b", pred_a.0);
            return false;
        }
    }

    for (pred_a, val_a) in &backedges_a {
        if let Some((_, val_b)) = backedges_b.iter().find(|(pred_b, _)| pred_b == pred_a) {
            if *val_a == dest_a && *val_b == dest_b {
                continue;
            }

            let def_a = get_instruction(ctx, *val_a);
            let def_b = get_instruction(ctx, *val_b);

            if let (Some(inst_a), Some(inst_b)) = (def_a, def_b) {
                match (&inst_a, &inst_b) {
                    (InstructionKind::Add(_d_a, l_a, r_a), InstructionKind::Add(_d_b, l_b, r_b)) => {
                        let (_var_a, step_a) = if *l_a == dest_a { (*l_a, *r_a) } else if *r_a == dest_a { (*r_a, *l_a) } else {
                            tracing::debug!(target: "lirien::verify", "  Add l_a/r_a mismatch for dest_a v{}", dest_a.0);
                            return false;
                        };
                        let (_var_b, step_b) = if *l_b == dest_b { (*l_b, *r_b) } else if *r_b == dest_b { (*r_b, *l_b) } else {
                            tracing::debug!(target: "lirien::verify", "  Add l_b/r_b mismatch for dest_b v{}", dest_b.0);
                            return false;
                        };

                        if step_a == step_b {
                            continue;
                        }
                        if let (Some(s_a), Some(s_b)) = (get_const_int(ctx, step_a), get_const_int(ctx, step_b)) {
                            if s_a == s_b {
                                continue;
                            }
                        }
                        tracing::debug!(target: "lirien::verify", "  Add step mismatch: step_a v{} vs step_b v{}", step_a.0, step_b.0);
                    }
                    (InstructionKind::Sub(_d_a, l_a, r_a), InstructionKind::Sub(_d_b, l_b, r_b)) => {
                        if *l_a == dest_a && *l_b == dest_b {
                            if *r_a == *r_b {
                                continue;
                            }
                            if let (Some(s_a), Some(s_b)) = (get_const_int(ctx, *r_a), get_const_int(ctx, *r_b)) {
                                if s_a == s_b {
                                    continue;
                                }
                            }
                            tracing::debug!(target: "lirien::verify", "  Sub step mismatch: r_a v{} vs r_b v{}", r_a.0, r_b.0);
                        } else {
                            tracing::debug!(target: "lirien::verify", "  Sub left-hand side mismatch: l_a v{} vs l_b v{}", l_a.0, l_b.0);
                        }
                    }
                    _ => {
                        tracing::debug!(target: "lirien::verify", "  Instruction type mismatch or unsupported: inst_a={:?}, inst_b={:?}", inst_a, inst_b);
                    }
                }
            } else {
                tracing::debug!(target: "lirien::verify", "  Failed to get instructions for v{} or v{}", val_a.0, val_b.0);
            }
            return false;
        } else {
            tracing::debug!(target: "lirien::verify", "  Missing backedge predecessor b{} in b", pred_a.0);
            return false;
        }
    }

    tracing::debug!(target: "lirien::verify", "  Equality detected successfully!");
    true
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
    CFG_SUCCESSORS.with(|successors_cell| {
        REACHABILITY_CACHE.with(|reach_cell| {
            CACHED_FUNC_NAME_REACH.with(|name_cell| {
                let mut successors = successors_cell.borrow_mut();
                let mut reach = reach_cell.borrow_mut();
                let mut name = name_cell.borrow_mut();

                if *name != ctx.func.name {
                    successors.clear();
                    reach.clear();
                    *name = ctx.func.name.clone();
                    for &(edge_src, edge_dst) in ctx.edge_conditions.keys() {
                        successors.entry(edge_src).or_default().push(edge_dst);
                    }
                }

                if let Some(&res) = reach.get(&(start, target)) {
                    return res;
                }

                let mut visited = std::collections::HashSet::new();
                let mut queue = VecDeque::new();
                queue.push_back(start);
                let mut found = false;

                while let Some(current) = queue.pop_front() {
                    if current == target {
                        found = true;
                        break;
                    }
                    if !visited.insert(current) {
                        continue;
                    }

                    if let Some(succs) = successors.get(&current) {
                        for &succ in succs {
                            queue.push_back(succ);
                        }
                    }
                }

                reach.insert((start, target), found);
                found
            })
        })
    })
}
