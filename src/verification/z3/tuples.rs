use super::TranslationContext;
use crate::ssa::ir::{Instruction, InstructionKind};
use z3::ast::Bool;

pub fn translate(
    ctx: &mut TranslationContext,
    inst: &Instruction,
    path_cond: &Bool,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::TupleCreate(dest, elts) => {
            ctx.tuple_mappings.insert(*dest, elts.clone());
        }
        InstructionKind::TupleExtract(dest, tuple_val, idx) => {
            if let Some(elts) = ctx.tuple_mappings.get(tuple_val) {
                if let Some(src_val) = elts.get(*idx) {
                    // Link the dest Z3 value to the source Z3 value
                    if let (Some(z3_dest), Some(z3_src)) =
                        (ctx.z3_bvs.get(dest), ctx.z3_bvs.get(src_val))
                    {
                        ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
                    } else if let (Some(z3_dest), Some(z3_src)) =
                        (ctx.z3_ints.get(dest), ctx.z3_ints.get(src_val))
                    {
                        ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
                    } else if let (Some(z3_dest), Some(z3_src)) =
                        (ctx.z3_floats.get(dest), ctx.z3_floats.get(src_val))
                    {
                        ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
                    } else if let (Some(z3_dest), Some(z3_src)) =
                        (ctx.z3_arrays.get(dest), ctx.z3_arrays.get(src_val))
                    {
                        ctx.solver.assert(path_cond.implies(z3_dest.eq(z3_src)));
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}
